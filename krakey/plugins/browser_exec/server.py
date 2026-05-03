"""Long-running browser RPC daemon for ``browser_exec``.

This module is **executed inside the target environment**, not by
the krakey runtime. The runtime never imports it directly at
heartbeat time — the dispatch snippet writes a copy of this file's
source into the env's workspace and spawns it as a detached
subprocess on first use, then connects to it over a localhost TCP
RPC for the lifetime of the env (host process for ``local``,
guest VM for ``sandbox``).

Because of that constraint, this file is **stdlib + Playwright
only** — no krakey imports, no third-party HTTP libs. ``http.server``
is sufficient for the RPC; the cadence is one heartbeat at a time
so a single-threaded handler is fine.

Wire protocol::

    POST /rpc
    Headers: X-Browser-Token: <token>
    Body (json): {"op": <name>, "args": {...}}

    Response 200 (op success):
      {"ok": true, "result": {...}, "tabs": [...]}
    Response 200 (op-level error, e.g. tab_id not found):
      {"ok": false, "error": "<msg>", "tabs": [...]}
    Response 401: bad / missing token
    Response 400: malformed body
    Response 500: server bug

Ops:
  - ``list_tabs {}`` — returns ``{tabs}`` only.
  - ``new_tab {start_url, label?}`` — opens a new ``Page``,
    navigates, returns ``{tab_id, url, title}``. Browser is
    launched lazily on the first call (browser/headless from
    server argv); subsequent calls' browser/headless overrides
    are ignored — locked at first launch.
  - ``close_tab {tab_id}`` — closes the page, removes from map.
  - ``operate {tab_id, actions, output, return_screenshot,
    timeout_ms, screenshot_path}`` — runs the action chain on the
    tab, extracts output, returns
    ``{final_url, output, output_format, screenshot_path,
       actions_completed, actions_total}``.

Lifecycle:
  - Started by the dispatch snippet via
    ``subprocess.Popen([python, server.py, --browser=...,
    --headless=..., --workspace=...], detach_flags...)``.
  - Picks an OS-assigned port (bind to ``127.0.0.1:0``) and
    writes ``{port, token, pid}`` to
    ``<workspace>/data/browser_exec/server.json`` atomically
    BEFORE entering the request loop.
  - On Linux: ``start_new_session=True`` is set by the spawner,
    so the daemon is reparented to PID 1 when the env worker
    exits. On Windows: ``CREATE_NEW_PROCESS_GROUP`` plus
    ``DETACHED_PROCESS``.
  - Dies when its host process / guest VM dies. No explicit
    shutdown hook from krakey side.

Crash semantics:
  - Tab map is in-memory only; a server crash = all tabs lost.
    Self sees an empty tab list and re-opens. Disk persistence
    deferred — see plan v2 "Out of scope".
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import socket
import sys
import threading
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Queue
from typing import Any


# ---------------------------------------------------------------------
# Logging — file-based so the daemon doesn't spam the env's stdout
# (that pipe closes when the spawner exits).
# ---------------------------------------------------------------------


def _setup_logging(workspace: Path) -> logging.Logger:
    log_dir = workspace / "data" / "browser_exec"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "server.log"

    logger = logging.getLogger("browser_exec.server")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.FileHandler(log_path, encoding="utf-8")
        h.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        ))
        logger.addHandler(h)
    return logger


# ---------------------------------------------------------------------
# Browser worker — Playwright sync API isn't safe across threads, so
# all browser ops run in a single dedicated worker thread. The HTTP
# handler enqueues a job and waits for the result. One client at a
# time (krakey heartbeats are serial).
# ---------------------------------------------------------------------


_TAB_ID_PREFIX = "tab_"


def _new_tab_id() -> str:
    return _TAB_ID_PREFIX + secrets.token_hex(4)


class BrowserWorker:
    """Holds the Playwright browser + tab map. Single-threaded by
    design (Playwright sync API requires it)."""

    def __init__(
        self, browser_name: str, headless: bool,
        logger: logging.Logger,
    ):
        self._browser_name = browser_name
        self._headless = headless
        self._logger = logger

        # Lazily imported at first ``start`` so the server still
        # writes server.json + opens the listening socket even if
        # Playwright import fails (so the dispatch client gets a
        # clean error response instead of a connection refused).
        self._playwright = None
        self._browser = None
        self._tabs: dict[str, Any] = {}     # tab_id -> Page
        self._labels: dict[str, str] = {}   # tab_id -> label

        self._jobs: Queue = Queue()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="browser_exec.worker",
        )

    def start(self) -> None:
        self._thread.start()

    def submit(self, op: str, args: dict) -> dict:
        """Enqueue an op, block until result. Returns ``{ok, ...}``
        envelope for HTTP handler."""
        result_box: dict = {}
        ev = threading.Event()
        self._jobs.put((op, args, result_box, ev))
        ev.wait()
        return result_box["envelope"]

    # ---- worker thread ----

    def _run(self) -> None:
        # Lazy Playwright import — failure here surfaces as op-level
        # error to the first caller instead of a server crash.
        try:
            from playwright.sync_api import sync_playwright
            self._playwright = sync_playwright().start()
            launcher = getattr(self._playwright, self._browser_name)
            self._browser = launcher.launch(headless=self._headless)
            self._logger.info(
                "browser launched: %s headless=%s",
                self._browser_name, self._headless,
            )
        except Exception as e:  # noqa: BLE001
            self._logger.exception("browser launch failed")
            # Drain pending jobs with a launch-failure error.
            while True:
                op, args, box, ev = self._jobs.get()
                box["envelope"] = self._envelope(
                    ok=False,
                    error=f"browser launch failed: "
                          f"{type(e).__name__}: {e}",
                )
                ev.set()

        while True:
            op, args, box, ev = self._jobs.get()
            try:
                box["envelope"] = self._dispatch(op, args)
            except Exception as e:  # noqa: BLE001
                self._logger.exception("op %r failed", op)
                box["envelope"] = self._envelope(
                    ok=False,
                    error=f"server error in op {op!r}: "
                          f"{type(e).__name__}: {e}",
                )
            ev.set()

    # ---- op dispatch ----

    def _dispatch(self, op: str, args: dict) -> dict:
        if op == "list_tabs":
            return self._envelope(ok=True, result={})
        if op == "new_tab":
            return self._op_new_tab(args)
        if op == "close_tab":
            return self._op_close_tab(args)
        if op == "operate":
            return self._op_operate(args)
        return self._envelope(
            ok=False, error=f"unknown op: {op!r}",
        )

    def _op_new_tab(self, args: dict) -> dict:
        url = args.get("start_url")
        label = args.get("label") or ""
        timeout_ms = int(args.get("timeout_ms") or 30_000)
        if not isinstance(url, str) or not url:
            return self._envelope(ok=False, error="start_url required")

        page = self._browser.new_page()
        try:
            page.goto(url, timeout=timeout_ms)
        except Exception as e:  # noqa: BLE001
            page.close()
            return self._envelope(
                ok=False,
                error=f"goto({url!r}) failed: "
                      f"{type(e).__name__}: {e}",
            )
        tab_id = _new_tab_id()
        self._tabs[tab_id] = page
        self._labels[tab_id] = label
        return self._envelope(
            ok=True,
            result={
                "tab_id": tab_id,
                "url": page.url,
                "title": page.title() or "",
            },
        )

    def _op_close_tab(self, args: dict) -> dict:
        tab_id = args.get("tab_id")
        if tab_id not in self._tabs:
            return self._envelope(
                ok=False, error=f"tab_id {tab_id!r} not found",
            )
        try:
            self._tabs[tab_id].close()
        except Exception:  # noqa: BLE001
            pass  # best-effort close
        self._tabs.pop(tab_id, None)
        self._labels.pop(tab_id, None)
        return self._envelope(ok=True, result={})

    def _op_operate(self, args: dict) -> dict:
        tab_id = args.get("tab_id")
        if tab_id not in self._tabs:
            return self._envelope(
                ok=False, error=f"tab_id {tab_id!r} not found",
            )
        page = self._tabs[tab_id]
        actions = args.get("actions") or []
        timeout_ms = int(args.get("timeout_ms") or 30_000)
        output_fmt = args.get("output") or "a11y"
        screenshot_path = args.get("screenshot_path")

        completed = 0
        for a in actions:
            kind = a.get("action")
            try:
                if kind == "navigate":
                    page.goto(a["url"], timeout=timeout_ms)
                elif kind == "click":
                    page.click(a["selector"], timeout=timeout_ms)
                elif kind == "type":
                    page.fill(
                        a["selector"], a["text"], timeout=timeout_ms,
                    )
                elif kind == "press":
                    page.keyboard.press(a["key"])
                elif kind == "scroll":
                    d = a["direction"]
                    amt = a["amount"]
                    if d == "down":
                        dx, dy = 0, amt
                    elif d == "up":
                        dx, dy = 0, -amt
                    elif d == "right":
                        dx, dy = amt, 0
                    else:
                        dx, dy = -amt, 0
                    page.evaluate(
                        "(args) => window.scrollBy(args[0], args[1])",
                        [dx, dy],
                    )
                elif kind == "wait_for":
                    page.wait_for_selector(
                        a["selector"],
                        timeout=int(a.get("timeout_ms") or timeout_ms),
                    )
                elif kind == "screenshot":
                    sp = screenshot_path
                    if not sp:
                        return self._envelope(
                            ok=False,
                            error="screenshot action requires "
                                  "screenshot_path in args",
                        )
                    os.makedirs(
                        os.path.dirname(sp) or ".", exist_ok=True,
                    )
                    page.screenshot(
                        path=sp,
                        full_page=bool(a.get("full_page", False)),
                    )
                else:
                    return self._envelope(
                        ok=False,
                        error=f"unknown action kind: {kind!r}",
                    )
            except Exception as e:  # noqa: BLE001
                return self._envelope(
                    ok=False,
                    error=f"action {kind!r} (#{completed}) failed: "
                          f"{type(e).__name__}: {e}",
                )
            completed += 1

        # Final-state extraction.
        try:
            if output_fmt == "a11y":
                output = page.accessibility.snapshot()
            elif output_fmt == "text":
                output = page.inner_text("body")
            elif output_fmt == "html":
                output = page.content()
            else:
                return self._envelope(
                    ok=False,
                    error=f"unknown output format: {output_fmt!r}",
                )
        except Exception as e:  # noqa: BLE001
            return self._envelope(
                ok=False,
                error=f"output extraction failed: "
                      f"{type(e).__name__}: {e}",
            )

        return self._envelope(
            ok=True,
            result={
                "final_url":         page.url,
                "output_format":     output_fmt,
                "output":            output,
                "screenshot_path":   screenshot_path,
                "actions_completed": completed,
                "actions_total":     len(actions),
            },
        )

    # ---- envelope ----

    def _envelope(
        self, *, ok: bool, result: dict | None = None,
        error: str | None = None,
    ) -> dict:
        env: dict = {"ok": ok, "tabs": self._tab_list()}
        if ok:
            env["result"] = result or {}
        else:
            env["error"] = error or "unspecified error"
        return env

    def _tab_list(self) -> list[dict]:
        out = []
        for tab_id, page in list(self._tabs.items()):
            try:
                url = page.url
                title = page.title() or ""
            except Exception:  # noqa: BLE001
                # Page may be closed underneath us; skip + reap.
                self._tabs.pop(tab_id, None)
                self._labels.pop(tab_id, None)
                continue
            out.append({
                "id":    tab_id,
                "url":   url,
                "title": title,
                "label": self._labels.get(tab_id, ""),
            })
        return out


# ---------------------------------------------------------------------
# HTTP front-end — single endpoint /rpc, token-authenticated, body
# is JSON {op, args}.
# ---------------------------------------------------------------------


class _RpcHandler(BaseHTTPRequestHandler):
    # Injected by ``serve``.
    worker: BrowserWorker = None  # type: ignore[assignment]
    token: str = ""
    logger: logging.Logger = None  # type: ignore[assignment]

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        self.logger.info("%s %s", self.address_string(), format % args)

    def do_POST(self):  # noqa: N802
        if self.path != "/rpc":
            return self._json(404, {"error": "not found"})
        if not self._auth_ok():
            return self._json(401, {"error": "bad token"})
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as e:
            return self._json(400, {"error": f"bad json: {e}"})

        op = body.get("op")
        args = body.get("args") or {}
        if not isinstance(op, str) or not isinstance(args, dict):
            return self._json(
                400,
                {"error": "body must be {op: str, args: object}"},
            )

        try:
            envelope = self.worker.submit(op, args)
        except Exception as e:  # noqa: BLE001
            self.logger.exception("submit failed")
            return self._json(500, {
                "error": f"submit failed: "
                         f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=5),
            })
        return self._json(200, envelope)

    def _auth_ok(self) -> bool:
        return self.headers.get("X-Browser-Token") == self.token

    def _json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------------
# server.json discovery file
# ---------------------------------------------------------------------


def _server_info_path(workspace: Path) -> Path:
    return workspace / "data" / "browser_exec" / "server.json"


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        # Windows: chmod is best-effort. The token is still random
        # per launch, so an unrelated process inside the env can't
        # forge requests.
        pass


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------


def serve(
    workspace: Path, browser_name: str, headless: bool,
) -> None:
    logger = _setup_logging(workspace)
    logger.info(
        "starting: workspace=%s browser=%s headless=%s pid=%s",
        workspace, browser_name, headless, os.getpid(),
    )

    # Pick port via OS (bind to 0). Token is random per lifetime.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    port = sock.getsockname()[1]
    token = secrets.token_hex(16)

    worker = BrowserWorker(browser_name, headless, logger)
    worker.start()

    handler_cls = _RpcHandler
    handler_cls.worker = worker
    handler_cls.token = token
    handler_cls.logger = logger

    server = ThreadingHTTPServer.__new__(ThreadingHTTPServer)
    # Init manually because we already bound the socket to grab a
    # port early (so we can write server.json before serve_forever
    # blocks).
    server.allow_reuse_address = False
    server.RequestHandlerClass = handler_cls
    server.socket = sock
    server.server_address = sock.getsockname()
    server.timeout = None
    # ThreadingHTTPServer needs ``daemon_threads``.
    server.daemon_threads = True

    info_path = _server_info_path(workspace)
    _atomic_write_json(info_path, {
        "port":    port,
        "token":   token,
        "pid":     os.getpid(),
        "started": datetime.now().isoformat(timespec="seconds"),
    })
    logger.info(
        "ready on 127.0.0.1:%d (server.json=%s)", port, info_path,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down")
    finally:
        try:
            info_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="browser_exec server")
    p.add_argument(
        "--workspace", type=Path, default=Path("workspace"),
        help="env's workspace dir (server.json + log live under "
             "data/browser_exec/ inside this).",
    )
    p.add_argument(
        "--browser", default="chromium",
        choices=("chromium", "firefox", "webkit"),
    )
    p.add_argument(
        "--headless", default="true",
        help="'true' or 'false' — string for argv compat.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    headless = args.headless.lower() != "false"
    try:
        serve(args.workspace, args.browser, headless)
    except Exception:  # noqa: BLE001
        # Last-ditch — make sure the failure shows up somewhere
        # the spawner can find on disk before the process exits.
        try:
            log_path = (
                args.workspace / "data" / "browser_exec" /
                "server.log"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write("\n--- fatal at " + datetime.now().isoformat() + "\n")
                traceback.print_exc(file=f)
        except Exception:
            pass
        sys.exit(1)
