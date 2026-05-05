"""Integration tests for ``browser_exec.server`` — actually launches
the daemon in a thread and talks to it over HTTP.

Why this file exists: a previous implementation constructed the
``ThreadingHTTPServer`` via ``__new__`` to skip the constructor (so
we could pre-bind the socket to grab the port). That bypassed
``BaseServer.__init__`` which sets up ``_BaseServer__is_shut_down``
(a ``threading.Event``) and ``_BaseServer__shutdown_request``;
``serve_forever`` references both immediately on entry. Result:
``AttributeError`` killed the daemon at startup, every browser_exec
call timed out at the dispatch snippet's 30s "server failed to
start" deadline, and no static test caught it because the failure
only manifested at runtime when ``serve_forever`` was called.

These tests stub ``BrowserWorker`` so Playwright isn't required in
CI but exercise the real HTTP server lifecycle. If
``serve_forever`` ever crashes again, the threading.Thread sees
the exception and the test's deadline-poll for ``server.json``
catches it.
"""
from __future__ import annotations

import http.client
import json
import threading
import time
from pathlib import Path

import pytest

from krakey.plugins.browser_exec import server as server_module


# =====================================================================
# Stub worker — replaces BrowserWorker so we can run serve() without
# importing Playwright (which would require browser binaries in CI).
# =====================================================================


class _StubWorker:
    """Mimics ``BrowserWorker``'s public surface
    (``start`` / ``submit``) without touching Playwright."""

    last: "_StubWorker | None" = None

    def __init__(self, browser_name: str, headless: bool, logger):
        self.browser_name = browser_name
        self.headless = headless
        self.logger = logger
        self.calls: list[tuple[str, dict]] = []
        type(self).last = self

    def start(self) -> None:
        pass

    def submit(self, op: str, args: dict) -> dict:
        self.calls.append((op, dict(args)))
        # Echo the op back so tests can verify the right one was
        # received without hard-coding an envelope shape per op.
        return {
            "ok":     True,
            "tabs":   [],
            "result": {"echoed_op": op, "echoed_args": args},
        }


@pytest.fixture
def stub_worker(monkeypatch):
    monkeypatch.setattr(
        server_module, "BrowserWorker", _StubWorker,
    )
    yield


def _spawn_server(workspace: Path) -> threading.Thread:
    t = threading.Thread(
        target=server_module.serve,
        args=(workspace, "chromium", True),
        name="browser_exec.server-test",
        daemon=True,
    )
    t.start()
    return t


def _wait_for_info(workspace: Path, deadline_s: float = 5.0) -> dict:
    info_path = server_module._server_info_path(workspace)
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        if info_path.exists():
            try:
                return json.loads(info_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # Mid-write, retry.
                pass
        time.sleep(0.05)
    raise AssertionError(
        f"server.json never appeared at {info_path} within "
        f"{deadline_s}s — daemon likely crashed at startup. "
        f"Check {workspace / 'data' / 'browser_exec' / 'server.log'}"
    )


def _post_rpc(
    port: int, token: str, op: str, args: dict | None = None,
    *, content_type: str = "application/json",
):
    body = json.dumps({"op": op, "args": args or {}}).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("POST", "/rpc", body=body, headers={
        "Content-Type":    content_type,
        "Content-Length":  str(len(body)),
        "X-Browser-Token": token,
    })
    resp = conn.getresponse()
    data = resp.read().decode("utf-8", errors="replace")
    conn.close()
    return resp.status, data


# =====================================================================
# Regression test for the __new__ bug
# =====================================================================


def test_server_starts_and_serve_forever_does_not_crash(
    tmp_path: Path, stub_worker,
):
    """The bug: constructing ``ThreadingHTTPServer`` via ``__new__``
    skipped ``BaseServer.__init__``, so ``serve_forever`` raised
    ``AttributeError`` on entry. With the fix in place, the daemon
    must start AND stay alive long enough to write server.json AND
    answer at least one RPC."""
    _spawn_server(tmp_path)
    info = _wait_for_info(tmp_path)

    assert isinstance(info["port"], int) and info["port"] > 0
    assert isinstance(info["token"], str) and len(info["token"]) >= 16
    assert isinstance(info["pid"], int)

    status, body = _post_rpc(info["port"], info["token"], "list_tabs")
    assert status == 200
    payload = json.loads(body)
    assert payload["ok"] is True
    assert payload["result"]["echoed_op"] == "list_tabs"


# =====================================================================
# Auth / protocol checks — exercise the HTTP front-end's other paths
# now that we know it stays alive
# =====================================================================


def test_server_rejects_missing_or_wrong_token(
    tmp_path: Path, stub_worker,
):
    _spawn_server(tmp_path)
    info = _wait_for_info(tmp_path)

    # No token header at all → 401.
    body = json.dumps({"op": "list_tabs", "args": {}}).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", info["port"], timeout=5)
    conn.request("POST", "/rpc", body=body, headers={
        "Content-Type":   "application/json",
        "Content-Length": str(len(body)),
    })
    resp = conn.getresponse()
    resp.read()
    conn.close()
    assert resp.status == 401

    # Wrong token → 401.
    status, _ = _post_rpc(
        info["port"], "wrong-token-value", "list_tabs",
    )
    assert status == 401


def test_server_returns_404_for_unknown_paths(
    tmp_path: Path, stub_worker,
):
    _spawn_server(tmp_path)
    info = _wait_for_info(tmp_path)

    conn = http.client.HTTPConnection("127.0.0.1", info["port"], timeout=5)
    conn.request("POST", "/not-rpc", body=b"{}", headers={
        "Content-Type":    "application/json",
        "Content-Length":  "2",
        "X-Browser-Token": info["token"],
    })
    resp = conn.getresponse()
    resp.read()
    conn.close()
    assert resp.status == 404


def test_server_rejects_malformed_json_body(
    tmp_path: Path, stub_worker,
):
    _spawn_server(tmp_path)
    info = _wait_for_info(tmp_path)

    body = b"not-json"
    conn = http.client.HTTPConnection("127.0.0.1", info["port"], timeout=5)
    conn.request("POST", "/rpc", body=body, headers={
        "Content-Type":    "application/json",
        "Content-Length":  str(len(body)),
        "X-Browser-Token": info["token"],
    })
    resp = conn.getresponse()
    resp.read()
    conn.close()
    assert resp.status == 400


def test_server_rejects_body_with_wrong_shape(
    tmp_path: Path, stub_worker,
):
    """Body must be ``{op: str, args: object}``. Other shapes are
    400 (not crashes)."""
    _spawn_server(tmp_path)
    info = _wait_for_info(tmp_path)

    # missing op
    status, _ = _post_rpc(info["port"], info["token"], op=42)  # type: ignore[arg-type]
    assert status == 400


def test_server_threads_op_through_to_worker(
    tmp_path: Path, stub_worker,
):
    """Args round-trip through HTTP → worker.submit verbatim."""
    _spawn_server(tmp_path)
    info = _wait_for_info(tmp_path)

    args = {"tab_id": "tab_xyz", "actions": [{"action": "click",
                                              "selector": "#go"}]}
    status, body = _post_rpc(
        info["port"], info["token"], "operate", args,
    )
    assert status == 200
    payload = json.loads(body)
    assert payload["ok"] is True
    assert payload["result"]["echoed_op"] == "operate"
    assert payload["result"]["echoed_args"] == args

    worker = _StubWorker.last
    assert worker is not None
    assert worker.calls == [("operate", args)]
