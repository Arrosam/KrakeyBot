"""Live-reload server for the architecture graph.

Watches ``krakey/`` for changes (mtime polling, no extra dependencies),
rebuilds the dependency graph in-process, and pushes Server-Sent
Events to connected browsers. The page-side JS (in
``docs/scripts/build_arch_graph.py``) listens for those events,
refetches the graph payload, and hot-swaps the elements while
preserving the user's expand/collapse, hide, and pan/zoom state — no
full reload.

Usage::

    python docs/scripts/serve_arch_graph.py
    # → http://127.0.0.1:8979/

    python docs/scripts/serve_arch_graph.py --port 9000 --host 0.0.0.0

The static ``docs/architecture-graph.html`` file is unrelated; this
server renders the graph in-memory and never writes to disk. Open the
URL printed on startup in any modern browser.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
# Make the sibling `build_arch_graph.py` importable regardless of how
# this script was launched (we live in `docs/scripts/`).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_arch_graph as bag  # noqa: E402

SRC = ROOT / "krakey"
SKIP_DIRS = {"__pycache__", ".pytest_cache"}

# ---- shared state ----

_state_lock = threading.Lock()
_graph_data: dict = {"nodes": [], "edges": []}
_graph_html: str = ""
_version: int = 0
_version_changed = threading.Event()


def _scan_mtimes() -> dict[Path, float]:
    out: dict[Path, float] = {}
    for p in SRC.rglob("*.py"):
        if any(s in p.parts for s in SKIP_DIRS):
            continue
        try:
            out[p] = p.stat().st_mtime
        except OSError:
            pass
    return out


def _rebuild() -> tuple[dict, str]:
    data = bag.build_graph()
    html = bag._render_html(data)
    return data, html


def _watcher(poll_seconds: float) -> None:
    global _graph_data, _graph_html, _version
    last = _scan_mtimes()
    while True:
        time.sleep(poll_seconds)
        try:
            cur = _scan_mtimes()
        except Exception:  # pragma: no cover
            continue
        if cur == last:
            continue
        try:
            new_data, new_html = _rebuild()
        except Exception as e:
            print(f"[serve] rebuild failed: {e}", flush=True)
            last = cur
            continue
        with _state_lock:
            _graph_data = new_data
            _graph_html = new_html
            _version += 1
            v = _version
        # Wake any SSE clients currently waiting on the event. set() then
        # clear() so future waiters block again until the next change.
        _version_changed.set()
        _version_changed.clear()
        n_nodes = len(new_data["nodes"])
        n_edges = len(new_data["edges"])
        print(
            f"[serve] rebuild v{v}: {n_nodes} nodes, {n_edges} edges",
            flush=True,
        )
        last = cur


# ---- HTTP handler ----


class Handler(BaseHTTPRequestHandler):
    # Quiet the default access-log spam — SSE keepalives would flood it.
    def log_message(self, fmt, *args):  # noqa: D401, ARG002
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            with _state_lock:
                html = _graph_html
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/graph.json":
            with _state_lock:
                payload = {"version": _version, "graph": _graph_data}
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if path == "/events":
            self._stream_events()
            return
        self._send(404, b"", "text/plain")

    def _stream_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        # Disable proxy buffering so events arrive promptly.
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            self.wfile.write(b"retry: 2000\n\n")
            self.wfile.flush()
            last_seen = -1
            while True:
                with _state_lock:
                    v = _version
                if v != last_seen:
                    msg = f"event: update\ndata: {v}\n\n".encode("utf-8")
                    self.wfile.write(msg)
                    self.wfile.flush()
                    last_seen = v
                # Wait for the next rebuild, or send a comment keepalive
                # every 15s so proxies don't time out the connection.
                triggered = _version_changed.wait(timeout=15.0)
                if not triggered:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            return


# ---- entry ----


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8979)
    parser.add_argument(
        "--poll",
        type=float,
        default=1.0,
        help="filesystem polling interval in seconds (default: 1.0)",
    )
    args = parser.parse_args()

    global _graph_data, _graph_html
    print(f"[serve] building initial graph from {SRC}", flush=True)
    _graph_data, _graph_html = _rebuild()
    n_nodes = len(_graph_data["nodes"])
    n_edges = len(_graph_data["edges"])
    print(
        f"[serve] initial graph: {n_nodes} nodes, {n_edges} edges",
        flush=True,
    )

    threading.Thread(
        target=_watcher, args=(args.poll,), daemon=True
    ).start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[serve] listening on {url}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] shutting down", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
