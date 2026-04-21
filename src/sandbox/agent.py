"""Krakey Guest Agent — runs inside the sandbox VM.

Stdlib-only HTTP server that exposes non-idempotent host operations
(exec for Phase S1; more to come) to the Krakey runtime on the host.

Deploy:

    $ python3 sandbox/agent.py \
        --host 0.0.0.0 \
        --port 8765 \
        --token "$SANDBOX_AGENT_TOKEN" \
        --workspace /home/krakey/work

The host-only NIC of the VM should expose the listening port; internet
NIC should NOT (agent must not be reachable from the wider net).

This file is intentionally dependency-free so it can be dropped into a
freshly-provisioned VM without `pip install` anything.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


AGENT_VERSION = "1"


def _guest_os() -> str:
    sysname = platform.system().lower()
    if "darwin" in sysname:
        return "macos"
    if "linux" in sysname:
        return "linux"
    if "windows" in sysname:
        return "windows"
    return sysname or "unknown"


class AgentState:
    def __init__(self, token: str, workspace: Path):
        self.token = token
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)


class AgentHandler(BaseHTTPRequestHandler):
    # Injected by serve()
    state: AgentState = None  # type: ignore[assignment]

    # ---------- common ----------

    def _json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _auth_ok(self) -> bool:
        got = self.headers.get("X-Krakey-Token") or ""
        return got == self.state.token and got != ""

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        # Quieter than default; prefix with agent tag
        print(f"[krakey-agent] {self.address_string()} - " + (format % args))

    # ---------- routes ----------

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            if not self._auth_ok():
                return self._json(401, {"error": "bad token"})
            return self._json(200, {
                "status": "ok",
                "guest_os": _guest_os(),
                "agent_version": AGENT_VERSION,
                "workspace": str(self.state.workspace),
            })
        return self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if not self._auth_ok():
            return self._json(401, {"error": "bad token"})
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as e:
            return self._json(400, {"error": f"bad json: {e}"})

        if self.path == "/exec":
            return self._handle_exec(payload)
        return self._json(404, {"error": "not found"})

    # ---------- /exec ----------

    def _handle_exec(self, p: dict) -> None:
        cmd = p.get("cmd")
        if not cmd or not isinstance(cmd, list):
            return self._json(400, {"error": "cmd must be list[str]"})
        timeout = float(p.get("timeout") or 30.0)
        stdin = p.get("stdin")
        cwd_raw = p.get("cwd")
        # Resolve cwd relative to workspace to keep Krakey out of the rest
        # of the guest FS by default. Absolute paths are allowed (she
        # has root anyway) but workspace-relative is the sane default.
        if cwd_raw:
            cwd_path = Path(cwd_raw)
            if not cwd_path.is_absolute():
                cwd_path = self.state.workspace / cwd_path
        else:
            cwd_path = self.state.workspace
        cwd_path.mkdir(parents=True, exist_ok=True)

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd_path),
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return self._json(200, {
                "exit": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            })
        except subprocess.TimeoutExpired as e:
            return self._json(200, {
                "exit": 124,
                "stdout": (e.stdout or "") if isinstance(e.stdout, str) else "",
                "stderr": (e.stderr or "") if isinstance(e.stderr, str) else "",
                "timeout": True,
            })
        except FileNotFoundError as e:
            return self._json(200, {
                "exit": 127,
                "stdout": "",
                "stderr": f"command not found: {shlex.join(cmd)} ({e})",
            })
        except Exception as e:  # noqa: BLE001
            return self._json(500, {"error": str(e)})


def serve(host: str, port: int, token: str, workspace: Path) -> None:
    handler_cls = AgentHandler
    handler_cls.state = AgentState(token=token, workspace=workspace)
    server = ThreadingHTTPServer((host, port), handler_cls)
    print(f"[krakey-agent] listening on {host}:{port} "
          f"guest_os={_guest_os()} workspace={workspace}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[krakey-agent] shutting down")
        server.server_close()


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Krakey guest agent")
    p.add_argument("--host", default="0.0.0.0",
                   help="bind address (host-only NIC inside VM)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--token",
                   default=os.environ.get("SANDBOX_AGENT_TOKEN", ""),
                   help="shared secret; also read from env "
                        "SANDBOX_AGENT_TOKEN")
    p.add_argument("--workspace", type=Path,
                   default=Path.home() / "krakey-work",
                   help="agent's default cwd for execs")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    if not args.token:
        raise SystemExit("refusing to start without a token; set --token or "
                         "SANDBOX_AGENT_TOKEN env var")
    serve(args.host, args.port, args.token, args.workspace)
