"""Phase 3.F.2: FastAPI dashboard server.

Skeleton: app factory + DashboardServer that starts/stops uvicorn
programmatically inside the runtime's event loop. Subsequent sub-phases
add WS feeds, REST endpoints, and the SPA assets.
"""
from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse


_STATIC_DIR = Path(__file__).parent / "static"


def create_app(*, runtime: Any | None) -> FastAPI:
    """Build the FastAPI app. `runtime` is the live Runtime instance (used
    by later REST endpoints); None is allowed for tests of the skeleton."""
    app = FastAPI(title="Krakey Dashboard",
                    version="0.1",
                    docs_url=None, redoc_url=None)

    @app.get("/api/health")
    async def health():  # noqa: ANN201
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def index():  # noqa: ANN201
        index_path = _STATIC_DIR / "index.html"
        if not index_path.exists():
            return HTMLResponse(_FALLBACK_INDEX, status_code=200)
        return FileResponse(index_path, media_type="text/html")

    return app


class DashboardServer:
    """Run uvicorn in the same asyncio loop as the runtime.

    Use port=0 to let the OS pick an ephemeral port (handy for tests);
    after `.start()`, `self.port` reflects the actually-bound port.
    """

    def __init__(self, app: FastAPI, *, host: str = "127.0.0.1",
                  port: int = 8765, log_level: str = "warning"):
        self._app = app
        self._host = host
        self._requested_port = port
        self._log_level = log_level
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task | None = None
        self.port: int | None = None

    async def start(self) -> None:
        # Pre-bind to detect port-in-use loud-and-early. uvicorn's own
        # bind happens deep in its serve loop; we want the OSError now.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            sock.bind((self._host, self._requested_port))
        except OSError:
            sock.close()
            raise
        self.port = sock.getsockname()[1]
        # Hand the bound socket to uvicorn so we keep the port without
        # a TOCTOU window.
        config = uvicorn.Config(
            self._app, log_level=self._log_level,
            access_log=False, lifespan="off",
        )
        self._server = uvicorn.Server(config)
        # uvicorn's serve(sockets=...) takes pre-bound sockets
        self._task = asyncio.create_task(self._server.serve(sockets=[sock]))
        # Wait for it to come up
        for _ in range(50):
            await asyncio.sleep(0.02)
            if self._server.started:
                return
        # If we get here, server didn't start in time but task may still be alive
        if self._task.done() and self._task.exception() is not None:
            raise self._task.exception()  # type: ignore[misc]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass
            self._task = None
            self._server = None


_FALLBACK_INDEX = """<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>Krakey Dashboard</title>
</head>
<body>
  <h1>Krakey Dashboard</h1>
  <p>Static assets not built yet. The API is up at <code>/api/health</code>.</p>
</body>
</html>
"""
