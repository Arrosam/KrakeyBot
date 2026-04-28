"""DashboardServer \u2014 run uvicorn in the runtime's asyncio loop.

Separated from the app factory (see `app_factory.create_app`) so the
network lifecycle (port binding, graceful shutdown) stays orthogonal
to route composition. Tests that only need the FastAPI app use the
factory directly; tests that need an HTTP server boot this class.
"""
from __future__ import annotations

import asyncio
import socket

import uvicorn
from fastapi import FastAPI


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
