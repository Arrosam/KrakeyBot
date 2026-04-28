"""ThreadedDashboardServer — uvicorn in its own daemon thread.

Used by the dashboard plugin's ``build_sensory`` factory so the
server can start at plugin-registration time. Plugin factories are
called synchronously during ``Runtime.__init__``, before any
asyncio loop is running, so the server can't share the runtime's
loop the way the regular ``DashboardServer`` does. Solution: run
uvicorn in a daemon background thread with its own asyncio loop.

Lifecycle: the daemon flag means the thread dies at process exit
*if* nobody stopped it explicitly. ``stop()`` is the graceful path
— sets uvicorn's ``should_exit`` and joins the thread so WS clients
get clean close frames and in-flight HTTP requests finish. The
plugin's ``WebChatSensory.stop()`` calls it during runtime shutdown
(``buffer.stop_all()`` in ``Runtime.run()``'s ``finally``).

Cross-thread concerns:
  * ``WebChatSensory.push_user_message`` (called from the chat WS
    handler in this thread) is responsible for shipping stimuli to
    runtime's queue on runtime's loop. It captures runtime's loop
    in its ``start()`` and uses ``run_coroutine_threadsafe`` for the
    push. The server itself stays loop-agnostic.

The bind happens in the calling thread (so OSError "port in use"
surfaces synchronously); the bound socket is then handed off to
uvicorn in the worker thread.
"""
from __future__ import annotations

import asyncio
import socket
import threading
import time

import uvicorn
from fastapi import FastAPI


class ThreadedDashboardServer:
    """Run uvicorn on a private event loop in a daemon thread.

    ``port=0`` lets the OS pick an ephemeral port; after ``start()``
    returns, ``self.port`` reflects the bound port.
    """

    def __init__(self, app: FastAPI, *, host: str = "127.0.0.1",
                  port: int = 8765, log_level: str = "warning"):
        self._app = app
        self._host = host
        self._requested_port = port
        self._log_level = log_level
        self._thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None
        self._startup_error: BaseException | None = None
        self.port: int | None = None

    def start(self) -> None:
        """Bind the port, spawn the daemon thread, block until uvicorn
        reports started (or raise)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            sock.bind((self._host, self._requested_port))
        except OSError:
            sock.close()
            raise
        self.port = sock.getsockname()[1]

        config = uvicorn.Config(
            self._app, log_level=self._log_level,
            access_log=False, lifespan="off",
        )
        self._server = uvicorn.Server(config)

        def _run() -> None:
            try:
                asyncio.run(self._server.serve(sockets=[sock]))
            except BaseException as e:  # noqa: BLE001
                self._startup_error = e

        self._thread = threading.Thread(
            target=_run, daemon=True,
            name=f"dashboard-server-{self.port}",
        )
        self._thread.start()

        # Poll for uvicorn's "started" flag so callers can rely on
        # the server actually accepting connections by the time start()
        # returns.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self._server.started:
                return
            if self._startup_error is not None:
                raise self._startup_error
            time.sleep(0.02)
        raise RuntimeError(
            "dashboard server failed to start within 2s "
            f"(host={self._host}, port={self.port})"
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal uvicorn to exit gracefully and join the daemon thread.

        Synchronous (no asyncio dependency) so callers from any context
        can wait for the server to actually be down. The plugin's
        ``WebChatSensory.stop()`` wraps this in ``asyncio.to_thread``
        so the runtime loop isn't blocked while uvicorn finishes its
        WS close frames + in-flight requests.

        Idempotent: safe to call when never started or already stopped.
        Past the timeout, the daemon flag still ensures the thread
        eventually dies with the process — this method just gives
        uvicorn a clean exit path when shutdown is orderly.
        """
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
        self._server = None
