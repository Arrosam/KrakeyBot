"""Threaded uvicorn server for the memory engine's web service.

``ThreadedMemoryWebServer`` binds the socket in the calling thread
(so a port-clash surfaces synchronously as an ``OSError``), then
drives uvicorn on a daemon thread.  The ``start()`` method polls
until uvicorn reports it has started.  ``stop()`` sets the
should_exit flag and joins the thread.

Usage::

    server = ThreadedMemoryWebServer(app, host="127.0.0.1", port=8766)
    server.start()   # blocks briefly until uvicorn is ready
    # ... engine runs ...
    server.stop()
"""
from __future__ import annotations

import socket
import threading
import time
import logging

import uvicorn

logger = logging.getLogger(__name__)


class ThreadedMemoryWebServer:
    """Runs a FastAPI ASGI *app* on a daemon thread via uvicorn.

    The socket is bound in the *calling* thread so any ``OSError``
    (e.g. address already in use) surfaces synchronously before
    ``start()`` returns.
    """

    def __init__(self, app, *, host: str = "127.0.0.1", port: int = 8766) -> None:
        self._app = app
        self._host = host
        self._port = port
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None

    def start(self, timeout: float = 10.0) -> None:
        """Bind the socket, start uvicorn on a daemon thread, and wait
        until it reports ``started``.  Raises ``OSError`` immediately if
        the port is already in use."""
        # Bind in the calling thread so port clash surfaces here
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, self._port))
        sock.setblocking(False)
        self._sock = sock

        config = uvicorn.Config(
            app=self._app,
            host=self._host,
            port=self._port,
            log_level="warning",
            loop="asyncio",
        )
        server = uvicorn.Server(config=config)
        self._server = server

        def _run():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # Hand the pre-bound socket to uvicorn
            loop.run_until_complete(server.serve(sockets=[sock]))
            loop.close()

        thread = threading.Thread(target=_run, daemon=True, name="memory-web-server")
        self._thread = thread
        thread.start()

        # Poll until uvicorn marks itself started
        deadline = time.monotonic() + timeout
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)

        if not server.started:
            logger.warning(
                "memory web server did not report started within %.1fs "
                "(host=%s port=%d) — continuing anyway",
                timeout, self._host, self._port,
            )
        else:
            logger.info(
                "memory web server listening on http://%s:%d",
                self._host, self._port,
            )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal uvicorn to exit and join the thread."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        self._server = None
        logger.debug("memory web server stopped")
