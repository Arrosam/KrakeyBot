"""Dedicated long-lived event loop thread for MCP I/O.

All MCP async operations — discovery at startup and every execute() call
at runtime — must run on a SINGLE continuously-running event loop so that
stdio subprocess transports (and their background reader tasks) stay alive.

Design
------
``get_loop_thread()`` returns the singleton ``_LoopThread``.  It is created
lazily on first call and lives for the process lifetime (daemon thread, so
process exit doesn't hang).

On Windows, stdio subprocess transports require the Proactor event loop.
We force ``WindowsProactorEventLoopPolicy`` when the platform is Windows
before creating the loop.

Usage::

    lt = get_loop_thread()

    # Submit a coroutine and block synchronously (discovery path):
    result = lt.run_sync(some_coro(), timeout=30.0)

    # Submit a coroutine and get a Future for async awaiting (execute path):
    fut = lt.submit(some_coro())
    result = await asyncio.wrap_future(fut)
"""
from __future__ import annotations

import asyncio
import logging
import sys
import threading
from typing import Any, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


class _LoopThread:
    """One background daemon thread running one event loop forever."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="mcp-connector-loop"
        )
        self._thread.start()
        # Block until the loop is actually running before we hand it out.
        self._ready.wait(timeout=10.0)
        if self._loop is None:
            raise RuntimeError(
                "mcp_connector: loop thread failed to start within 10 s."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, coro) -> "asyncio.Future[Any]":
        """Schedule *coro* on the dedicated loop; return a concurrent Future.

        Safe to call from any thread (including the dedicated loop thread
        itself, though that's unusual).  The returned future can be waited
        on with ``fut.result(timeout=...)`` from a sync thread or wrapped
        via ``asyncio.wrap_future(fut, loop=caller_loop)`` from async code.
        """
        return asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore[arg-type]

    def run_sync(self, coro, *, timeout: float = 30.0) -> Any:
        """Submit *coro* and block until it completes or *timeout* expires.

        On timeout: cancels the coroutine, raises ``TimeoutError``.
        On other exceptions: re-raises them in the calling thread.
        """
        fut = self.submit(coro)
        try:
            return fut.result(timeout=timeout)
        except TimeoutError:
            fut.cancel()
            raise

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Thread target — create the loop and run it forever."""
        # On Windows, stdio subprocess transports require ProactorEventLoop.
        if sys.platform == "win32":
            loop = asyncio.ProactorEventLoop()
        else:
            loop = asyncio.new_event_loop()

        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            # Best-effort cleanup when the loop stops (e.g. process exit).
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:  # noqa: BLE001
                pass
            loop.close()


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_singleton: _LoopThread | None = None
_singleton_lock = threading.Lock()


def get_loop_thread() -> _LoopThread:
    """Return (creating if necessary) the module-level singleton _LoopThread."""
    global _singleton  # noqa: PLW0603
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = _LoopThread()
    return _singleton
