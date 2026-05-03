"""Concrete EventBroadcaster \u2014 bus \u2192 sockets fan-out."""
from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Awaitable, Callable

from krakey.plugins.dashboard.events.serializer import serialize_event
from krakey.runtime.events.event_types import _BaseEvent
from krakey.runtime.events.event_bus import EventBus


SocketSend = Callable[[dict[str, Any]], Awaitable[None]]


# Event kinds we DO NOT keep in the reconnect-snapshot ring buffer.
# Right now this is just ``prompt_built``: each one carries the full
# heartbeat prompt as a single string (often 50\u2013200 KB), so saving
# 50+ of them in memory and shipping them all on every WS connect
# blows up the snapshot size to multiple MB. The Prompts tab fetches
# /api/prompts directly when the user opens it (canonical source);
# live new ``prompt_built`` events still flow through the broadcaster
# in real time, they just don't get replayed on reconnect.
_HISTORY_EXCLUDE_KINDS = frozenset({"prompt_built"})


class EventBroadcaster:
    """One bus \u2192 many sockets. Keeps a small ring of recent serialized
    events so new connections can repaint timelines instantly.

    Captures the WS server's event loop on first socket attach so that
    publish() called from any thread/loop can hand the send back to the
    correct loop via ``run_coroutine_threadsafe``.
    """

    def __init__(self, bus: EventBus, *, history_size: int = 50):
        self._bus = bus
        self._recent: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._sockets: list[SocketSend] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        bus.subscribe(self._on_event)

    def recent(self) -> list[dict[str, Any]]:
        return list(self._recent)

    def add_socket(self, send: SocketSend) -> None:
        self._sockets.append(send)
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

    def remove_socket(self, send: SocketSend) -> None:
        try:
            self._sockets.remove(send)
        except ValueError:
            pass

    def _on_event(self, event: _BaseEvent) -> None:
        msg = serialize_event(event)
        # Save to history ONLY if this event kind isn't on the
        # exclude list. Live broadcast still happens for every event.
        if msg.get("kind") not in _HISTORY_EXCLUDE_KINDS:
            self._recent.append(msg)
        if not self._sockets or self._loop is None:
            return
        for send in list(self._sockets):
            try:
                asyncio.run_coroutine_threadsafe(
                    self._safe_send(send, msg), self._loop,
                )
            except RuntimeError:
                # Loop closed; drop the socket.
                self.remove_socket(send)

    async def _safe_send(self, send: SocketSend,
                           msg: dict[str, Any]) -> None:
        try:
            await send(msg)
        except Exception:  # noqa: BLE001
            self.remove_socket(send)
