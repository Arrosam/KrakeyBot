"""Concrete EventBroadcaster \u2014 bus \u2192 sockets fan-out."""
from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Awaitable, Callable

from src.dashboard.events.serializer import serialize_event
from src.runtime.events.event_bus import EventBus, _BaseEvent


SocketSend = Callable[[dict[str, Any]], Awaitable[None]]


class EventBroadcaster:
    """One bus \u2192 many sockets. Keeps `history_size` recent serialized
    events for new connections.

    Captures the WS server's event loop on first socket attach so that
    publish() called from any thread/loop can hand the send back to the
    correct loop via `run_coroutine_threadsafe`.
    """

    def __init__(self, bus: EventBus, *, history_size: int = 200):
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
