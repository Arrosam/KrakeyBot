"""EventBroadcasterService — Protocol for the events WS route."""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol


SocketSend = Callable[[dict[str, Any]], Awaitable[None]]


class EventBroadcasterService(Protocol):
    """What the /ws/events route needs from the broadcaster.

    Narrower than the concrete `EventBroadcaster` class: the route only
    needs to replay recent history on connect and (un)register itself
    as a socket. Event publishing is handled by the bus subscription
    inside the concrete broadcaster; routes don't publish.
    """

    def recent(self) -> list[dict[str, Any]]: ...

    def add_socket(self, send: SocketSend) -> None: ...

    def remove_socket(self, send: SocketSend) -> None: ...
