"""WebChatService — Protocol for the chat WS route."""
from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from krakey.plugins.dashboard.web_chat.history import WebChatHistory


class WebChatService(Protocol):
    """What the /ws/chat route needs.

    The route reads history on connect, subscribes for broadcasts, and
    dispatches inbound user messages via `receive_user_message`. All
    three are Protocol surface so the route can be unit-tested without
    a live Runtime: an in-memory fake implements the three and it
    works.
    """

    @property
    def history(self) -> "WebChatHistory": ...

    async def receive_user_message(
        self, text: str, attachments: list[dict[str, Any]],
    ) -> None: ...
