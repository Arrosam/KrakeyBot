"""Concrete WebChatService — wraps WebChatHistory + a user-message
callback supplied by Runtime at app construction time.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from src.dashboard.web_chat.history import WebChatHistory


UserMessageCallback = Callable[[str, list[dict[str, Any]]], Awaitable[None]]


class RuntimeWebChatService:
    """Adapts a WebChatHistory + a dispatch callback to the
    `WebChatService` Protocol."""

    def __init__(
        self,
        history: WebChatHistory,
        on_user_message: UserMessageCallback | None = None,
    ) -> None:
        self._history = history
        self._on_user_message = on_user_message

    @property
    def history(self) -> WebChatHistory:
        return self._history

    async def receive_user_message(
        self, text: str, attachments: list[dict[str, Any]],
    ) -> None:
        # Persist first — a message is considered received the moment
        # it hits disk, even if downstream dispatch later raises.
        await self._history.append("user", text, attachments=attachments)
        if self._on_user_message is None:
            return
        try:
            await self._on_user_message(text, attachments)
        except Exception:  # noqa: BLE001
            # Dispatch errors must not kill the WS loop; the user's
            # message is already safe on disk.
            pass
