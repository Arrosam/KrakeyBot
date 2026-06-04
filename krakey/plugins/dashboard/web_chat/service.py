"""Concrete WebChatService — wraps WebChatHistory + a user-message
callback supplied by Runtime at app construction time.
"""
from __future__ import annotations

import uuid
from typing import Any, Awaitable, Callable

from krakey.plugins.dashboard.web_chat.history import WebChatHistory


# Callback receives (text, attachments, message_id) and returns True on
# successful push to the runtime buffer, False when the runtime channel
# is offline.
UserMessageCallback = Callable[[str, list[dict[str, Any]], str | None], Awaitable[bool]]


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
    ) -> dict[str, Any]:
        """Receive a user message, determine delivery status, persist, and
        return a status dict.

        Returns ``{"id": message_id, "status": status}`` plus
        ``"reason": reason`` when the status is "failed".
        """
        message_id = str(uuid.uuid4())
        status: str
        reason: str | None = None

        if self._on_user_message is None:
            status = "failed"
            reason = "no_runtime"
        else:
            try:
                ok = await self._on_user_message(text, attachments, message_id)
                if ok:
                    status = "delivered"
                else:
                    status = "failed"
                    reason = "offline"
            except Exception:  # noqa: BLE001
                status = "failed"
                reason = "dispatch_error"

        await self._history.append(
            "user", text,
            attachments=attachments,
            message_id=message_id,
            status=status,
        )

        result: dict[str, Any] = {"id": message_id, "status": status}
        if reason is not None:
            result["reason"] = reason
        return result
