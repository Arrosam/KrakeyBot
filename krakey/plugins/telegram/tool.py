"""Telegram outbound reply tool.

Takes the same `TelegramClient` instance the sensory polls with;
sending the message IS Krakey's real outward chat to a human (the
tool_feedback returned to Self is just a delivery receipt).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from krakey.interfaces.tool import Tool
from krakey.models.stimulus import Stimulus

from .client import TelegramClient


class TelegramReplyTool(Tool):
    def __init__(self, client: TelegramClient, *,
                  default_chat_id: int | None = None):
        self._client = client
        self._default_chat_id = default_chat_id

    @property
    def name(self) -> str:
        return "telegram_reply"

    @property
    def description(self) -> str:
        return ("Send a Telegram message to a chat the bot is in. "
                "params: chat_id (optional, defaults to default_chat_id "
                "from plugin config), text (optional, defaults to the "
                "natural-language intent).")

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "chat_id": "Telegram chat id (defaults to default_chat_id)",
            "text": "message body (defaults to intent)",
        }


    async def execute(self, intent: str,
                        params: dict[str, Any]) -> Stimulus:
        chat_id_raw = params.get("chat_id") or self._default_chat_id
        if chat_id_raw is None:
            return self._stim(
                "No chat_id provided and no default set; "
                "cannot send Telegram reply.",
                adrenalin=True,
            )
        text = (params.get("text") or intent or "").strip()
        if not text:
            return self._stim("Empty message body; nothing sent.")
        try:
            await self._client.send_message(int(chat_id_raw), text)
        except Exception as e:  # noqa: BLE001
            return self._stim(f"Telegram send failed: {e}",
                                adrenalin=True)
        return self._stim(f"Sent to chat {chat_id_raw}.")

    def _stim(self, content: str, *, adrenalin: bool = False) -> Stimulus:
        return Stimulus(
            type="tool_feedback",
            source=f"tool:{self.name}",
            content=content,
            timestamp=datetime.now(),
            adrenalin=adrenalin,
        )
