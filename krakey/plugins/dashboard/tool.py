"""Krakey → web chat outbound reply tool and mark-read tool.

No LLM in these tools — Self already wrote the text in [DECISION];
reply tool persists + broadcasts text to connected chat clients.
Mark-read tool reads unread messages and marks them read so the
unread reminder stops.

Lives with the dashboard plugin because the embedded chat UI is part
of the dashboard bundle. Runtime never references these tools by
name.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from krakey.interfaces.tool import Tool
from krakey.models.stimulus import Stimulus
from krakey.runtime.events.event_types import StimulusReadEvent


class WebChatReplyTool(Tool):
    def __init__(self, history: Any):
        self._history = history

    @property
    def name(self) -> str:
        return "web_chat_reply"

    @property
    def description(self) -> str:
        return ("Send a message to the web chat (dashboard). No LLM layer; "
                "the text is persisted + broadcast to all connected browsers "
                "as Krakey's voice. params: text (defaults to intent).")

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"text": "message body (defaults to intent)"}

    async def execute(self, intent: str,
                      params: dict[str, Any]) -> Stimulus:
        text = (params.get("text") or intent or "").strip()
        if not text:
            return self._stim("Empty message body; nothing sent.")
        try:
            await self._history.append("krakey", text)
        except Exception as e:  # noqa: BLE001
            return self._stim(f"WebChat send failed: {e}", adrenalin=True)
        return self._stim(f"Sent to web chat ({len(text)} chars).")

    def _stim(self, content: str, *, adrenalin: bool = False) -> Stimulus:
        return Stimulus(
            type="tool_feedback",
            source=f"tool:{self.name}",
            content=content,
            timestamp=datetime.now(),
            adrenalin=adrenalin,
        )


class WebChatMarkReadTool(Tool):
    """Read unread web-chat messages, return their full text, and mark them
    read so the unread reminder stops firing."""

    def __init__(self, history: Any, events: Any):
        self._history = history
        self._events = events

    @property
    def name(self) -> str:
        return "web_chat_mark_read"

    @property
    def description(self) -> str:
        return (
            "Reads unread web-chat messages, returns their full text, and "
            "marks them read so the unread reminder stops. Optional param "
            "message_ids: list of message IDs to mark read (default: all unread)."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "message_ids": "optional list of message IDs to mark read; omit to mark all unread",
        }

    async def execute(self, intent: str, params: dict[str, Any]) -> Stimulus:
        ids_filter = params.get("message_ids") or []
        unread = [
            m for m in self._history.all_messages()
            if m.get("status") == "delivered"
            and (not ids_filter or m.get("id") in ids_filter)
        ]
        if not unread:
            return self._stim("No unread web-chat messages.")

        ids = [m["id"] for m in unread if m.get("id")]
        if self._events is not None and ids:
            self._events.publish(StimulusReadEvent(chat_message_ids=ids))

        lines = [f"Unread web-chat messages ({len(unread)}):"]
        for m in unread:
            lines.append(f"[{m.get('ts', '')}] {m.get('content', '')}")
        content = "\n".join(lines)
        return self._stim(content)

    def _stim(self, content: str, *, adrenalin: bool = False) -> Stimulus:
        return Stimulus(
            type="tool_feedback",
            source=f"tool:{self.name}",
            content=content,
            timestamp=datetime.now(),
            adrenalin=adrenalin,
        )
