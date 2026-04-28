"""Krakey → web chat outbound reply tool.

No LLM in this tool — Self already wrote the text in [DECISION];
this tool just persists + broadcasts it to the connected chat
clients. Returns success/failure as feedback. Failure preserves the
underlying error message and is marked adrenalin so Self knows to
react.

Lives with the dashboard plugin because the embedded chat UI is part
of the dashboard bundle. Runtime never references this tool by
name.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from krakey.interfaces.tool import Tool
from krakey.models.stimulus import Stimulus


class _HistoryLike(Protocol):
    async def append(self, sender: str, content: str) -> Any: ...


class WebChatReplyTool(Tool):
    def __init__(self, history: _HistoryLike):
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
