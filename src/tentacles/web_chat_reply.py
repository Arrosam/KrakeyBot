"""Phase 3.F.3: Krakey → web chat outbound.

No LLM — Hypothalamus already crafted the text; this tentacle just
persists + broadcasts it. Returns success/failure as feedback.
Failure preserves the underlying error message and is marked adrenalin
so Self knows to react.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from src.interfaces.tentacle import Tentacle
from src.models.stimulus import Stimulus


class _HistoryLike(Protocol):
    async def append(self, sender: str, content: str) -> Any: ...


class WebChatTentacle(Tentacle):
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

    @property
    def is_internal(self) -> bool:
        return False

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
            type="tentacle_feedback",
            source=f"tentacle:{self.name}",
            content=content,
            timestamp=datetime.now(),
            adrenalin=adrenalin,
        )
