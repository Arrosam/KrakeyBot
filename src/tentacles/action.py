"""Action Tentacle (DevSpec §5.2).

Phase-0 minimal impl: one LLM turn per intent. Maintains working
context across calls; on token bloat, auto-summarizes, resets
context, and returns the summary as the stimulus.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from src.interfaces.tentacle import Tentacle
from src.models.stimulus import Stimulus


SUMMARIZE_PROMPT = (
    "Summarize the conversation so far in a short paragraph suitable "
    "as a single tentacle feedback stimulus."
)


class ChatLike(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _context_tokens(context: list[dict]) -> int:
    return sum(_approx_tokens(m["content"]) for m in context)


class ActionTentacle(Tentacle):
    def __init__(self, llm: ChatLike, max_context_tokens: int = 4096):
        self._llm = llm
        self._max_tokens = max_context_tokens
        self.context: list[dict[str, str]] = []

    @property
    def name(self) -> str:
        return "action"

    @property
    def description(self) -> str:
        return "General computer-use agent. Search, browse, file I/O, messaging."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"intent": "natural language instruction"}

    async def execute(self, intent: str, params: dict[str, Any]) -> Stimulus:
        self.context.append({"role": "user", "content": intent})

        if _context_tokens(self.context) > self._max_tokens:
            summary = await self._llm.chat(
                [{"role": "user",
                  "content": SUMMARIZE_PROMPT + "\n\n" + str(self.context)}]
            )
            self.context = []
            return Stimulus(
                type="tentacle_feedback",
                source=f"tentacle:{self.name}",
                content=f"[Context limit. Summary] {summary}",
                timestamp=datetime.now(),
                adrenalin=False,
            )

        reply = await self._llm.chat(self.context)
        self.context.append({"role": "assistant", "content": reply})
        return Stimulus(
            type="tentacle_feedback",
            source=f"tentacle:{self.name}",
            content=reply,
            timestamp=datetime.now(),
            adrenalin=False,
        )
