"""Default Hypothalamus Reflect — thin wrapper around the existing
``src.hypothalamus.Hypothalamus`` class.

Currently this is the only ``kind="hypothalamus"`` Reflect that ships
with Krakey. Reflect #1 (the toggle-able variant + action-tag executor
engine for direct Self → tentacle calls) will register an alternative
implementation; users pick which one to load via ``config.yaml``.
"""
from __future__ import annotations

from typing import Any

from src.hypothalamus import Hypothalamus, HypothalamusResult


class DefaultHypothalamusReflect:
    """LLM-driven decision → tentacle-call translator. Registered by
    default unless the user explicitly disables it in ``config.yaml``.
    """

    name = "default_hypothalamus"
    kind = "hypothalamus"

    def __init__(self, llm: Any):
        # Composition over inheritance: keeps the existing Hypothalamus
        # class entirely free of Reflect-specific concerns. The wrapper
        # is the seam.
        self._inner = Hypothalamus(llm)

    async def translate(
        self, decision: str, tentacles: list[dict[str, Any]],
    ) -> HypothalamusResult:
        return await self._inner.translate(decision, tentacles)
