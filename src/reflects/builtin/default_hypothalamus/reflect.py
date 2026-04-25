"""Default Hypothalamus Reflect — thin wrapper around the existing
``src.hypothalamus.Hypothalamus`` class.

Imported lazily by ``src.reflects.discovery.load_reflect`` only when
the user enables ``default_hypothalamus`` in ``config.yaml``'s
``reflects:`` list. Until then this module is invisible to the
process — discovery walks ``meta.yaml`` only.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.hypothalamus import Hypothalamus, HypothalamusResult

if TYPE_CHECKING:
    from src.main import RuntimeDeps


class DefaultHypothalamusReflect:
    """LLM-driven decision → tentacle-call translator.

    The wrapper composes over the existing ``Hypothalamus`` so the
    historical class stays free of Reflect-specific concerns.
    """

    name = "default_hypothalamus"
    kind = "hypothalamus"

    def __init__(self, llm: Any):
        self._inner = Hypothalamus(llm)

    async def translate(
        self, decision: str, tentacles: list[dict[str, Any]],
    ) -> HypothalamusResult:
        return await self._inner.translate(decision, tentacles)


def build_reflect(deps: "RuntimeDeps") -> DefaultHypothalamusReflect:
    """Factory invoked by ``load_reflect``. Receives the runtime's
    ``RuntimeDeps`` bundle and pulls whatever LLM / config it needs."""
    return DefaultHypothalamusReflect(deps.hypo_llm)
