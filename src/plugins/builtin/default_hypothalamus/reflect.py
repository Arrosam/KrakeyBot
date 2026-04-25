"""Default Hypothalamus Reflect — thin wrapper around the existing
``src.hypothalamus.Hypothalamus`` class.

Imported lazily by
``src.plugins.unified_discovery.load_component`` only when the user
enables ``default_hypothalamus`` in ``config.yaml``'s ``plugins:``
list. Until then this module is invisible to the process — discovery
walks ``meta.yaml`` only.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.hypothalamus import Hypothalamus, HypothalamusResult

if TYPE_CHECKING:
    from src.reflects.context import PluginContext


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


def build_reflect(ctx: "PluginContext") -> DefaultHypothalamusReflect | None:
    """Factory invoked by ``load_component``.

    Pulls the ``translator`` LLM via the plugin's per-plugin config
    binding (``workspace/plugins/default_hypothalamus/config.yaml``
    → ``llm_purposes.translator: <tag_name>``). When the user hasn't
    bound the tag, returns ``None`` — the loader skips this Reflect
    rather than crashing the runtime (additive plugin model).
    """
    import logging
    llm = ctx.get_llm("translator")
    if llm is None:
        logging.getLogger(__name__).warning(
            "default_hypothalamus: no LLM bound for purpose 'translator' "
            "(check workspace/plugins/default_hypothalamus/config.yaml "
            "and llm.tags in central config). Skipping registration."
        )
        return None
    return DefaultHypothalamusReflect(llm)
