"""Default recall-anchor Reflect — wraps the scripted
``IncrementalRecall`` factory.

Imported lazily by ``src.plugin_system.load_component`` only when
the user enables ``default_recall_anchor`` in ``config.yaml``'s
``reflects:`` list.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.memory.recall import IncrementalRecall

if TYPE_CHECKING:
    from src.main import Runtime
    from src.interfaces.plugin_context import PluginContext


class DefaultRecallAnchorReflect:
    """Per-beat ``IncrementalRecall`` factory using the runtime's
    embedder + reranker + per-stim-K + token budget.
    """

    name = "default_recall_anchor"
    role = "recall_anchor"

    def make_recall(self, runtime: "Runtime") -> IncrementalRecall:
        # Reads config + deps off the runtime. Construction logic
        # lives here (rather than on Runtime) so future recall
        # Reflects can vary it independently — different embedder,
        # different per_k, an LLM-anchor preprocessor — without
        # Runtime knowing the difference.
        from src.models.config import LLMParams
        self_params = (
            runtime.config.llm.core_params("self_thinking") or LLMParams()
        )
        return IncrementalRecall(
            runtime.gm,
            embedder=runtime.embedder,
            per_stimulus_k=runtime.config.graph_memory.recall_per_stimulus_k,
            recall_token_budget=self_params.recall_token_budget,
            reranker=runtime.reranker,
            neighbor_depth=runtime.config.graph_memory.neighbor_expand_depth,
        )


def build_reflect(ctx: "PluginContext") -> DefaultRecallAnchorReflect:
    """Factory invoked by ``load_reflect``. ``ctx`` unused here —
    DefaultRecallAnchorReflect reads everything off ``runtime`` at
    ``make_recall`` call time, not at construction. No LLM purposes
    declared in meta.yaml so ctx.get_llm is never needed."""
    del ctx
    return DefaultRecallAnchorReflect()
