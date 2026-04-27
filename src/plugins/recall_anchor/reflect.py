"""Recall-anchor Reflect — wraps the ``IncrementalRecall`` factory.

Imported lazily by ``src.plugin_system.load_component`` only when
the user enables ``recall_anchor`` in ``config.yaml``'s ``plugins:``
list.

The Reflect captures everything it needs (GraphMemory, embedder,
reranker, config knobs) from ``PluginContext`` at construction time.
``make_recall(runtime)`` then ignores its ``runtime`` argument —
captured state is sufficient. The runtime parameter is part of the
``RecallAnchorReflect`` Protocol (in ``src.interfaces.reflect``) so
other plugins can choose to read from runtime if they prefer; this
implementation does not.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.models.config import LLMParams
from src.plugins.recall_anchor.incremental import IncrementalRecall

if TYPE_CHECKING:
    from src.interfaces.plugin_context import PluginContext
    from src.memory.graph_memory import GraphMemory
    from src.memory.recall import AsyncEmbedder, RecallLike, Reranker


class RecallAnchorReflectImpl:
    """Per-beat ``IncrementalRecall`` factory using captured state."""

    name = "recall_anchor"
    role = "recall_anchor"

    def __init__(
        self, *,
        gm: "GraphMemory",
        embedder: "AsyncEmbedder",
        reranker: "Reranker | None",
        per_stimulus_k: int,
        neighbor_depth: int,
        recall_token_budget: int,
    ):
        self._gm = gm
        self._embedder = embedder
        self._reranker = reranker
        self._per_k = per_stimulus_k
        self._neighbor_depth = neighbor_depth
        self._token_budget = recall_token_budget

    def make_recall(self, runtime: Any) -> "RecallLike":
        del runtime
        return IncrementalRecall(
            self._gm,
            embedder=self._embedder,
            per_stimulus_k=self._per_k,
            recall_token_budget=self._token_budget,
            reranker=self._reranker,
            neighbor_depth=self._neighbor_depth,
        )


def build_reflect(ctx: "PluginContext") -> RecallAnchorReflectImpl:
    """Factory invoked by ``load_component``. Pulls GM + embedder +
    reranker from ``ctx.services`` and recall config knobs from
    ``ctx.deps.config`` so ``make_recall`` does not need a runtime
    ref. No LLM purposes declared in meta.yaml — ``ctx.get_llm_for_tag``
    is never needed."""
    cfg = ctx.deps.config
    self_params = cfg.llm.core_params("self_thinking") or LLMParams()
    return RecallAnchorReflectImpl(
        gm=ctx.services["gm"],
        embedder=ctx.services["embedder"],
        reranker=ctx.services.get("reranker"),
        per_stimulus_k=cfg.graph_memory.recall_per_stimulus_k,
        neighbor_depth=cfg.graph_memory.neighbor_expand_depth,
        recall_token_budget=self_params.recall_token_budget,
    )
