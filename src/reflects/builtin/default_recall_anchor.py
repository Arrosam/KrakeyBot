"""Default recall-anchor Reflect — wraps the scripted
``IncrementalRecall`` factory that Runtime used to call directly via
``_new_recall``.

This Reflect represents "auto-recall" mode: vec_search + scripted
weighted scoring against GM, no LLM in the recall path. It's the
default because it's fast, deterministic, and free of extra LLM
round-trips. Reflect #2 (LLM-driven anchor extractor) is the planned
opt-in alternative, defaulting OFF per the 2026-04-25 design.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.memory.recall import IncrementalRecall

if TYPE_CHECKING:
    from src.main import Runtime


class DefaultRecallAnchorReflect:
    """Per-beat ``IncrementalRecall`` factory using the runtime's
    embedder + reranker + per-stim-K + token budget.
    """

    name = "default_recall_anchor"
    kind = "recall_anchor"

    def make_recall(self, runtime: "Runtime") -> IncrementalRecall:
        # Reads config + deps off the runtime. Keeping construction
        # logic here (rather than on Runtime) means future recall
        # Reflects can vary it independently — e.g. a different
        # embedder, a different per_k, an LLM-anchor preprocessor —
        # without Runtime knowing the difference.
        self_params = runtime.config.llm.roles["self"].params
        return IncrementalRecall(
            runtime.gm,
            embedder=runtime.embedder,
            per_stimulus_k=runtime.config.graph_memory.recall_per_stimulus_k,
            recall_token_budget=self_params.recall_token_budget,
            reranker=runtime.reranker,
            neighbor_depth=runtime.config.graph_memory.neighbor_expand_depth,
        )
