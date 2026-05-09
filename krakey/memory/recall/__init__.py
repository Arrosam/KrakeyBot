"""Recall subsystem: embedder Protocol + scoring helpers.

The per-beat recall driver itself lives in
``krakey.engines.recall.incremental`` (uplifted from the retired
``recall_anchor`` plugin into the RecallEngine slot). Public surface
re-exported at the package root keeps ``from krakey.memory.recall
import AsyncEmbedder, Reranker, ScoringWeights`` working unchanged.
"""
from krakey.memory.recall.incremental import AsyncEmbedder  # noqa: F401
from krakey.memory.recall.scoring import (  # noqa: F401
    CATEGORY_WEIGHTS,
    Reranker,
    ScoringWeights,
    category_weight,
    rerank,
    scripted_score,
    time_decay,
)
