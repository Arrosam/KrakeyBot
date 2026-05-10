"""Embedder Protocol + scoring helpers shared across the memory subsystem.

The per-beat recall driver lives in
``krakey.engines.recall.incremental``; this package only exposes the
math + the embedder callable shape.
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
