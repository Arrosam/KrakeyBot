"""Recall subsystem (DevSpec §9): per-stimulus search → score → bound.

Two modules:

  * ``scoring``     — pure functions + ``ScoringWeights``. Category
                       weights, time decay, scripted score, reranker
                       orchestration, ``Reranker`` Protocol.
  * ``incremental`` — the ``IncrementalRecall`` driver itself plus the
                       ``RecallResult`` dataclass and the ``NoopRecall``
                       null-object (zero-plugin invariant fallback).

Public API re-exported at this package root so existing imports
``from src.memory.recall import IncrementalRecall, NoopRecall,
RecallResult, Reranker`` keep working unchanged.
"""
from src.memory.recall.incremental import (  # noqa: F401
    AsyncEmbedder,
    IncrementalRecall,
    NoopRecall,
    RecallResult,
)
from src.memory.recall.scoring import (  # noqa: F401
    CATEGORY_WEIGHTS,
    Reranker,
    ScoringWeights,
    category_weight,
    rank_candidates,
    scripted_score,
    time_decay,
)
