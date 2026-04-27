"""Recall subsystem (DevSpec §9): per-stimulus search → score → bound.

Two modules:

  * ``scoring``     — pure functions + ``ScoringWeights``. Category
                       weights, time decay, scripted score, reranker
                       orchestration, ``Reranker`` Protocol.
  * ``incremental`` — Protocols + null object: ``RecallLike``
                       (the duck-typed surface runtime types
                       against), ``RecallResult`` dataclass,
                       ``NoopRecall`` (zero-plugin fallback),
                       ``AsyncEmbedder`` Protocol.

Public API re-exported at this package root so
``from src.memory.recall import RecallLike, NoopRecall,
RecallResult, Reranker`` keeps working unchanged.

Note: the concrete ``IncrementalRecall`` driver lives in the
``default_recall_anchor`` plugin and is NOT re-exported from core.
Disabling that plugin removes the class from the import graph
entirely.
"""
from src.memory.recall.incremental import (  # noqa: F401
    AsyncEmbedder,
    NoopRecall,
    RecallLike,
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
