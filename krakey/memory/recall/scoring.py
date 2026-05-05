"""Recall scoring (DevSpec §9.1).

Pure functions + the ``ScoringWeights`` config dataclass. No I/O,
no GM access. Given a candidate node + its vec_similarity,
``scripted_score`` returns a float. ``rerank`` is a thin wrapper
around an injected Reranker that returns ordered pairs on success
or ``None`` on any failure — callers choose their own fallback
(scripted_score for the recall plugin; raw cosine ordering for the
sleep-migration dedup pass).

Lives separate from the IncrementalRecall driver in
``recall/incremental.py`` because:
  * Pure (testable in isolation, no DB fixture needed).
  * Stable interface — once the formula is tuned, this rarely
    changes; the driver evolves more (FTS fallback, neighbor
    expansion, token budget enforcement, ...).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


CATEGORY_WEIGHTS: dict[str, float] = {
    "TARGET": 1.5,
    "FOCUS": 1.5,
    "KNOWLEDGE": 1.2,
    "RELATION": 1.0,
    "FACT": 0.8,
}


def category_weight(category: str) -> float:
    return CATEGORY_WEIGHTS.get(category, 1.0)


def time_decay(created_at: datetime | str, now: datetime, *,
                half_life_seconds: float = 86400.0) -> float:
    """Exponential half-life decay. 1.0 at t=now, 0.5 after one half-life."""
    created = _as_dt(created_at)
    age = max(0.0, (now - created).total_seconds())
    return 0.5 ** (age / half_life_seconds)


@dataclass
class ScoringWeights:
    vec: float = 1.0         # w1
    time: float = 0.3        # w2
    access: float = 0.2      # w3
    importance: float = 0.5  # w4
    type: float = 0.5        # w5
    half_life_seconds: float = 86400.0


def scripted_score(node: dict[str, Any], *, vec_sim: float, now: datetime,
                    weights: ScoringWeights) -> float:
    """DevSpec §9.1 fallback formula."""
    decay = time_decay(node["created_at"], now,
                        half_life_seconds=weights.half_life_seconds)
    access_term = math.log((node.get("access_count") or 0) + 1)
    importance = float(node.get("importance") or 1.0)
    cat_w = category_weight(node.get("category", ""))
    return (
        vec_sim * weights.vec
        + decay * weights.time
        + access_term * weights.access
        + importance * weights.importance
        + cat_w * weights.type
    )


@runtime_checkable
class Reranker(Protocol):
    """Async reranker — returns one float score per doc in ``docs``,
    in the same order. Higher = better.

    ``@runtime_checkable`` so ``ServiceResolver`` can isinstance-check
    user-supplied reranker slots at startup (see
    ``core_implementations.reranker`` in config.yaml).
    """
    async def rerank(self, query: str, docs: list[str]) -> list[float]: ...


async def rerank(candidates: list[tuple[dict[str, Any], float]],
                   *, query: str,
                   reranker: Reranker | None
                   ) -> list[tuple[dict[str, Any], float]] | None:
    """Reorder candidates using the injected reranker. Returns
    reranker-scored ``(node, score)`` pairs sorted descending.

    Returns ``None`` on any fail-soft condition so the caller can
    apply its own fallback policy:
      * ``reranker is None`` (no reranker configured)
      * ``reranker.rerank`` raises
      * the reranker returned a different number of scores than
        candidates

    Empty candidate list short-circuits to ``[]`` (not ``None``) —
    nothing to rerank, but also nothing went wrong.
    """
    if not candidates:
        return []
    if reranker is None:
        return None
    try:
        docs = [_doc_for_rerank(n) for (n, _sim) in candidates]
        scores = await reranker.rerank(query, docs)
    except Exception:  # noqa: BLE001
        return None
    if len(scores) != len(candidates):
        return None
    paired = list(zip((c[0] for c in candidates), scores))
    paired.sort(key=lambda x: x[1], reverse=True)
    return paired


def _doc_for_rerank(node: dict[str, Any]) -> str:
    name = node.get("name") or ""
    desc = node.get("description") or ""
    return f"{name}: {desc}" if desc else name


def _as_dt(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    # SQLite default format "YYYY-MM-DD HH:MM:SS"
    return datetime.fromisoformat(value.replace("T", " ")[:19])
