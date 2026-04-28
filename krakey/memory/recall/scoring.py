"""Recall scoring (DevSpec §9.1).

Pure functions + the ``ScoringWeights`` config dataclass. No I/O,
no GM access, no LLM. Given a candidate node + its vec_similarity,
``scripted_score`` returns a float; ``rank_candidates`` orchestrates
"use reranker if available else fall back to scripted_score" and
returns the ordered list.

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
from typing import Any, Protocol


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


class Reranker(Protocol):
    async def rerank(self, query: str, docs: list[str]) -> list[float]: ...


async def rank_candidates(candidates: list[tuple[dict[str, Any], float]],
                            *, query: str, reranker: Reranker | None,
                            weights: ScoringWeights,
                            now: datetime
                            ) -> list[tuple[dict[str, Any], float]]:
    """Layer 2/3 of DevSpec §9.1:
      - If a reranker is provided and succeeds, use its scores.
      - On any failure (missing, network error, score count mismatch),
        fall back to scripted_score().
    Returns (node, final_score) sorted descending.
    """
    if not candidates:
        return []

    if reranker is not None:
        try:
            docs = [_doc_for_rerank(n) for (n, _sim) in candidates]
            scores = await reranker.rerank(query, docs)
            if len(scores) == len(candidates):
                paired = list(zip((c[0] for c in candidates), scores))
                paired.sort(key=lambda x: x[1], reverse=True)
                return paired
        except Exception:  # noqa: BLE001
            pass  # fall through to scripted

    scored = [
        (n, scripted_score(n, vec_sim=sim, now=now, weights=weights))
        for (n, sim) in candidates
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _doc_for_rerank(node: dict[str, Any]) -> str:
    name = node.get("name") or ""
    desc = node.get("description") or ""
    return f"{name}: {desc}" if desc else name


def _as_dt(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    # SQLite default format "YYYY-MM-DD HH:MM:SS"
    return datetime.fromisoformat(value.replace("T", " ")[:19])
