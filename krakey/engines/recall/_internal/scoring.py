"""Recall scoring helpers — recall engine private (DevSpec §9.1).

Pure functions + the ``ScoringWeights`` config dataclass. No I/O,
no GM access. Given a candidate node + its vec_similarity,
``scripted_score`` returns a float that the recall engine uses as
its fallback ranking when no reranker is bound (or the bound
reranker fails).

Lives inside the recall engine package (underscore prefix = private)
because no other engine needs these formulas. The reranker-call
fail-soft wrapper that used to live alongside this code is now
inlined at each caller — six lines of try/except is cheaper than a
shared utility module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any


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


def doc_for_rerank(node: dict[str, Any]) -> str:
    """Render a GM node into a short string the reranker scores
    against the query. Recall-engine private; sleep migration's
    dedup pass keeps its own copy of this 3-line formula rather
    than crossing into the recall engine to import it."""
    name = node.get("name") or ""
    desc = node.get("description") or ""
    return f"{name}: {desc}" if desc else name


def _as_dt(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    # SQLite default format "YYYY-MM-DD HH:MM:SS"
    return datetime.fromisoformat(value.replace("T", " ")[:19])
