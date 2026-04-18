"""Recall scoring helpers + IncrementalRecall driver (DevSpec §9).

Phase 1.3a+b: scripted weighted sort, category weights, time decay.
Later sub-phases add reranker integration, FTS5 fallback, neighbor
expansion, and the IncrementalRecall class.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.graph_memory import GraphMemory
    from src.models.stimulus import Stimulus


class Reranker(Protocol):
    async def rerank(self, query: str, docs: list[str]) -> list[float]: ...


class AsyncEmbedder(Protocol):
    async def __call__(self, text: str) -> list[float]: ...


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


@dataclass
class RecallResult:
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    covered_stimuli: list[Any] = field(default_factory=list)
    uncovered_stimuli: list[Any] = field(default_factory=list)


class IncrementalRecall:
    """Per-stimulus vector search + weighted merge (DevSpec §9.2).

    Supports embedder failure → FTS5 fallback, optional reranker,
    adrenalin weight multiplier, covered/uncovered classification,
    neighbor-keyword hints, and edges among the selected set.
    """

    def __init__(self, gm: "GraphMemory", *,
                  embedder: AsyncEmbedder,
                  per_stimulus_k: int,
                  max_recall_nodes: int,
                  weights: ScoringWeights | None = None,
                  reranker: Reranker | None = None,
                  neighbor_depth: int = 1,
                  vec_min_similarity: float = 0.3,
                  now: Callable[[], datetime] | None = None):
        self.gm = gm
        self.embedder = embedder
        self.per_k = per_stimulus_k
        self.max_nodes = max_recall_nodes
        self.weights = weights or ScoringWeights()
        self.reranker = reranker
        self.neighbor_depth = neighbor_depth
        self._vec_min_sim = vec_min_similarity
        self._now = now or datetime.now
        self.merged: dict[int, dict[str, Any]] = {}
        self.processed_stimuli: list[Any] = []
        self._per_stimulus_ids: list[set[int]] = []

    async def add_stimuli(self, stimuli: list["Stimulus"]) -> None:
        for s in stimuli:
            candidates = await self._search_for(s.content)
            ranked = await rank_candidates(
                candidates, query=s.content, reranker=self.reranker,
                weights=self.weights, now=self._now(),
            )
            weight = 10.0 if getattr(s, "adrenalin", False) else 1.0
            hit_ids: set[int] = set()
            for (node, _score) in ranked:
                nid = node["id"]
                hit_ids.add(nid)
                if nid in self.merged:
                    self.merged[nid]["weight"] += weight
                else:
                    self.merged[nid] = {"node": node, "weight": weight}
            self._per_stimulus_ids.append(hit_ids)
            self.processed_stimuli.append(s)

    async def _search_for(self, text: str
                           ) -> list[tuple[dict[str, Any], float]]:
        candidates: list[tuple[dict[str, Any], float]] = []
        try:
            vec = await self.embedder(text)
            candidates = await self.gm.vec_search(
                vec, top_k=self.per_k, min_similarity=self._vec_min_sim,
            )
        except Exception:  # noqa: BLE001
            candidates = []
        if not candidates:
            fts_hits = await self.gm.fts_search(text, top_k=self.per_k)
            candidates = [(n, 0.0) for n in fts_hits]
        return candidates

    async def finalize(self) -> RecallResult:
        sorted_entries = sorted(self.merged.values(),
                                  key=lambda e: e["weight"], reverse=True)
        top = sorted_entries[: self.max_nodes]
        selected_ids = {e["node"]["id"] for e in top}

        covered = []
        uncovered = []
        for s, hit_ids in zip(self.processed_stimuli, self._per_stimulus_ids):
            if hit_ids & selected_ids:
                covered.append(s)
            else:
                uncovered.append(s)

        if selected_ids:
            neighbor_map = await self.gm.get_neighbor_keywords(
                list(selected_ids), depth=self.neighbor_depth,
            )
            edges = await self.gm.get_edges_among(list(selected_ids))
        else:
            neighbor_map = {}
            edges = []

        nodes = []
        for entry in top:
            node = dict(entry["node"])
            node["neighbor_keywords"] = neighbor_map.get(node["id"], [])
            node["score"] = entry["weight"]
            nodes.append(node)

        return RecallResult(nodes=nodes, edges=edges,
                              covered_stimuli=covered,
                              uncovered_stimuli=uncovered)


def _as_dt(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    # SQLite default format "YYYY-MM-DD HH:MM:SS"
    return datetime.fromisoformat(value.replace("T", " ")[:19])
