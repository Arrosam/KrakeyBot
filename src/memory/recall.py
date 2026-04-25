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

from src.utils.tokens import estimate_tokens

if TYPE_CHECKING:
    from src.memory.graph_memory import GraphMemory
    from src.models.stimulus import Stimulus


def _estimate_node_render_tokens(node: dict[str, Any],
                                   neighbor_keywords: list[str]) -> int:
    """Token cost of rendering one recall node in the [GRAPH MEMORY]
    layer. Mirrors PromptBuilder._layer_recall so the budget math
    stays honest.

    Format (see src/prompt/builder.py):
        - [name] (category) \u2014 description
          \u90bb\u76f8: kw1, kw2, ...
    """
    name = node.get("name", "") or ""
    cat = node.get("category", "") or ""
    desc = node.get("description", "") or ""
    header = f"- [{name}] ({cat}) \u2014 {desc}"
    total = estimate_tokens(header)
    if neighbor_keywords:
        total += estimate_tokens("  \u76f8\u90bb: " + ", ".join(neighbor_keywords))
    return total


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


class NoopRecall:
    """No-op stand-in for ``IncrementalRecall``.

    Returned by ``ReflectRegistry.make_recall`` when no
    ``recall_anchor`` Reflect is registered. Honors the core design
    principle (Samuel 2026-04-25): **disabling any plugin must not
    break runtime operation**. Without recall, Self simply sees an
    empty ``[GRAPH MEMORY]`` layer every beat — graceful degradation,
    not a crash.

    Implements the duck-typed surface that ``Runtime`` consumes:
    ``processed_stimuli`` (read), ``add_stimuli`` (no-op),
    ``finalize`` (returns empty ``RecallResult``).
    """

    def __init__(self) -> None:
        self.processed_stimuli: list[Any] = []
        # _per_stimulus_ids and merged are read by some test paths and
        # by recall-result rebuilding logic. Empty containers are fine
        # — the runtime never iterates them past length zero.
        self._per_stimulus_ids: list[set[int]] = []
        self.merged: dict[int, dict[str, Any]] = {}

    async def add_stimuli(self, stimuli: list[Any]) -> None:
        # Track them as "processed" so the dedup logic in
        # _phase_drain_and_seed_recall doesn't keep re-feeding the
        # same Stimulus across beats (which would be harmless here
        # but pointless).
        self.processed_stimuli.extend(stimuli)

    async def finalize(self) -> RecallResult:
        return RecallResult()


class IncrementalRecall:
    """Per-stimulus vector search + weighted merge (DevSpec §9.2).

    Supports embedder failure → FTS5 fallback, optional reranker,
    adrenalin weight multiplier, covered/uncovered classification,
    neighbor-keyword hints, and edges among the selected set.
    """

    def __init__(self, gm: "GraphMemory", *,
                  embedder: AsyncEmbedder,
                  per_stimulus_k: int,
                  recall_token_budget: int,
                  weights: ScoringWeights | None = None,
                  reranker: Reranker | None = None,
                  neighbor_depth: int = 1,
                  vec_min_similarity: float = 0.3,
                  now: Callable[[], datetime] | None = None):
        """``recall_token_budget`` replaces the old ``max_recall_nodes``
        cap: finalize() walks candidates in weight-descending order and
        stops once the cumulative rendered-token cost would exceed the
        budget. An absolute token cap (not a fraction of context) —
        too many recall items pollute the prompt regardless of how big
        the model's context is. See ``LLMParams.recall_token_budget``.
        """
        self.gm = gm
        self.embedder = embedder
        self.per_k = per_stimulus_k
        self.recall_token_budget = recall_token_budget
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

        # Token-budget cap. Neighbor keywords participate in the render
        # cost, so we fetch them up front for the full candidate pool
        # (not just "top" — selection depends on the cost), then walk
        # in weight order and admit until the budget would be exceeded.
        all_ids = [e["node"]["id"] for e in sorted_entries]
        if all_ids:
            neighbor_map_full = await self.gm.get_neighbor_keywords(
                all_ids, depth=self.neighbor_depth,
            )
        else:
            neighbor_map_full = {}

        top: list[dict[str, Any]] = []
        cost_so_far = 0
        for entry in sorted_entries:
            n = entry["node"]
            kws = neighbor_map_full.get(n["id"], [])
            cost = _estimate_node_render_tokens(n, kws)
            if top and cost_so_far + cost > self.recall_token_budget:
                # Admit at least one node even if it alone exceeds the
                # budget — dropping everything would be worse UX than
                # overshooting the soft cap. The overall-prompt
                # enforcement in the heartbeat handles the true limit.
                break
            top.append(entry)
            cost_so_far += cost

        selected_ids = {e["node"]["id"] for e in top}

        covered = []
        uncovered = []
        for s, hit_ids in zip(self.processed_stimuli, self._per_stimulus_ids):
            if hit_ids & selected_ids:
                covered.append(s)
            else:
                uncovered.append(s)

        if selected_ids:
            edges = await self.gm.get_edges_among(list(selected_ids))
        else:
            edges = []

        nodes = []
        for entry in top:
            node = dict(entry["node"])
            node["neighbor_keywords"] = neighbor_map_full.get(node["id"], [])
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
