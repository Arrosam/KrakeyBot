"""Per-stimulus search + weighted-merge recall driver.

One per beat; the ``IncrementalRecallEngine`` factory in
``default.py`` builds a fresh instance via ``new_session()`` each
heartbeat. Implements the ``RecallSession`` Protocol declared in
``krakey.interfaces.engines.recall``.

Per-beat algorithm:
  1. ``add_stimuli(stims)`` — for each stimulus, vec_search top-K
     candidates (via the shared ``gm_query`` helper, FTS fallback
     included), rerank, accumulate into ``merged`` with weight
     accumulation across stimuli (adrenalin ×10).
  2. ``finalize()`` — sort merged entries by weight, walk in order
     admitting nodes whose rendered token cost fits the budget,
     fetch edges among the selected set, classify each input
     stimulus as covered/uncovered.

Pure scoring helpers (``scripted_score`` / ``ScoringWeights`` /
``doc_for_rerank``) are recall-engine-private and live in the
sibling ``_scoring`` module.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Callable, TYPE_CHECKING

from krakey.engines.recall._scoring import (
    ScoringWeights, doc_for_rerank, scripted_score,
)
from krakey.engines.recall.gm_query import query_gm_with_fts_fallback
from krakey.interfaces.engines.recall import RecallResult
from krakey.utils.tokens import estimate_tokens

if TYPE_CHECKING:
    from krakey.interfaces.engines.memory import MemoryEngine
    from krakey.interfaces.engines.reranker import RerankerEngine
    from krakey.interfaces.duck import AsyncEmbedder
    from krakey.models.stimulus import Stimulus


# Heuristic average rendered-token cost of a single GM node header.
# Used to translate the per-stimulus screening token target into a
# vec_search top_k. Doesn't need to be precise — finalize() does the
# real token accounting against the budget.
_AVG_NODE_TOKENS_FOR_SCREENING = 30


def _estimate_node_render_tokens(node: dict[str, Any],
                                   neighbor_keywords: list[str]) -> int:
    """Token cost of rendering one recall node in the [GRAPH MEMORY]
    layer. Mirrors the prompt builder's render_recall so the budget
    math stays honest.

    Format:
        - [name] (category) — description
          neighbors: kw1, kw2, ...
    """
    name = node.get("name", "") or ""
    cat = node.get("category", "") or ""
    desc = node.get("description", "") or ""
    header = f"- [{name}] ({cat}) — {desc}"
    total = estimate_tokens(header)
    if neighbor_keywords:
        total += estimate_tokens(
            "  neighbors: " + ", ".join(neighbor_keywords),
        )
    return total


class IncrementalRecall:
    """Per-stimulus vector search + weighted merge.

    Supports embedder failure → FTS5 fallback, optional reranker,
    adrenalin weight multiplier, covered/uncovered classification,
    neighbor-keyword hints, and edges among the selected set.
    """

    def __init__(self, memory: "MemoryEngine", *,
                  embedder: "AsyncEmbedder",
                  per_stimulus_k: int,
                  recall_token_budget: int,
                  screening_token_multiplier: float = 1.0,
                  weights: ScoringWeights | None = None,
                  reranker: "RerankerEngine | None" = None,
                  neighbor_depth: int = 1,
                  vec_min_similarity: float = 0.3,
                  now: Callable[[], datetime] | None = None):
        """``recall_token_budget`` is an absolute cap: finalize() walks
        candidates in weight-descending order and stops once the
        cumulative rendered-token cost would exceed the budget.

        ``screening_token_multiplier`` widens the per-stimulus vec_search
        pool: each stimulus aims to surface ``recall_token_budget *
        screening_token_multiplier`` tokens worth of candidates so the
        cross-stimulus dedup + weight-merge has real material to compete
        on before the final budget cut admits the winners. ``per_stimulus_k``
        is the hard ceiling on vec_search top_k either way.
        """
        self._memory = memory
        self._embedder = embedder
        self._per_k = per_stimulus_k
        self._recall_token_budget = recall_token_budget
        self._screening_multiplier = screening_token_multiplier
        self._weights = weights or ScoringWeights()
        self._reranker = reranker
        self._neighbor_depth = neighbor_depth
        self._vec_min_sim = vec_min_similarity
        self._now = now or datetime.now
        self._merged: dict[int, dict[str, Any]] = {}
        self.processed_stimuli: list["Stimulus"] = []
        self._per_stimulus_ids: list[set[int]] = []

    async def _rerank_or_fallback(
        self,
        query: str,
        candidates: list[tuple[dict[str, Any], float]],
    ) -> list[tuple[dict[str, Any], float]]:
        """Run the bound RerankerEngine over ``candidates``; on any
        fail-soft condition (no reranker, raise, length mismatch, empty
        candidate list short-circuits to []) fall back to scripted
        multi-axis scoring against ``vec_sim``. Returns ``(node, score)``
        sorted descending."""
        if not candidates:
            return []
        if self._reranker is not None:
            try:
                docs = [doc_for_rerank(n) for (n, _sim) in candidates]
                scores = await self._reranker.rerank(query, docs)
                if len(scores) == len(candidates):
                    paired = list(zip(
                        (c[0] for c in candidates), scores,
                    ))
                    paired.sort(key=lambda x: x[1], reverse=True)
                    return paired
            except Exception:  # noqa: BLE001
                pass
        # Fallback: scripted multi-axis scoring keyed off vec similarity.
        now = self._now()
        ranked = [
            (n, scripted_score(n, vec_sim=sim, now=now,
                                  weights=self._weights))
            for (n, sim) in candidates
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def _screening_top_k(self) -> int:
        """Per-stimulus vec_search top_k. Sized to roughly cover the
        screening token target; capped by ``per_stimulus_k``; floored
        at 1 so degenerate config doesn't silently disable recall."""
        target = self._recall_token_budget * self._screening_multiplier
        soft = math.ceil(target / _AVG_NODE_TOKENS_FOR_SCREENING)
        return max(1, min(self._per_k, soft))

    async def add_stimuli(self, stimuli: list["Stimulus"]) -> None:
        for s in stimuli:
            candidates = await query_gm_with_fts_fallback(
                self._memory, self._embedder, s.content,
                top_k=self._screening_top_k(),
                min_similarity=self._vec_min_sim,
            )
            ranked = await self._rerank_or_fallback(s.content, candidates)
            weight = 10.0 if getattr(s, "adrenalin", False) else 1.0
            hit_ids: set[int] = set()
            for (node, _score) in ranked:
                nid = node["id"]
                hit_ids.add(nid)
                if nid in self._merged:
                    self._merged[nid]["weight"] += weight
                else:
                    self._merged[nid] = {"node": node, "weight": weight}
            self._per_stimulus_ids.append(hit_ids)
            self.processed_stimuli.append(s)

    async def finalize(self) -> RecallResult:
        sorted_entries = sorted(self._merged.values(),
                                  key=lambda e: e["weight"], reverse=True)

        # Token-budget cap. Neighbor keywords participate in the render
        # cost, so we fetch them up front for the full candidate pool
        # (not just "top" — selection depends on the cost), then walk
        # in weight order and admit until the budget would be exceeded.
        all_ids = [e["node"]["id"] for e in sorted_entries]
        if all_ids:
            neighbor_map_full = await self._memory.get_neighbor_keywords(
                all_ids, depth=self._neighbor_depth,
            )
        else:
            neighbor_map_full = {}

        top: list[dict[str, Any]] = []
        cost_so_far = 0
        for entry in sorted_entries:
            n = entry["node"]
            kws = neighbor_map_full.get(n["id"], [])
            cost = _estimate_node_render_tokens(n, kws)
            if top and cost_so_far + cost > self._recall_token_budget:
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
            edges = await self._memory.get_edges_among(list(selected_ids))
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
