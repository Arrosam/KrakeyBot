"""IncrementalRecall — the per-stimulus search + weighted-merge driver.

Lives in this plugin (not in ``src/memory/recall/``) because it is
the *implementation* of the ``recall_anchor`` Reflect role and the
**only producer** of the type. Core defines the ``RecallLike``
Protocol that ``Runtime`` types against; this module supplies one
concrete implementation. A future plugin can ship a different
RecallLike (LLM-anchor, semantic-clustered, hybrid, …) without
touching core.

Per-beat algorithm:
  1. ``add_stimuli(stims)`` — for each stimulus, vec_search top-K
     candidates (via the shared ``gm_query`` helper, FTS fallback
     included), rerank, accumulate into ``merged`` with weight
     accumulation across stimuli (adrenalin ×10).
  2. ``finalize()`` — sort merged entries by weight, walk in order
     admitting nodes whose rendered token cost fits the budget,
     fetch edges among the selected set, classify each input
     stimulus as covered/uncovered.

Pure scoring helpers (``rank_candidates`` / ``scripted_score`` /
``ScoringWeights``) stay in core ``src.memory.recall.scoring`` —
they're math, not implementation policy, and a future RecallLike
plugin may want to reuse them.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Callable, TYPE_CHECKING

from src.memory.recall import (
    AsyncEmbedder, RecallResult, Reranker, ScoringWeights, rank_candidates,
)
from src.plugins.recall.gm_query import query_gm_with_fts_fallback
from src.utils.tokens import estimate_tokens

if TYPE_CHECKING:
    from src.memory.graph_memory import GraphMemory
    from src.models.stimulus import Stimulus


# Heuristic average rendered-token cost of a single GM node header.
# Used to translate the per-stimulus screening token target into a
# vec_search top_k. Doesn't need to be precise — finalize() does the
# real token accounting against the budget.
_AVG_NODE_TOKENS_FOR_SCREENING = 30


def _estimate_node_render_tokens(node: dict[str, Any],
                                   neighbor_keywords: list[str]) -> int:
    """Token cost of rendering one recall node in the [GRAPH MEMORY]
    layer. Mirrors PromptBuilder._layer_recall so the budget math
    stays honest.

    Format (see src/prompt/builder.py):
        - [name] (category) — description
          相邻: kw1, kw2, ...
    """
    name = node.get("name", "") or ""
    cat = node.get("category", "") or ""
    desc = node.get("description", "") or ""
    header = f"- [{name}] ({cat}) — {desc}"
    total = estimate_tokens(header)
    if neighbor_keywords:
        total += estimate_tokens("  相邻: " + ", ".join(neighbor_keywords))
    return total


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
                  screening_token_multiplier: float = 1.0,
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

        ``screening_token_multiplier`` widens the per-stimulus vec_search
        pool: each stimulus aims to surface ``recall_token_budget *
        screening_token_multiplier`` tokens worth of candidates so the
        cross-stimulus dedup + weight-merge has real material to compete
        on before the final budget cut admits the winners. ``per_stimulus_k``
        is the hard ceiling on vec_search top_k either way.
        """
        self.gm = gm
        self.embedder = embedder
        self.per_k = per_stimulus_k
        self.recall_token_budget = recall_token_budget
        self.screening_multiplier = screening_token_multiplier
        self.weights = weights or ScoringWeights()
        self.reranker = reranker
        self.neighbor_depth = neighbor_depth
        self._vec_min_sim = vec_min_similarity
        self._now = now or datetime.now
        self.merged: dict[int, dict[str, Any]] = {}
        self.processed_stimuli: list[Any] = []
        self._per_stimulus_ids: list[set[int]] = []

    def _screening_top_k(self) -> int:
        """Per-stimulus vec_search top_k. Sized to roughly cover the
        screening token target; capped by ``per_stimulus_k``; floored
        at 1 so degenerate config doesn't silently disable recall."""
        target = self.recall_token_budget * self.screening_multiplier
        soft = math.ceil(target / _AVG_NODE_TOKENS_FOR_SCREENING)
        return max(1, min(self.per_k, soft))

    async def add_stimuli(self, stimuli: list["Stimulus"]) -> None:
        for s in stimuli:
            candidates = await query_gm_with_fts_fallback(
                self.gm, self.embedder, s.content,
                top_k=self._screening_top_k(),
                min_similarity=self._vec_min_sim,
            )
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
