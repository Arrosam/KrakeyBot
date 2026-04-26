"""Incremental recall driver (DevSpec §9.2).

Per-stimulus vector search → reranker / scripted_score → token-bounded
selection of the best subset → covered/uncovered classification.

State (across one heartbeat):
  * ``processed_stimuli`` — stimuli we've already searched for. Used
    by Runtime's ``_phase_drain_and_seed_recall`` to avoid re-feeding
    the same Stimulus across heartbeats.
  * ``merged``             — id → ``{node, weight}``. Adrenalin
    stimuli contribute 10× weight; same node hit by multiple stimuli
    accumulates.
  * ``_per_stimulus_ids``  — per-call hit set, for finalize() to mark
    each stimulus covered iff at least one of its hits made the cut.

Pure scoring functions live in ``recall.scoring``; this module
focuses on the orchestration + GM access + budget enforcement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Protocol, TYPE_CHECKING

from src.memory.recall.scoring import (
    Reranker, ScoringWeights, rank_candidates,
)
from src.utils.tokens import estimate_tokens

if TYPE_CHECKING:
    from src.memory.graph_memory import GraphMemory
    from src.models.stimulus import Stimulus


class AsyncEmbedder(Protocol):
    async def __call__(self, text: str) -> list[float]: ...


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
