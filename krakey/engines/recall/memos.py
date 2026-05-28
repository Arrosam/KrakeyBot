"""``MemOSRecallEngine`` — recall adapter for the MemOS memory engine.

Pairs with ``core_implementations.memory: memos``. MemOS retrieves by
TEXT (not raw vectors), so this session bypasses the embedder + raw
``vec_search`` path entirely: each stimulus's content is forwarded to
``memory.fts_search`` (which the MemOS memory adapter routes to
``MOS.search``); results are deduped by node id, token-budgeted, and
partitioned into covered/uncovered. Edges always empty — MemOS does
not expose its internal graph at the MOS API surface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from krakey.interfaces.engines.recall import RecallResult
from krakey.models.config import LLMParams
from krakey.utils.tokens import estimate_tokens

if TYPE_CHECKING:
    from krakey.interfaces.engines.memory import MemoryEngine
    from krakey.interfaces.engines.recall import RecallSession
    from krakey.models.config import Config
    from krakey.models.stimulus import Stimulus


def _node_token_cost(node: dict[str, Any]) -> int:
    name = node.get("name", "") or ""
    cat = node.get("category", "") or ""
    desc = node.get("description", "") or ""
    return estimate_tokens(f"- [{name}] ({cat}) — {desc}")


class MemOSRecallEngine:
    """RecallEngine paired with the MemOS memory adapter.

    Reads per_stimulus_k + recall_token_budget from cfg at construction;
    each ``new_session()`` yields a fresh, independent session. Accepts
    and ignores embedder/reranker/factory (passed by the runtime for the
    default impl) — MemOS does its own embedding + ranking.
    """

    def __init__(
        self,
        *,
        cfg: "Config",
        memory: "MemoryEngine",
        config: dict | None = None,
        **_ignored: Any,
    ):
        self._memory = memory
        self._per_stimulus_k = cfg.graph_memory.recall_per_stimulus_k
        self_params = cfg.llm.core_params("self_thinking") or LLMParams()
        self._recall_token_budget = self_params.recall_token_budget
        self._engine_cfg = config or {}

    def new_session(self) -> "RecallSession":
        return MemOSRecallSession(
            memory=self._memory,
            per_stimulus_k=self._per_stimulus_k,
            recall_token_budget=self._recall_token_budget,
        )


class MemOSRecallSession:
    """Per-beat session: text retrieval via ``memory.fts_search``."""

    def __init__(
        self,
        *,
        memory: "MemoryEngine",
        per_stimulus_k: int,
        recall_token_budget: int,
    ):
        self._memory = memory
        self._per_stimulus_k = per_stimulus_k
        self._recall_token_budget = recall_token_budget
        self.processed_stimuli: list["Stimulus"] = []
        self._per_query_results: list[
            tuple["Stimulus", list[dict[str, Any]]]
        ] = []

    async def add_stimuli(self, stimuli: list["Stimulus"]) -> None:
        for stim in stimuli:
            results = await self._memory.fts_search(
                stim.content, top_k=self._per_stimulus_k,
            )
            self._per_query_results.append((stim, list(results or [])))
            self.processed_stimuli.append(stim)

    async def finalize(self) -> RecallResult:
        merged: dict[Any, dict[str, Any]] = {}
        freq: dict[Any, int] = {}
        first_seen: dict[Any, int] = {}
        cursor = 0
        for _stim, results in self._per_query_results:
            for node in results:
                nid = node.get("id")
                if nid is None:
                    continue
                if nid not in merged:
                    merged[nid] = node
                    first_seen[nid] = cursor
                    cursor += 1
                freq[nid] = freq.get(nid, 0) + 1

        candidates = sorted(
            merged.values(),
            key=lambda n: (-freq[n["id"]], first_seen[n["id"]]),
        )

        admitted: list[dict[str, Any]] = []
        cumulative = 0
        for node in candidates:
            cost = _node_token_cost(node)
            # Contract invariant: at least one node admitted when any
            # exist, even if it exceeds the budget.
            if not admitted or cumulative + cost <= self._recall_token_budget:
                admitted.append(node)
                cumulative += cost
            else:
                break

        covered: list["Stimulus"] = []
        uncovered: list["Stimulus"] = []
        for stim, results in self._per_query_results:
            (covered if results else uncovered).append(stim)

        return RecallResult(
            nodes=admitted,
            edges=[],
            covered_stimuli=covered,
            uncovered_stimuli=uncovered,
        )
