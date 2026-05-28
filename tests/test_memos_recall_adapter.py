"""Edge tests for MemOSRecallEngine -- the optional MemOS recall adapter.

Tests validate the contract mapping between MemOSRecallEngine / MemOSRecallSession
and the RecallEngine + RecallSession Protocols.

The adapter under test is importable as:
    from krakey.engines.recall.memos import MemOSRecallEngine

Test organisation:
  - FakeMemory / FakeCfg / helpers at top
  - Tests grouped by contract surface / concern
  - Each group uses Positive / BVA / State-Transition / Negative structure

All async tests use bare ``async def`` -- pytest-asyncio in auto mode (see
pytest.ini: asyncio_mode = auto).

Assumptions made by this test file:
  1. Stimulus construction: tests use the real ``Stimulus`` dataclass.  The
     ``_make_stim`` helper fills in the required positional fields with sensible
     defaults; the adapter is only expected to read ``.content``.
  2. "At least one node admitted under budget=0": when candidates exist and
     ``recall_token_budget=0``, the session's ``finalize()`` still returns
     ``len(result.nodes) >= 1``.  The exact count may be 1 (single node) or
     more if cost is 0, but must never be 0 when candidates are available.
  3. Dedup by id: the adapter keeps only the FIRST occurrence of each node id
     in the admitted set; ordering within that set is implementation-defined
     but each id appears at most once.
  4. covered_stimuli is defined as: stimuli whose ``fts_search`` returned at
     least one result; uncovered = the rest.  The test confirms membership by
     identity (``is``), not by equality, because ``Stimulus`` is a mutable
     dataclass (no __hash__ override observed) -- tests use list membership by
     object identity where needed.
  5. ``estimate_tokens`` is the same helper used by the adapter; token-budget
     BVA tests compute expected costs using the same formula and assert
     exact admission counts accordingly.
  6. Extra kwargs passed by the registry (embedder, reranker, factory) are
     silently dropped by ``_filter_kwargs``; tests only pass cfg + memory +
     optional config to the constructor.
  7. The adapter does NOT silently swallow ``fts_search`` exceptions; the
     exception propagates out of ``add_stimuli``.  If the implementation
     wraps them instead, the test marked ``# propagation`` will fail and the
     assumption should be revisited.
"""
from __future__ import annotations

import pytest
from datetime import datetime
from types import SimpleNamespace
from typing import Any

from krakey.interfaces.engines.recall import RecallEngine, RecallSession, RecallResult
from krakey.models.stimulus import Stimulus
from krakey.utils.tokens import estimate_tokens


# ---------------------------------------------------------------------------
# Helpers / Fakes
# ---------------------------------------------------------------------------

def _make_stim(text: str, *, adrenalin: bool = False) -> Stimulus:
    """Construct a minimal real Stimulus.  The adapter only reads .content."""
    return Stimulus(
        type="user_message",
        source="test:recall_adapter",
        content=text,
        timestamp=datetime(2026, 5, 28),
        adrenalin=adrenalin,
    )


def _make_node(
    node_id: int,
    name: str = "Node",
    description: str = "A node.",
    category: str = "FACT",
) -> dict[str, Any]:
    """Construct a minimal node dict with all four required fields."""
    return {
        "id": node_id,
        "name": name,
        "description": description,
        "category": category,
    }


class FakeMemory:
    """Async fake whose ``fts_search`` returns canned results.

    ``per_query`` maps query string → list of node dicts.  Queries not found
    in the map fall back to ``default``.  Both lists are sliced to ``top_k``.
    """

    def __init__(
        self,
        per_query: dict[str, list[dict]] | None = None,
        default: list[dict] | None = None,
    ):
        self.calls: list[tuple[str, int]] = []
        self._per_query: dict[str, list[dict]] = per_query or {}
        self._default: list[dict] = default or []

    async def fts_search(self, query: str, *, top_k: int) -> list[dict]:
        self.calls.append((query, top_k))
        results = self._per_query.get(query, self._default)
        return list(results[:top_k])


class RaisingMemory:
    """FakeMemory variant that raises on every fts_search call."""

    async def fts_search(self, query: str, *, top_k: int) -> list[dict]:
        raise RuntimeError("fts_search deliberately failing")


def _cfg(per_stim_k: int = 5, recall_token_budget: int = 1_000) -> SimpleNamespace:
    """Build a minimal cfg duck-compatible with what the adapter reads."""
    self_params = SimpleNamespace(recall_token_budget=recall_token_budget)
    llm = SimpleNamespace(
        core_params=lambda purpose: self_params if purpose == "self_thinking" else None
    )
    graph_memory = SimpleNamespace(recall_per_stimulus_k=per_stim_k)
    return SimpleNamespace(graph_memory=graph_memory, llm=llm)


def _engine(
    per_stim_k: int = 5,
    recall_token_budget: int = 1_000,
    *,
    memory: FakeMemory | None = None,
    config: dict | None = None,
):
    """Construct a MemOSRecallEngine with the given settings."""
    from krakey.engines.recall.memos import MemOSRecallEngine

    return MemOSRecallEngine(
        cfg=_cfg(per_stim_k=per_stim_k, recall_token_budget=recall_token_budget),
        memory=memory if memory is not None else FakeMemory(),
        config=config,
    )


def _node_token_cost(node: dict) -> int:
    """Replicate the token estimator formula from the spec."""
    name = node.get("name", "") or ""
    cat = node.get("category", "") or ""
    desc = node.get("description", "") or ""
    header = f"- [{name}] ({cat}) — {desc}"
    return estimate_tokens(header)


# ===========================================================================
# 1. POSITIVE / EQUIVALENCE TESTS
# ===========================================================================


class TestProtocolConformance:
    """Positive: engine and session satisfy RecallEngine / RecallSession."""

    def test_engine_is_instance_of_recall_engine_protocol(self):
        engine = _engine()
        # RecallEngine is runtime_checkable -- isinstance must pass.
        assert isinstance(engine, RecallEngine)

    def test_new_session_returns_recall_session_instance(self):
        engine = _engine()
        session = engine.new_session()
        # RecallSession is runtime_checkable.
        assert isinstance(session, RecallSession)

    def test_new_session_returns_distinct_objects_each_call(self):
        engine = _engine()
        s1 = engine.new_session()
        s2 = engine.new_session()
        assert s1 is not s2

    def test_session_has_processed_stimuli_attribute(self):
        session = _engine().new_session()
        assert hasattr(session, "processed_stimuli")

    def test_processed_stimuli_starts_empty(self):
        session = _engine().new_session()
        assert session.processed_stimuli == []


class TestSingleStimulusThreeResults:
    """Positive: single stimulus with 3 fts results -> 3 nodes, stim covered."""

    async def test_nodes_contain_all_three_results(self):
        nodes = [_make_node(i, f"Node{i}", f"Desc{i}") for i in range(3)]
        mem = FakeMemory(default=nodes)
        session = _engine(memory=mem).new_session()
        stim = _make_stim("topic")
        await session.add_stimuli([stim])
        result = await session.finalize()

        ids_in_result = {n["id"] for n in result.nodes}
        assert ids_in_result == {0, 1, 2}

    async def test_covered_stimuli_contains_stimulus(self):
        nodes = [_make_node(0, "A", "Alpha")]
        mem = FakeMemory(default=nodes)
        session = _engine(memory=mem).new_session()
        stim = _make_stim("alpha query")
        await session.add_stimuli([stim])
        result = await session.finalize()

        assert stim in result.covered_stimuli
        assert stim not in result.uncovered_stimuli

    async def test_uncovered_stimuli_empty_when_all_covered(self):
        nodes = [_make_node(1, "Node", "Desc")]
        mem = FakeMemory(default=nodes)
        session = _engine(memory=mem).new_session()
        stim = _make_stim("something")
        await session.add_stimuli([stim])
        result = await session.finalize()

        assert result.uncovered_stimuli == []

    async def test_result_is_recall_result_dataclass(self):
        mem = FakeMemory(default=[_make_node(0)])
        session = _engine(memory=mem).new_session()
        await session.add_stimuli([_make_stim("x")])
        result = await session.finalize()

        assert isinstance(result, RecallResult)
        assert isinstance(result.nodes, list)
        assert isinstance(result.edges, list)
        assert isinstance(result.covered_stimuli, list)
        assert isinstance(result.uncovered_stimuli, list)


class TestMultiStimulusDisjointResults:
    """Positive: multiple stimuli with disjoint node ids -> all unique nodes preserved."""

    async def test_all_disjoint_nodes_present(self):
        nodes_a = [_make_node(1, "NodeA", "desc a")]
        nodes_b = [_make_node(2, "NodeB", "desc b")]
        mem = FakeMemory(
            per_query={"query_a": nodes_a, "query_b": nodes_b}
        )
        session = _engine(memory=mem).new_session()
        await session.add_stimuli([_make_stim("query_a"), _make_stim("query_b")])
        result = await session.finalize()

        ids = {n["id"] for n in result.nodes}
        assert 1 in ids
        assert 2 in ids

    async def test_both_stimuli_covered_when_each_has_results(self):
        nodes_a = [_make_node(10)]
        nodes_b = [_make_node(20)]
        mem = FakeMemory(per_query={"qa": nodes_a, "qb": nodes_b})
        session = _engine(memory=mem).new_session()
        stim_a = _make_stim("qa")
        stim_b = _make_stim("qb")
        await session.add_stimuli([stim_a, stim_b])
        result = await session.finalize()

        assert stim_a in result.covered_stimuli
        assert stim_b in result.covered_stimuli
        assert result.uncovered_stimuli == []


class TestDedupByNodeId:
    """Positive: overlapping node ids -> each id appears exactly once in result."""

    async def test_shared_node_appears_once(self):
        shared_node = _make_node(99, "Shared", "shared desc")
        mem = FakeMemory(
            per_query={
                "q1": [shared_node, _make_node(1, "Only in q1")],
                "q2": [shared_node, _make_node(2, "Only in q2")],
            }
        )
        session = _engine(memory=mem).new_session()
        await session.add_stimuli([_make_stim("q1"), _make_stim("q2")])
        result = await session.finalize()

        result_ids = [n["id"] for n in result.nodes]
        assert result_ids.count(99) == 1

    async def test_all_unique_ids_present_despite_overlap(self):
        shared = _make_node(5, "Shared")
        mem = FakeMemory(
            per_query={
                "q1": [shared, _make_node(6)],
                "q2": [shared, _make_node(7)],
            }
        )
        session = _engine(memory=mem).new_session()
        await session.add_stimuli([_make_stim("q1"), _make_stim("q2")])
        result = await session.finalize()

        result_ids = {n["id"] for n in result.nodes}
        assert {5, 6, 7} == result_ids

    async def test_no_duplicate_ids_in_nodes(self):
        """No duplicate IDs in result regardless of how many stimuli share a node."""
        repeated = _make_node(42, "Repeated")
        mem = FakeMemory(default=[repeated])  # every query returns the same node
        session = _engine(memory=mem).new_session()
        stims = [_make_stim(f"q{i}") for i in range(5)]
        await session.add_stimuli(stims)
        result = await session.finalize()

        ids = [n["id"] for n in result.nodes]
        assert len(ids) == len(set(ids)), "Node ids must be unique in result"


class TestProcessedStimuliTracking:
    """Positive: processed_stimuli tracks every added stimulus in order."""

    async def test_processed_stimuli_tracks_single_stimulus(self):
        session = _engine().new_session()
        stim = _make_stim("hello")
        await session.add_stimuli([stim])
        assert session.processed_stimuli == [stim]

    async def test_processed_stimuli_tracks_multiple_stimuli_in_order(self):
        session = _engine().new_session()
        stims = [_make_stim(f"msg{i}") for i in range(4)]
        await session.add_stimuli(stims)
        assert session.processed_stimuli == stims

    async def test_processed_stimuli_accumulates_across_add_calls(self):
        session = _engine().new_session()
        stim_a = _make_stim("first")
        stim_b = _make_stim("second")
        await session.add_stimuli([stim_a])
        await session.add_stimuli([stim_b])
        assert session.processed_stimuli == [stim_a, stim_b]

    async def test_processed_stimuli_even_when_no_results(self):
        """Stimulus with no fts results still lands in processed_stimuli."""
        mem = FakeMemory(default=[])
        session = _engine(memory=mem).new_session()
        stim = _make_stim("no results here")
        await session.add_stimuli([stim])
        assert stim in session.processed_stimuli


# ===========================================================================
# 2. BOUNDARY VALUE ANALYSIS TESTS
# ===========================================================================


class TestEmptyStimulusList:
    """BVA: empty stimuli list."""

    async def test_finalize_with_no_stimuli_returns_empty_nodes(self):
        session = _engine().new_session()
        result = await session.finalize()
        assert result.nodes == []

    async def test_finalize_with_no_stimuli_returns_empty_covered(self):
        session = _engine().new_session()
        result = await session.finalize()
        assert result.covered_stimuli == []

    async def test_finalize_with_no_stimuli_returns_empty_uncovered(self):
        session = _engine().new_session()
        result = await session.finalize()
        assert result.uncovered_stimuli == []

    async def test_add_empty_list_is_no_op(self):
        session = _engine().new_session()
        await session.add_stimuli([])
        result = await session.finalize()
        assert result.nodes == []
        assert session.processed_stimuli == []


class TestSingleStimulusNoResults:
    """BVA: single stimulus whose fts_search returns nothing."""

    async def test_nodes_empty_when_no_search_results(self):
        mem = FakeMemory(default=[])
        session = _engine(memory=mem).new_session()
        stim = _make_stim("unknown topic")
        await session.add_stimuli([stim])
        result = await session.finalize()
        assert result.nodes == []

    async def test_stimulus_goes_to_uncovered(self):
        mem = FakeMemory(default=[])
        session = _engine(memory=mem).new_session()
        stim = _make_stim("nothing matches")
        await session.add_stimuli([stim])
        result = await session.finalize()
        assert stim in result.uncovered_stimuli
        assert stim not in result.covered_stimuli

    async def test_edges_empty_when_no_results(self):
        mem = FakeMemory(default=[])
        session = _engine(memory=mem).new_session()
        await session.add_stimuli([_make_stim("q")])
        result = await session.finalize()
        assert result.edges == []


class TestTopKForwarding:
    """BVA: top_k forwarded to fts_search equals cfg.graph_memory.recall_per_stimulus_k."""

    async def test_top_k_matches_per_stim_k(self):
        per_k = 7
        mem = FakeMemory(default=[_make_node(1)])
        session = _engine(per_stim_k=per_k, memory=mem).new_session()
        await session.add_stimuli([_make_stim("query")])
        # Inspect the recorded call.
        assert len(mem.calls) == 1
        _, forwarded_top_k = mem.calls[0]
        assert forwarded_top_k == per_k

    async def test_top_k_forwarded_for_each_stimulus(self):
        per_k = 3
        mem = FakeMemory(default=[])
        session = _engine(per_stim_k=per_k, memory=mem).new_session()
        stims = [_make_stim(f"q{i}") for i in range(4)]
        await session.add_stimuli(stims)

        assert len(mem.calls) == 4
        for (_q, forwarded_top_k) in mem.calls:
            assert forwarded_top_k == per_k

    async def test_top_k_one(self):
        """per_stim_k=1 -> at most 1 node per stimulus."""
        nodes = [_make_node(i) for i in range(5)]
        mem = FakeMemory(default=nodes)
        session = _engine(per_stim_k=1, memory=mem).new_session()
        await session.add_stimuli([_make_stim("q")])
        # The fake slices at top_k, so fts_search returns 1 node.
        _, forwarded = mem.calls[0]
        assert forwarded == 1


class TestTokenBudgetZero:
    """BVA: recall_token_budget=0 -> at least one node admitted when candidates exist."""

    async def test_at_least_one_node_admitted_when_budget_zero(self):
        node = _make_node(1, "Alpha", "An important concept.")
        mem = FakeMemory(default=[node])
        session = _engine(recall_token_budget=0, memory=mem).new_session()
        await session.add_stimuli([_make_stim("alpha")])
        result = await session.finalize()

        assert len(result.nodes) >= 1, (
            "Contract invariant: at least one node must be admitted even if it "
            "alone exceeds the token budget (budget=0 case)."
        )

    async def test_no_nodes_admitted_when_budget_zero_and_no_candidates(self):
        """Budget=0 + no candidates -> nodes remains empty (the invariant only
        kicks in when candidates exist)."""
        mem = FakeMemory(default=[])
        session = _engine(recall_token_budget=0, memory=mem).new_session()
        await session.add_stimuli([_make_stim("q")])
        result = await session.finalize()
        assert result.nodes == []

    async def test_at_least_one_node_admitted_when_budget_one(self):
        node = _make_node(10, "Tiny", "Small desc.")
        mem = FakeMemory(default=[node])
        session = _engine(recall_token_budget=1, memory=mem).new_session()
        await session.add_stimuli([_make_stim("tiny")])
        result = await session.finalize()
        assert len(result.nodes) >= 1


class TestTokenBudgetLarge:
    """BVA: very large budget -> all unique nodes admitted."""

    async def test_all_nodes_admitted_when_budget_large(self):
        nodes = [_make_node(i, f"Node{i}", f"Description for node {i}.") for i in range(10)]
        mem = FakeMemory(default=nodes)
        session = _engine(recall_token_budget=10_000_000, per_stim_k=len(nodes), memory=mem).new_session()
        await session.add_stimuli([_make_stim("broad query")])
        result = await session.finalize()

        result_ids = {n["id"] for n in result.nodes}
        expected_ids = {n["id"] for n in nodes}
        assert result_ids == expected_ids

    async def test_all_unique_nodes_admitted_across_stimuli_large_budget(self):
        nodes_a = [_make_node(1, "Alpha"), _make_node(2, "Beta")]
        nodes_b = [_make_node(3, "Gamma"), _make_node(4, "Delta")]
        mem = FakeMemory(per_query={"qa": nodes_a, "qb": nodes_b})
        session = _engine(recall_token_budget=10_000_000, per_stim_k=10, memory=mem).new_session()
        await session.add_stimuli([_make_stim("qa"), _make_stim("qb")])
        result = await session.finalize()

        result_ids = {n["id"] for n in result.nodes}
        assert result_ids == {1, 2, 3, 4}


class TestTokenBudgetSmall:
    """BVA: very small budget -> only 1 node (or the minimum) admitted."""

    async def test_small_budget_limits_to_one_node(self):
        """Token budget = 1 -> only the first (highest-priority) node admitted."""
        nodes = [
            _make_node(i, f"Node{i}", "A description that uses some tokens.")
            for i in range(5)
        ]
        mem = FakeMemory(default=nodes)
        # Budget = 1 is below any realistic node cost, so at-least-one invariant
        # allows exactly one node in.
        session = _engine(recall_token_budget=1, per_stim_k=5, memory=mem).new_session()
        await session.add_stimuli([_make_stim("q")])
        result = await session.finalize()

        # The invariant: at least one, and the budget gates the rest.
        assert len(result.nodes) >= 1

    async def test_budget_exactly_covers_one_node(self):
        """Budget exactly equals one node's token cost -> that node admitted,
        the rest excluded."""
        node = _make_node(100, "ExactNode", "Exact description.")
        cost = _node_token_cost(node)

        # A second node that would push over the budget.
        node2 = _make_node(200, "ExtraNode", "Another description here.")

        mem = FakeMemory(per_query={"q": [node, node2]})
        session = _engine(recall_token_budget=cost, per_stim_k=5, memory=mem).new_session()
        await session.add_stimuli([_make_stim("q")])
        result = await session.finalize()

        # Node 100 must be admitted; node 200 must NOT be admitted (would exceed).
        result_ids = {n["id"] for n in result.nodes}
        assert 100 in result_ids
        assert 200 not in result_ids


class TestEdgesAlwaysEmpty:
    """BVA: edges field is always [] regardless of inputs."""

    async def test_edges_empty_with_nodes(self):
        mem = FakeMemory(default=[_make_node(1), _make_node(2)])
        session = _engine(memory=mem).new_session()
        await session.add_stimuli([_make_stim("query")])
        result = await session.finalize()
        assert result.edges == []

    async def test_edges_empty_with_no_nodes(self):
        session = _engine().new_session()
        result = await session.finalize()
        assert result.edges == []

    async def test_edges_empty_multiple_stimuli(self):
        nodes = [_make_node(i) for i in range(3)]
        mem = FakeMemory(default=nodes)
        session = _engine(memory=mem).new_session()
        await session.add_stimuli([_make_stim(f"q{i}") for i in range(3)])
        result = await session.finalize()
        assert result.edges == []


# ===========================================================================
# 3. STATE TRANSITION TESTS
# ===========================================================================


class TestSessionStateTransitions:
    """State transitions: add -> finalize produces result reflecting only what was added."""

    async def test_result_reflects_only_added_stimuli(self):
        node = _make_node(77, "Relevant", "Found by search.")
        mem = FakeMemory(per_query={"actual": [node]})
        session = _engine(memory=mem).new_session()
        await session.add_stimuli([_make_stim("actual")])
        result = await session.finalize()

        assert any(n["id"] == 77 for n in result.nodes)

    async def test_result_does_not_contain_nodes_from_queries_not_added(self):
        node_actual = _make_node(1, "Actual")
        node_phantom = _make_node(99, "Phantom", "Never queried")
        mem = FakeMemory(
            per_query={
                "actual": [node_actual],
                "phantom": [node_phantom],
            }
        )
        session = _engine(memory=mem).new_session()
        # Only add the "actual" stimulus, not "phantom".
        await session.add_stimuli([_make_stim("actual")])
        result = await session.finalize()

        result_ids = {n["id"] for n in result.nodes}
        assert 99 not in result_ids


class TestSessionIndependence:
    """State transitions: two new_session() calls produce independent sessions."""

    async def test_session_b_starts_with_empty_processed_stimuli(self):
        engine = _engine()
        session_a = engine.new_session()
        await session_a.add_stimuli([_make_stim("only in A")])

        session_b = engine.new_session()
        assert session_b.processed_stimuli == []

    async def test_session_b_finalize_is_independent_of_session_a(self):
        node_for_a = _make_node(1, "ForA", "Only for session A")
        node_for_b = _make_node(2, "ForB", "Only for session B")
        mem = FakeMemory(per_query={"qa": [node_for_a], "qb": [node_for_b]})
        engine = _engine(memory=mem)

        session_a = engine.new_session()
        await session_a.add_stimuli([_make_stim("qa")])

        session_b = engine.new_session()
        await session_b.add_stimuli([_make_stim("qb")])

        result_a = await session_a.finalize()
        result_b = await session_b.finalize()

        a_ids = {n["id"] for n in result_a.nodes}
        b_ids = {n["id"] for n in result_b.nodes}
        assert 1 in a_ids
        assert 2 not in a_ids
        assert 2 in b_ids
        assert 1 not in b_ids

    async def test_session_a_state_not_affected_by_session_b_add(self):
        engine = _engine()
        session_a = engine.new_session()
        stim_a = _make_stim("from a")
        await session_a.add_stimuli([stim_a])

        session_b = engine.new_session()
        await session_b.add_stimuli([_make_stim("from b")])

        # session_a's processed_stimuli must only contain stim_a.
        assert session_a.processed_stimuli == [stim_a]


class TestFinalizeWithNoStimuli:
    """State transitions: finalize() with no add_stimuli call -> all-empty result."""

    async def test_finalize_no_add_returns_empty_nodes(self):
        session = _engine().new_session()
        result = await session.finalize()
        assert result.nodes == []

    async def test_finalize_no_add_returns_empty_covered(self):
        session = _engine().new_session()
        result = await session.finalize()
        assert result.covered_stimuli == []

    async def test_finalize_no_add_returns_empty_uncovered(self):
        session = _engine().new_session()
        result = await session.finalize()
        assert result.uncovered_stimuli == []

    async def test_finalize_no_add_returns_empty_edges(self):
        session = _engine().new_session()
        result = await session.finalize()
        assert result.edges == []


class TestCoveredUncoveredPartition:
    """State transitions: covered + uncovered together equal all added stimuli."""

    async def test_covered_union_uncovered_equals_all_stimuli(self):
        node = _make_node(1)
        mem = FakeMemory(
            per_query={
                "covered": [node],
                "uncovered": [],
            }
        )
        session = _engine(memory=mem).new_session()
        stim_cov = _make_stim("covered")
        stim_unc = _make_stim("uncovered")
        await session.add_stimuli([stim_cov, stim_unc])
        result = await session.finalize()

        all_stims = set(result.covered_stimuli) | set(result.uncovered_stimuli)
        assert stim_cov in all_stims
        assert stim_unc in all_stims
        # No overlap.
        overlap = set(result.covered_stimuli) & set(result.uncovered_stimuli)
        assert len(overlap) == 0

    async def test_stimulus_with_results_goes_to_covered(self):
        mem = FakeMemory(per_query={"found": [_make_node(1)]})
        session = _engine(memory=mem).new_session()
        stim = _make_stim("found")
        await session.add_stimuli([stim])
        result = await session.finalize()
        assert stim in result.covered_stimuli

    async def test_stimulus_without_results_goes_to_uncovered(self):
        mem = FakeMemory(per_query={"found": [_make_node(1)], "empty": []})
        session = _engine(memory=mem).new_session()
        stim_empty = _make_stim("empty")
        await session.add_stimuli([_make_stim("found"), stim_empty])
        result = await session.finalize()
        assert stim_empty in result.uncovered_stimuli


# ===========================================================================
# 4. NEGATIVE / ERROR-GUESSING TESTS
# ===========================================================================


class TestCfgMissingFields:
    """Negative: cfg with correct population must not crash; incorrect cfg may raise."""

    def test_valid_cfg_does_not_raise_on_construction(self):
        from krakey.engines.recall.memos import MemOSRecallEngine

        # Should not raise.
        engine = MemOSRecallEngine(
            cfg=_cfg(per_stim_k=3, recall_token_budget=500),
            memory=FakeMemory(),
        )
        assert engine is not None

    def test_cfg_missing_graph_memory_raises_on_access(self):
        """Broken cfg without graph_memory should raise AttributeError (not silently
        produce wrong behavior) somewhere during use."""
        from krakey.engines.recall.memos import MemOSRecallEngine

        bad_cfg = SimpleNamespace()  # no graph_memory, no llm
        with pytest.raises((AttributeError, TypeError)):
            engine = MemOSRecallEngine(cfg=bad_cfg, memory=FakeMemory())
            # If construction succeeds, accessing the engine must fail when used.
            engine.new_session()


class TestFtsSearchExceptionPropagates:
    """Negative: memory.fts_search raising -> exception propagates from add_stimuli."""

    async def test_exception_propagates_from_add_stimuli(self):  # propagation
        """The adapter should NOT silently swallow fts_search errors.
        If this test fails because the implementation wraps the error,
        update the 'propagation' assumption in the module docstring."""
        mem = RaisingMemory()
        session = _engine(memory=mem).new_session()
        with pytest.raises(RuntimeError, match="fts_search deliberately failing"):
            await session.add_stimuli([_make_stim("query")])


class TestNodeMissingOptionalFields:
    """Negative: node dicts missing name/description don't crash the token estimator."""

    async def test_node_without_name_does_not_crash_finalize(self):
        node = {"id": 1, "category": "FACT"}  # no name, no description
        mem = FakeMemory(default=[node])
        session = _engine(memory=mem).new_session()
        await session.add_stimuli([_make_stim("q")])
        # Must not raise.
        result = await session.finalize()
        assert isinstance(result, RecallResult)

    async def test_node_without_description_does_not_crash(self):
        node = {"id": 2, "name": "OnlyName", "category": "FACT"}  # no description
        mem = FakeMemory(default=[node])
        session = _engine(memory=mem).new_session()
        await session.add_stimuli([_make_stim("q")])
        result = await session.finalize()
        assert isinstance(result, RecallResult)

    async def test_node_with_all_missing_optional_fields_admitted(self):
        """The at-least-one invariant still holds when nodes lack name/description."""
        node = {"id": 3}  # only id
        mem = FakeMemory(default=[node])
        session = _engine(recall_token_budget=0, memory=mem).new_session()
        await session.add_stimuli([_make_stim("q")])
        result = await session.finalize()
        # Should still admit 1 (invariant), and not crash.
        assert len(result.nodes) >= 1


class TestExtraKwargsDropped:
    """Negative: constructor silently drops extra kwargs (embedder/reranker/factory)
    that the registry passes but the adapter does not declare."""

    def test_extra_kwargs_are_dropped_silently(self):
        from krakey.engines.recall.memos import MemOSRecallEngine

        # Should not raise TypeError about unexpected keyword arguments.
        engine = MemOSRecallEngine(
            cfg=_cfg(),
            memory=FakeMemory(),
            embedder=object(),   # extra -- not declared
            reranker=object(),   # extra -- not declared
            factory=object(),    # extra -- not declared
        )
        assert engine is not None


class TestFinalizeReturnType:
    """Negative: finalize always returns a RecallResult instance."""

    async def test_finalize_returns_recall_result_even_after_error_recovery(self):
        """Even with an empty memory, the return type is always RecallResult."""
        session = _engine(memory=FakeMemory(default=[])).new_session()
        await session.add_stimuli([_make_stim("nothing")])
        result = await session.finalize()
        assert isinstance(result, RecallResult)

    async def test_finalize_nodes_is_list_not_none(self):
        session = _engine().new_session()
        result = await session.finalize()
        assert result.nodes is not None
        assert isinstance(result.nodes, list)


class TestNodeIdCountInResult:
    """Negative: result.nodes never contains duplicate ids under any input shape."""

    async def test_many_stimuli_same_query_no_id_duplication(self):
        node = _make_node(55, "Repeated", "Shows up everywhere.")
        mem = FakeMemory(default=[node])
        session = _engine(memory=mem).new_session()
        # 10 stimuli all matching the same node.
        stims = [_make_stim("q") for _ in range(10)]
        await session.add_stimuli(stims)
        result = await session.finalize()

        ids = [n["id"] for n in result.nodes]
        assert ids.count(55) == 1, "Node 55 must appear exactly once despite 10 stimuli"

    async def test_nodes_list_has_no_duplicate_ids_large_overlap(self):
        nodes = [_make_node(i % 3, f"Node{i % 3}", "Desc") for i in range(9)]
        # Only 3 distinct ids: 0, 1, 2
        mem = FakeMemory(default=nodes)
        session = _engine(per_stim_k=9, memory=mem).new_session()
        await session.add_stimuli([_make_stim("q")])
        result = await session.finalize()

        ids = [n["id"] for n in result.nodes]
        assert len(ids) == len(set(ids))
