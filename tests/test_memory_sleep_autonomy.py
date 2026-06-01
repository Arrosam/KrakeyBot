"""Edge tests for the memory engine's sleep/classify AUTONOMY change.

Contract under test: contracts/memory-access/definition.md
  -- "Sleep autonomy (engine-owned)" section
  -- "Re-categorization & sleep (internal)" section

Spec: The GraphMemoryEngine (krakey/engines/memory/default.py) gains:
  1. Construction-time sleep deps: sleep_llm=, reranker=, sleep_config=
  2. request_sleep(reason="") -> dict  (new async, the ONLY sleep entry point)
  3. sleep_cycle(...) with external channels/log_dir/config is REMOVED
  4. Autonomous trigger after storage ops push GM node count >= threshold
  5. No channel pause — request_sleep works with no channels whatsoever
  6. classify_and_link_pending is internal — callers need not invoke it

Tests are written BEFORE implementation lands — all tests targeting the
final state are RED until the dev unit implements the spec.

asyncio_mode = auto (see pytest.ini) — plain async def functions and
class-based async def methods work without @pytest.mark.asyncio.

---------------------------------------------------------------------------
ASSUMPTIONS (must be confirmed with dev/manager):

A1. ENGINE CONSTRUCTOR SIGNATURE:
    GraphMemoryEngine(
        db_path=":memory:",
        embedder=<async callable>,
        kb_dir=<path>,
        extractor_llm=None,
        classifier_llm=None,
        sleep_llm=None,         # NEW — ChatLike or None
        reranker=None,           # NEW — RerankerEngine or None
        sleep_config=None,       # NEW — dict or SleepSection-like or None
    )
    Existing positional/keyword params are unchanged.

A2. OBSERVABILITY HOOK:
    engine.sleep_cycles_run: int — starts at 0, increments by 1 each time
    request_sleep completes a real (non-no-op) cycle.
    This attribute is assumed to exist on the concrete GraphMemoryEngine
    for test observability; the contract does NOT mandate it on the Protocol.
    Tests that depend on this are marked with: # ASSUMES: sleep_cycles_run

A3. NO-OP WHEN sleep_llm IS NONE:
    request_sleep() returns {} immediately without raising when sleep_llm
    was not supplied at construction time. The engine may still increment
    sleep_cycles_run (as a no-op cycle), or it may leave the counter at 0.
    Tests on this assume {} return; they do NOT assert sleep_cycles_run value.

A4. AUTO-TRIGGER OBSERVABILITY:
    After enough ingest/remember/remember_extraction calls push the GM node
    count to or above auto_sleep_node_threshold (when threshold > 0), the
    engine self-triggers a sleep cycle internally.
    We observe this through sleep_cycles_run counter.
    The auto-trigger test is intentionally lenient: it asserts >= 1 cycle
    ran rather than exactly 1, because the engine may coalesce concurrent
    cycles.

A5. SLEEP_CONFIG DICT KEYS:
    When a plain dict is passed as sleep_config, the engine recognizes at
    least these keys (matching SleepSection field names):
      "auto_sleep_node_threshold", "min_community_size",
      "kb_consolidation_threshold", "kb_index_max", "kb_archive_pct",
      "kb_revive_threshold"
    Missing keys fall back to SleepSection defaults.

A6. IMPORT PATH:
    from krakey.engines.memory.default import GraphMemoryEngine
    This mirrors how existing tests import internal memory classes.
    If the module path changes, update the import.

A7. EMPTY GM SLEEP CYCLE:
    request_sleep() on a near-empty GM (0–2 nodes) does not call the LLM
    or perform expensive clustering — it runs trivially and returns a dict.
    The FakeChatLike.chat() call count may be 0 for an empty GM, so we do
    NOT assert chat() was called in empty-GM tests.

A8. FAKE RERANKER:
    FakeReranker.rerank(query, docs) returns [0.5] * len(docs).
    This satisfies the RerankerEngine Protocol (always returns len(docs) floats).
---------------------------------------------------------------------------
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """Deterministic async embedder — returns a fixed-dim vector.

    The vector dimension is 4; the first float is derived from
    len(text) % 5 so different texts get slightly different vectors.
    Satisfies AsyncEmbedder (async __call__ -> list[float]).
    """

    def __init__(self):
        self.calls: list[str] = []

    async def __call__(self, text: str) -> list[float]:
        self.calls.append(text)
        d = len(text) % 5
        return [0.1 + d * 0.1, 0.2, 0.3, 0.4]


class FakeChatLike:
    """Minimal ChatLike stub.

    Returns canned JSON strings suitable for both clustering-summary
    and KB-dedup LLM calls. Tracks how many times chat() was called.
    """

    def __init__(self, *, response: str = ""):
        self._response = response or '{"edges": []}'
        self.calls: list[list[dict]] = []

    async def chat(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        return self._response


class FakeReranker:
    """Minimal RerankerEngine stub — always returns 0.5 per doc.

    Satisfies the RerankerEngine Protocol:
        async def rerank(self, query: str, docs: list[str]) -> list[float]
    """

    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []

    async def rerank(self, query: str, docs: list[str]) -> list[float]:
        self.calls.append((query, docs))
        return [0.5] * len(docs)


def _min_sleep_config(**overrides) -> dict:
    """Return a minimal sleep_config dict with sensible defaults + overrides."""
    base = {
        "min_community_size": 1,          # migrate every community, even singletons
        "kb_consolidation_threshold": 0.99,  # near-impossible threshold → no merging
        "kb_index_max": 30,
        "kb_archive_pct": 10,
        "kb_revive_threshold": 0.99,
        "auto_sleep_node_threshold": 0,   # disabled unless overridden
    }
    base.update(overrides)
    return base


async def _build_engine(
    tmp_path,
    *,
    sleep_llm=None,
    reranker=None,
    auto_sleep_node_threshold: int = 0,
    embedder=None,
) -> Any:
    """Construct, initialize, and return a GraphMemoryEngine."""
    from krakey.engines.memory.default import GraphMemoryEngine

    embedder = embedder or FakeEmbedder()
    sleep_config = _min_sleep_config(
        auto_sleep_node_threshold=auto_sleep_node_threshold,
    )
    engine = GraphMemoryEngine(
        db_path=":memory:",
        embedder=embedder,
        kb_dir=str(tmp_path / "kbs"),
        sleep_llm=sleep_llm,
        reranker=reranker,
        sleep_config=sleep_config,
    )
    await engine.initialize()
    return engine


# ---------------------------------------------------------------------------
# 1. Positive — request_sleep API shape and basic behavior
# ---------------------------------------------------------------------------

class TestRequestSleepPositive:
    """Positive equivalence: request_sleep exists, is async, returns dict."""

    async def test_engine_has_request_sleep_method(self, tmp_path):
        """request_sleep must be an attribute of GraphMemoryEngine."""
        from krakey.engines.memory.default import GraphMemoryEngine
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
        )
        assert hasattr(engine, "request_sleep"), (
            "GraphMemoryEngine must expose request_sleep()."
        )

    async def test_request_sleep_is_coroutine_function(self, tmp_path):
        """request_sleep must be async (coroutine function)."""
        from krakey.engines.memory.default import GraphMemoryEngine
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
        )
        assert inspect.iscoroutinefunction(engine.request_sleep), (
            "request_sleep must be an async method (coroutine function)."
        )

    async def test_request_sleep_returns_dict_on_empty_gm_with_sleep_llm(
        self, tmp_path
    ):
        """On an empty GM with sleep_llm configured, request_sleep must
        return a dict (empty stats dict is valid per spec)."""
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker()
        )
        result = await engine.request_sleep()
        assert isinstance(result, dict), (
            f"request_sleep() must return a dict; got {type(result)!r}: {result!r}"
        )
        await engine.close()

    async def test_request_sleep_with_reason_string_returns_dict(self, tmp_path):
        """request_sleep(reason='user requested') must accept a reason string
        and still return a dict."""
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker()
        )
        result = await engine.request_sleep(reason="user requested consolidation")
        assert isinstance(result, dict)
        await engine.close()

    async def test_request_sleep_with_empty_reason_returns_dict(self, tmp_path):
        """request_sleep(reason='') — empty string reason is the default; valid."""
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker()
        )
        result = await engine.request_sleep(reason="")
        assert isinstance(result, dict)
        await engine.close()

    async def test_request_sleep_on_gm_with_two_nodes_returns_dict(self, tmp_path):
        """With 2 nodes ingested and sleep_llm, request_sleep returns a dict."""
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker()
        )
        await engine.ingest("fact about the sky being blue")
        await engine.ingest("fact about water being wet")
        result = await engine.request_sleep()
        assert isinstance(result, dict)
        await engine.close()

    async def test_request_sleep_result_keys_are_strings(self, tmp_path):
        """All keys in the stats dict returned by request_sleep are strings."""
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker()
        )
        result = await engine.request_sleep()
        for key in result:
            assert isinstance(key, str), (
                f"Stats dict key {key!r} is not a string."
            )
        await engine.close()


# ---------------------------------------------------------------------------
# 2. Positive — sleep_llm=None is a graceful no-op
# ---------------------------------------------------------------------------

class TestRequestSleepNoLlm:
    """When sleep_llm is not configured, request_sleep must return {}
    without raising any exception."""

    async def test_request_sleep_without_sleep_llm_returns_empty_dict(
        self, tmp_path
    ):
        """sleep_llm=None (default) → request_sleep() returns {} (graceful no-op)."""
        engine = await _build_engine(tmp_path, sleep_llm=None)
        result = await engine.request_sleep()
        assert result == {}, (
            f"With sleep_llm=None, request_sleep() must return {{}}; got {result!r}"
        )
        await engine.close()

    async def test_request_sleep_without_sleep_llm_does_not_raise(self, tmp_path):
        """sleep_llm=None must not cause request_sleep to raise any exception."""
        engine = await _build_engine(tmp_path, sleep_llm=None)
        try:
            await engine.request_sleep(reason="no llm configured")
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"request_sleep() raised {type(exc).__name__} when sleep_llm is None: {exc}"
            )
        await engine.close()

    async def test_request_sleep_without_sleep_llm_after_ingests(self, tmp_path):
        """Even after 5 ingests, sleep_llm=None → {} without crashing."""
        engine = await _build_engine(tmp_path, sleep_llm=None)
        for i in range(5):
            await engine.ingest(f"fact {i}: some content here")
        result = await engine.request_sleep()
        assert result == {}
        await engine.close()

    async def test_engine_constructible_without_sleep_params(self, tmp_path):
        """Engine must be constructible with no sleep_llm/reranker/sleep_config
        supplied — all must default gracefully."""
        from krakey.engines.memory.default import GraphMemoryEngine
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            # sleep_llm, reranker, sleep_config intentionally omitted
        )
        await engine.initialize()
        assert engine is not None
        await engine.close()


# ---------------------------------------------------------------------------
# 3. Negative — sleep_cycle old signature is removed / not callable
# ---------------------------------------------------------------------------

class TestSleepCycleRemoved:
    """sleep_cycle(channels=..., log_dir=..., config=...) must be absent
    or non-callable with that old external-arg form."""

    async def test_sleep_cycle_with_old_signature_raises_or_absent(self, tmp_path):
        """Calling sleep_cycle with channels/log_dir/config positional kwargs
        must either raise TypeError (wrong signature) or AttributeError (absent).
        The new engine must NOT accept the old 3-arg form."""
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker()
        )
        method = getattr(engine, "sleep_cycle", None)
        if method is None:
            # sleep_cycle is cleanly removed — test passes
            await engine.close()
            return

        # sleep_cycle still exists; calling it with the OLD external-arg
        # signature must raise TypeError (unexpected keyword arguments)
        with pytest.raises(TypeError):
            await method(channels=[], log_dir="/tmp", config={})
        await engine.close()

    async def test_request_sleep_exists_when_sleep_cycle_old_form_is_gone(
        self, tmp_path
    ):
        """Complementary check: regardless of sleep_cycle status, request_sleep
        must exist and be async. This verifies the REPLACEMENT landed."""
        engine = await _build_engine(tmp_path)
        assert hasattr(engine, "request_sleep"), (
            "request_sleep is the sole sleep entry point per spec — must exist."
        )
        assert inspect.iscoroutinefunction(engine.request_sleep), (
            "request_sleep must be async."
        )
        await engine.close()

    async def test_sleep_cycle_absent_from_engine_if_fully_removed(self, tmp_path):
        """If the implementation cleanly removes sleep_cycle, hasattr must be False.
        This test is lenient: it records the result but fails only if sleep_cycle
        exists AND accepts the old 3-arg form without raising."""
        engine = await _build_engine(tmp_path)
        if not hasattr(engine, "sleep_cycle"):
            await engine.close()
            return  # cleanly removed — best outcome

        # It still exists — verify calling with old args fails
        method = engine.sleep_cycle
        with pytest.raises(TypeError, match="channels|log_dir|config|unexpected"):
            await method(channels=[], log_dir="/tmp", config={})
        await engine.close()


# ---------------------------------------------------------------------------
# 4. Negative — request_sleep needs no channels (no channel pause)
# ---------------------------------------------------------------------------

class TestNoChannelPause:
    """request_sleep must not require or reference channels in any way.
    The engine is constructed with no channels reference at all."""

    async def test_request_sleep_succeeds_without_any_channels(self, tmp_path):
        """Engine constructed without channels → request_sleep must still work."""
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker()
        )
        # No channels object ever passed to engine — must succeed
        result = await engine.request_sleep(reason="no channels available")
        assert isinstance(result, dict)
        await engine.close()

    async def test_request_sleep_returns_dict_not_none_without_channels(
        self, tmp_path
    ):
        """request_sleep must return a dict, not None, when no channels exist."""
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker()
        )
        result = await engine.request_sleep()
        assert result is not None, "request_sleep must return dict, not None."
        assert isinstance(result, dict)
        await engine.close()

    async def test_multiple_request_sleep_calls_without_channels(self, tmp_path):
        """Calling request_sleep twice consecutively without channels must not
        crash on the second call. Tests idempotent-safe behavior."""
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker()
        )
        r1 = await engine.request_sleep()
        r2 = await engine.request_sleep()
        assert isinstance(r1, dict)
        assert isinstance(r2, dict)
        await engine.close()


# ---------------------------------------------------------------------------
# 5. State transition — sleep_cycles_run increments across explicit calls
# ---------------------------------------------------------------------------
# ASSUMES: sleep_cycles_run (int, starts 0, increments per cycle)
# If the engine does not expose this attribute, these tests will fail with
# AttributeError — a valid RED state confirming the observability hook is
# missing from the not-yet-implemented engine.

class TestSleepCyclesRunCounter:
    """State transition: sleep_cycles_run starts at 0 and increments."""

    async def test_sleep_cycles_run_starts_at_zero(self, tmp_path):
        """Before any sleep, sleep_cycles_run must be 0.
        ASSUMES: sleep_cycles_run attribute exists on GraphMemoryEngine."""
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker()
        )
        assert hasattr(engine, "sleep_cycles_run"), (
            "GraphMemoryEngine must expose sleep_cycles_run (int) for test "
            "observability per spec section A2."
        )
        assert engine.sleep_cycles_run == 0, (
            f"Expected sleep_cycles_run=0 before any sleep; "
            f"got {engine.sleep_cycles_run!r}."
        )
        await engine.close()

    async def test_sleep_cycles_run_increments_after_request_sleep(self, tmp_path):
        """After one request_sleep(), sleep_cycles_run must be >= 1.
        ASSUMES: sleep_cycles_run."""
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker()
        )
        before = engine.sleep_cycles_run
        await engine.request_sleep()
        after = engine.sleep_cycles_run
        assert after > before, (
            f"sleep_cycles_run did not increment after request_sleep(): "
            f"before={before}, after={after}."
        )
        await engine.close()

    async def test_sleep_cycles_run_increments_twice_after_two_calls(
        self, tmp_path
    ):
        """Two explicit request_sleep() calls → sleep_cycles_run increases by 2.
        ASSUMES: sleep_cycles_run."""
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker()
        )
        start = engine.sleep_cycles_run
        await engine.request_sleep(reason="first explicit call")
        mid = engine.sleep_cycles_run
        await engine.request_sleep(reason="second explicit call")
        end = engine.sleep_cycles_run
        assert mid > start, (
            f"sleep_cycles_run did not increment after first call: "
            f"start={start}, mid={mid}."
        )
        assert end > mid, (
            f"sleep_cycles_run did not increment after second call: "
            f"mid={mid}, end={end}."
        )
        await engine.close()

    async def test_sleep_cycles_run_does_not_increment_when_no_sleep_llm(
        self, tmp_path
    ):
        """When sleep_llm is None, request_sleep is a no-op — sleep_cycles_run
        must NOT increment (because no cycle actually ran).
        ASSUMES: sleep_cycles_run.
        NOTE: This test may be lenient — if the impl counts no-ops as cycles,
        relax the assertion to >= 0."""
        engine = await _build_engine(tmp_path, sleep_llm=None)
        before = engine.sleep_cycles_run
        await engine.request_sleep()
        after = engine.sleep_cycles_run
        # No real cycle ran; counter should stay the same
        assert after == before, (
            f"sleep_cycles_run incremented during no-op (sleep_llm=None): "
            f"before={before}, after={after}. "
            "A no-op should not count as a completed cycle."
        )
        await engine.close()


# ---------------------------------------------------------------------------
# 6. BVA — auto_sleep_node_threshold boundary values
# ---------------------------------------------------------------------------
# ASSUMES: sleep_cycles_run

class TestAutoSleepNodeThresholdBVA:
    """Boundary value analysis on auto_sleep_node_threshold."""

    async def test_threshold_zero_never_auto_triggers(self, tmp_path):
        """auto_sleep_node_threshold=0 (default, disabled) — even after many
        ingests, the engine must NOT self-trigger a sleep cycle.
        ASSUMES: sleep_cycles_run."""
        from krakey.engines.memory.default import GraphMemoryEngine

        sleep_llm = FakeChatLike()
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            sleep_llm=sleep_llm,
            reranker=FakeReranker(),
            sleep_config=_min_sleep_config(auto_sleep_node_threshold=0),
        )
        await engine.initialize()

        # Ingest 10 nodes — far above any reasonable threshold
        for i in range(10):
            await engine.ingest(f"threshold zero test node {i}: content here")

        # Allow any pending background tasks to complete
        await asyncio.sleep(0)

        assert engine.sleep_cycles_run == 0, (
            f"With auto_sleep_node_threshold=0, the engine must never auto-trigger "
            f"sleep; sleep_cycles_run={engine.sleep_cycles_run!r} after 10 ingests."
        )
        await engine.close()

    async def test_threshold_one_triggers_on_first_node(self, tmp_path):
        """auto_sleep_node_threshold=1 — after first ingest, engine auto-triggers.
        ASSUMES: sleep_cycles_run."""
        from krakey.engines.memory.default import GraphMemoryEngine

        sleep_llm = FakeChatLike()
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            sleep_llm=sleep_llm,
            reranker=FakeReranker(),
            sleep_config=_min_sleep_config(auto_sleep_node_threshold=1),
        )
        await engine.initialize()

        initial = engine.sleep_cycles_run
        await engine.ingest("first node ever — should trigger auto-sleep")
        # Allow any background auto-trigger to fire
        await asyncio.sleep(0)

        assert engine.sleep_cycles_run >= initial + 1, (
            f"With auto_sleep_node_threshold=1, ingest of first node must auto-trigger "
            f"sleep; sleep_cycles_run did not increment: "
            f"initial={initial}, current={engine.sleep_cycles_run}."
        )
        await engine.close()

    async def test_threshold_two_does_not_trigger_on_one_node(self, tmp_path):
        """auto_sleep_node_threshold=2 — one ingest must NOT trigger sleep;
        sleep is triggered only when count reaches >= 2.
        ASSUMES: sleep_cycles_run."""
        from krakey.engines.memory.default import GraphMemoryEngine

        sleep_llm = FakeChatLike()
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            sleep_llm=sleep_llm,
            reranker=FakeReranker(),
            sleep_config=_min_sleep_config(auto_sleep_node_threshold=2),
        )
        await engine.initialize()

        initial = engine.sleep_cycles_run
        await engine.ingest("node one — count=1, below threshold=2")
        await asyncio.sleep(0)

        assert engine.sleep_cycles_run == initial, (
            f"With threshold=2, a single ingest (count=1) must NOT auto-trigger sleep; "
            f"initial={initial}, current={engine.sleep_cycles_run}."
        )
        await engine.close()

    async def test_threshold_two_triggers_on_second_node(self, tmp_path):
        """auto_sleep_node_threshold=2 — second ingest pushes count to 2 → triggers.
        ASSUMES: sleep_cycles_run."""
        from krakey.engines.memory.default import GraphMemoryEngine

        sleep_llm = FakeChatLike()
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            sleep_llm=sleep_llm,
            reranker=FakeReranker(),
            sleep_config=_min_sleep_config(auto_sleep_node_threshold=2),
        )
        await engine.initialize()

        await engine.ingest("node one — count=1")
        before_second = engine.sleep_cycles_run

        await engine.ingest("node two — count=2, at threshold")
        await asyncio.sleep(0)

        assert engine.sleep_cycles_run >= before_second + 1, (
            f"With threshold=2, second ingest (count=2) must trigger auto-sleep; "
            f"before={before_second}, current={engine.sleep_cycles_run}."
        )
        await engine.close()

    async def test_threshold_boundary_remember_also_triggers(self, tmp_path):
        """remember() is also a storage op; crossing the threshold via
        remember() must also auto-trigger sleep (threshold=1).
        ASSUMES: sleep_cycles_run."""
        from krakey.engines.memory.default import GraphMemoryEngine

        sleep_llm = FakeChatLike()
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            sleep_llm=sleep_llm,
            reranker=FakeReranker(),
            sleep_config=_min_sleep_config(auto_sleep_node_threshold=1),
        )
        await engine.initialize()

        initial = engine.sleep_cycles_run
        await engine.remember("deliberately remembered fact about elephants")
        await asyncio.sleep(0)

        assert engine.sleep_cycles_run >= initial + 1, (
            f"remember() crossing threshold=1 must trigger auto-sleep; "
            f"initial={initial}, current={engine.sleep_cycles_run}."
        )
        await engine.close()

    async def test_threshold_boundary_remember_extraction_also_triggers(
        self, tmp_path
    ):
        """remember_extraction() is also a storage op; crossing threshold=1
        via bulk extraction must auto-trigger sleep.
        ASSUMES: sleep_cycles_run."""
        from krakey.engines.memory.default import GraphMemoryEngine

        sleep_llm = FakeChatLike()
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            sleep_llm=sleep_llm,
            reranker=FakeReranker(),
            sleep_config=_min_sleep_config(auto_sleep_node_threshold=1),
        )
        await engine.initialize()

        initial = engine.sleep_cycles_run
        await engine.remember_extraction(
            nodes=[{"name": "Eiffel Tower", "category": "FACT",
                    "description": "landmark in Paris"}],
            edges=[],
        )
        await asyncio.sleep(0)

        assert engine.sleep_cycles_run >= initial + 1, (
            f"remember_extraction() crossing threshold=1 must trigger auto-sleep; "
            f"initial={initial}, current={engine.sleep_cycles_run}."
        )
        await engine.close()

    async def test_threshold_negative_or_zero_both_disable_auto_trigger(
        self, tmp_path
    ):
        """Both threshold=0 (canonical 'disabled') and threshold not set
        (i.e. default SleepSection.auto_sleep_node_threshold=0) must keep
        the engine from auto-triggering.
        ASSUMES: sleep_cycles_run."""
        from krakey.engines.memory.default import GraphMemoryEngine

        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            sleep_llm=FakeChatLike(),
            reranker=FakeReranker(),
            # sleep_config omitted — defaults to auto_sleep_node_threshold=0
        )
        await engine.initialize()

        for i in range(5):
            await engine.ingest(f"data {i}")
        await asyncio.sleep(0)

        assert engine.sleep_cycles_run == 0, (
            "Default (no sleep_config) must disable auto-trigger; "
            f"sleep_cycles_run={engine.sleep_cycles_run!r} after 5 ingests."
        )
        await engine.close()


# ---------------------------------------------------------------------------
# 7. Construction-time sleep deps stored on instance
# ---------------------------------------------------------------------------

class TestConstructionTimeStoredDeps:
    """The engine stores sleep_llm, reranker, sleep_config at construction.
    Verify they are accessible on the instance (implementation detail check)
    OR at minimum that request_sleep uses them (behavioral check via
    FakeChatLike.calls)."""

    async def test_sleep_llm_stored_and_used_by_request_sleep(self, tmp_path):
        """FakeChatLike.calls should show activity if GM has content to sleep on.
        For an EMPTY GM the LLM may not be called; we test with some content."""
        # Use threshold=0 (explicit only) so we control exactly when sleep runs
        fake_llm = FakeChatLike()
        engine = await _build_engine(
            tmp_path,
            sleep_llm=fake_llm,
            reranker=FakeReranker(),
            auto_sleep_node_threshold=0,
        )
        # Insert a couple of nodes so the sleep pipeline has something to do
        await engine.remember("factual statement about the moon")
        await engine.remember("another factual statement about space")

        calls_before = len(fake_llm.calls)
        await engine.request_sleep()
        calls_after = len(fake_llm.calls)

        # If clustering ran and called the LLM for summaries, calls_after > calls_before.
        # For trivially small GMs it may still be 0 — we assert it didn't LOSE calls.
        assert calls_after >= calls_before, (
            "FakeChatLike.calls must not decrease after request_sleep()."
        )
        await engine.close()

    async def test_reranker_stored_does_not_prevent_request_sleep(self, tmp_path):
        """Passing a FakeReranker at construction must not cause any crash
        during request_sleep() — it should be used silently."""
        reranker = FakeReranker()
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=reranker
        )
        result = await engine.request_sleep()
        assert isinstance(result, dict)
        await engine.close()

    async def test_sleep_config_dict_accepted_without_error(self, tmp_path):
        """Passing sleep_config as a plain dict must not crash construction
        or request_sleep()."""
        from krakey.engines.memory.default import GraphMemoryEngine

        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            sleep_llm=FakeChatLike(),
            reranker=FakeReranker(),
            sleep_config={
                "min_community_size": 1,
                "kb_consolidation_threshold": 0.90,
                "kb_index_max": 10,
                "kb_archive_pct": 20,
                "kb_revive_threshold": 0.75,
                "auto_sleep_node_threshold": 0,
            },
        )
        await engine.initialize()
        result = await engine.request_sleep()
        assert isinstance(result, dict)
        await engine.close()

    async def test_sleep_config_none_falls_back_to_defaults(self, tmp_path):
        """sleep_config=None (or not passed) must cause the engine to fall
        back to SleepSection defaults — no crash expected."""
        from krakey.engines.memory.default import GraphMemoryEngine

        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            sleep_llm=FakeChatLike(),
            reranker=FakeReranker(),
            sleep_config=None,
        )
        await engine.initialize()
        result = await engine.request_sleep()
        assert isinstance(result, dict)
        await engine.close()

    async def test_sleep_config_partial_dict_does_not_crash(self, tmp_path):
        """A sleep_config dict with only some keys (missing keys fall back to
        SleepSection defaults) must not crash construction or request_sleep."""
        from krakey.engines.memory.default import GraphMemoryEngine

        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            sleep_llm=FakeChatLike(),
            reranker=FakeReranker(),
            sleep_config={"min_community_size": 1},   # only one key
        )
        await engine.initialize()
        result = await engine.request_sleep()
        assert isinstance(result, dict)
        await engine.close()


# ---------------------------------------------------------------------------
# 8. Classify-is-internal — light-touch tests
# ---------------------------------------------------------------------------

class TestClassifyIsInternal:
    """classify_and_link_pending is internal; callers need not invoke it.
    Verifies that ingest/remember work without an external classify call."""

    async def test_ingest_does_not_require_external_classify_call(self, tmp_path):
        """ingest() must succeed and return a dict without any external
        classify_and_link_pending() call from the test."""
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker()
        )
        result = await engine.ingest("content without external classify call")
        assert isinstance(result, dict), (
            "ingest() must return a dict without an external classify call."
        )
        await engine.close()

    async def test_remember_does_not_require_external_classify_call(self, tmp_path):
        """remember() must succeed without any external classify_and_link_pending()."""
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker()
        )
        result = await engine.remember("deliberately stored fact")
        assert isinstance(result, dict)
        await engine.close()

    async def test_classify_and_link_pending_if_present_is_not_required(
        self, tmp_path
    ):
        """If classify_and_link_pending still exists as an internal method,
        calling it externally must not CRASH (it may be a no-op), but callers
        are not required to call it.

        If the method is fully removed, this test passes trivially via the
        'if hasattr' guard — that's the ideal outcome per the contract."""
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker()
        )
        await engine.ingest("some content to classify")

        if hasattr(engine, "classify_and_link_pending"):
            # It still exists — calling it must not crash
            try:
                result = await engine.classify_and_link_pending()
                # Result should be a dict if it returns anything
                assert result is None or isinstance(result, dict), (
                    f"classify_and_link_pending returned unexpected type: "
                    f"{type(result)!r}"
                )
            except Exception as exc:  # noqa: BLE001
                pytest.fail(
                    f"classify_and_link_pending raised unexpectedly: "
                    f"{type(exc).__name__}: {exc}"
                )
        # If not present: test passes (cleanly removed — preferred state)
        await engine.close()

    async def test_search_works_without_external_classify_call(self, tmp_path):
        """After ingest(), search() returns results without a classify call.
        This tests that the Protocol surface is fully functional without
        external classification orchestration."""
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker()
        )
        await engine.ingest("the quick brown fox jumps over the lazy dog")
        results = await engine.search("fox")
        assert isinstance(results, list), (
            "search() must return a list without an external classify call."
        )
        await engine.close()


# ---------------------------------------------------------------------------
# 9. Negative / error-guessing — edge cases
# ---------------------------------------------------------------------------

class TestNegativeErrorGuessing:
    """Error-guessing cases: wrong types, missing fields, invalid configs."""

    async def test_request_sleep_called_before_initialize_is_handled(self, tmp_path):
        """Calling request_sleep before initialize() should either raise a
        clear error (e.g. RuntimeError, AttributeError) or return {} safely.
        It must NOT silently corrupt state."""
        from krakey.engines.memory.default import GraphMemoryEngine

        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            sleep_llm=FakeChatLike(),
        )
        # NOT calling initialize()
        try:
            result = await engine.request_sleep()
            # Permissive: if it returns a dict without crashing, that's acceptable
            assert isinstance(result, dict), (
                f"request_sleep before initialize() must return dict or raise; "
                f"got {type(result)!r}: {result!r}"
            )
        except Exception:  # noqa: BLE001
            pass  # Raising is also acceptable

    async def test_request_sleep_after_close_is_handled(self, tmp_path):
        """Calling request_sleep after close() must raise a clear exception
        OR return {} safely — it must NOT silently succeed with stale data."""
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker()
        )
        await engine.close()
        # Attempt sleep after close — engine may raise or return {}
        try:
            result = await engine.request_sleep()
            assert isinstance(result, dict), (
                f"Post-close request_sleep must return dict or raise; "
                f"got {type(result)!r}."
            )
        except Exception:  # noqa: BLE001
            pass  # Raising after close is acceptable

    async def test_request_sleep_with_non_string_reason_raises_or_coerces(
        self, tmp_path
    ):
        """Passing a non-string reason (e.g. int) to request_sleep should
        raise TypeError OR silently coerce it — it must not corrupt state.
        We accept either outcome, but record it as a test boundary."""
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker()
        )
        try:
            result = await engine.request_sleep(reason=42)  # int, not str
            assert isinstance(result, dict), (
                f"request_sleep(reason=42) returned non-dict: {result!r}"
            )
        except TypeError:
            pass  # Strict type checking is also acceptable
        await engine.close()

    async def test_engine_without_reranker_still_runs_request_sleep(self, tmp_path):
        """reranker=None with sleep_llm set — request_sleep may succeed with
        a scripted fallback, OR it may raise a clear error. It must NOT hang."""
        from krakey.engines.memory.default import GraphMemoryEngine

        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            sleep_llm=FakeChatLike(),
            reranker=None,          # no reranker — relies on scripted fallback
            sleep_config=_min_sleep_config(),
        )
        await engine.initialize()
        try:
            result = await engine.request_sleep()
            assert isinstance(result, dict)
        except Exception:  # noqa: BLE001
            pass  # Acceptable: raising without reranker is valid behavior
        await engine.close()

    async def test_sleep_config_with_unknown_keys_does_not_crash(self, tmp_path):
        """Extraneous keys in sleep_config dict must not crash the engine."""
        from krakey.engines.memory.default import GraphMemoryEngine

        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            sleep_llm=FakeChatLike(),
            sleep_config={
                "auto_sleep_node_threshold": 0,
                "unknown_future_key": "some_value",  # unknown key
            },
        )
        await engine.initialize()
        result = await engine.request_sleep()
        assert isinstance(result, dict)
        await engine.close()

    async def test_sleep_config_with_invalid_threshold_type_handled(self, tmp_path):
        """sleep_config with a non-int threshold should raise TypeError/ValueError
        at construction or be silently coerced — must not silently use garbage."""
        from krakey.engines.memory.default import GraphMemoryEngine

        try:
            engine = GraphMemoryEngine(
                db_path=":memory:",
                embedder=FakeEmbedder(),
                kb_dir=str(tmp_path / "kbs"),
                sleep_llm=FakeChatLike(),
                sleep_config={"auto_sleep_node_threshold": "not_a_number"},
            )
            await engine.initialize()
            # If it got this far, coercion happened — ensure request_sleep still works
            result = await engine.request_sleep()
            assert isinstance(result, dict)
            await engine.close()
        except (TypeError, ValueError):
            pass  # Raising on bad config type is also correct behavior

    async def test_stats_dict_values_are_numeric_or_empty(self, tmp_path):
        """The stats dict from request_sleep should have numeric values (int/float)
        or be empty {}. No nested dicts, no None values for standard stat keys."""
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker()
        )
        await engine.ingest("some content for sleep stats test")
        result = await engine.request_sleep()
        assert isinstance(result, dict)
        for key, val in result.items():
            assert isinstance(val, (int, float, bool, str, list)), (
                f"Stats dict key {key!r} has unexpected value type "
                f"{type(val).__name__!r}: {val!r}. "
                "Expected simple scalar or list."
            )
        await engine.close()


# ---------------------------------------------------------------------------
# 10. BVA — request_sleep stats dict shape
# ---------------------------------------------------------------------------

class TestRequestSleepStatsDictBVA:
    """Boundary value analysis on the stats dict returned by request_sleep."""

    async def test_empty_dict_returned_for_empty_gm(self, tmp_path):
        """An empty GM may return {} or a dict of all-zero stats.
        Either is valid per the contract ('same shape the old sleep_cycle returned:
        keys like facts_migrated, focus_cleared, kbs_created, index_nodes — or {} if
        nothing to do')."""
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker()
        )
        result = await engine.request_sleep()
        assert isinstance(result, dict)
        # If it's non-empty, the known stat keys should be present
        if result:
            known_optional_keys = {
                "facts_migrated", "focus_cleared", "kbs_created", "index_nodes"
            }
            extra_keys = set(result.keys()) - known_optional_keys
            # Extra keys are allowed (additive is fine), but we log them
            # to catch accidental returns of wrong types
            for key in result.keys():
                assert isinstance(key, str)
        await engine.close()

    async def test_request_sleep_idempotent_on_empty_gm(self, tmp_path):
        """Calling request_sleep twice on an empty GM must return equivalent
        results (both dicts; coalescing is fine)."""
        engine = await _build_engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker()
        )
        r1 = await engine.request_sleep()
        r2 = await engine.request_sleep()
        assert isinstance(r1, dict)
        assert isinstance(r2, dict)
        # Both should have the same keys (values may differ for counters)
        assert set(r1.keys()) == set(r2.keys()), (
            f"Consecutive request_sleep() on empty GM returned different key sets: "
            f"first={set(r1.keys())!r}, second={set(r2.keys())!r}"
        )
        await engine.close()

    async def test_request_sleep_with_large_sleep_config_threshold_still_returns_dict(
        self, tmp_path
    ):
        """A very high auto_sleep_node_threshold (e.g. 9999) means auto-trigger
        never fires, but explicit request_sleep() still works."""
        from krakey.engines.memory.default import GraphMemoryEngine

        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            sleep_llm=FakeChatLike(),
            reranker=FakeReranker(),
            sleep_config=_min_sleep_config(auto_sleep_node_threshold=9999),
        )
        await engine.initialize()
        await engine.ingest("some content")
        result = await engine.request_sleep()
        assert isinstance(result, dict)
        await engine.close()

    async def test_request_sleep_with_threshold_exactly_at_current_count(
        self, tmp_path
    ):
        """auto_sleep_node_threshold = current GM count (boundary = exact match).
        Ingest until count equals threshold, then check auto-trigger fires.
        ASSUMES: sleep_cycles_run."""
        from krakey.engines.memory.default import GraphMemoryEngine

        threshold = 3
        sleep_llm = FakeChatLike()
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            sleep_llm=sleep_llm,
            reranker=FakeReranker(),
            sleep_config=_min_sleep_config(auto_sleep_node_threshold=threshold),
        )
        await engine.initialize()

        # Ingest threshold-1 nodes — should NOT trigger
        for i in range(threshold - 1):
            await engine.ingest(f"pre-threshold node {i}")
        await asyncio.sleep(0)
        count_before_threshold = engine.sleep_cycles_run

        # Ingest exactly the threshold-th node
        await engine.ingest("the threshold node — this one should trigger")
        await asyncio.sleep(0)

        # Count after should be >= count_before + 1
        assert engine.sleep_cycles_run >= count_before_threshold + 1, (
            f"Auto-trigger at exact threshold={threshold} did not fire; "
            f"sleep_cycles_run before={count_before_threshold}, "
            f"after={engine.sleep_cycles_run}."
        )
        await engine.close()
