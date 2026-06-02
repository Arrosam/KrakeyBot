"""Regression guards for two HIGH bugs caught in review of the
engine-decoupling work:

1. classify_and_link_pending was dropped — the heartbeat stopped scheduling
   it and nothing replaced it, so auto-ingested nodes were never classified.
   FIX: the memory engine's sleep cycle (enter_sleep_mode) now runs
   classify_and_link_pending as its first phase.

2. A failed / no-op sleep was reported to Self + the dashboard as a
   COMPLETED cycle. request_sleep returned {} for no-llm / coalesced /
   internal-failure alike, and trigger_memory_sleep treated {} as success.
   FIX: request_sleep RAISES on real pipeline failure (so the runtime hook
   surfaces SleepFailed + a corrective stimulus), and returns {} only for a
   genuine no-op.
"""
from __future__ import annotations

import pytest


class FakeEmbedder:
    async def __call__(self, text: str) -> list[float]:
        d = len(text) % 5
        return [0.1 + d * 0.1, 0.2, 0.3, 0.4]


class FakeChatLike:
    def __init__(self, *, response: str = ""):
        self._response = response
        self.calls: list = []

    async def chat(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        return self._response or '{"nodes": [], "edges": []}'


class FakeReranker:
    async def rerank(self, query: str, docs: list[str]) -> list[float]:
        return [0.5] * len(docs)


async def _engine(tmp_path, *, sleep_llm=None, reranker=None):
    from krakey.engines.memory.default import GraphMemoryEngine
    eng = GraphMemoryEngine(
        db_path=":memory:",
        embedder=FakeEmbedder(),
        kb_dir=str(tmp_path / "kbs"),
        sleep_llm=sleep_llm,
        reranker=reranker,
        sleep_config={"min_community_size": 1},
    )
    await eng.initialize()
    return eng


# ---------------------------------------------------------------------------
# Bug 1: classification runs during sleep
# ---------------------------------------------------------------------------

class TestClassifyRunsDuringSleep:
    async def test_request_sleep_invokes_classify_and_link_pending(
        self, tmp_path, monkeypatch,
    ):
        """The sleep cycle must drive classification — the engine, not the
        heartbeat, owns it now. Spy on classify_and_link_pending and assert
        request_sleep calls it."""
        eng = await _engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker(),
        )
        calls = {"n": 0}
        orig = eng.classify_and_link_pending

        async def _spy():
            calls["n"] += 1
            return await orig()

        monkeypatch.setattr(eng, "classify_and_link_pending", _spy)
        await eng.request_sleep("regression")
        assert calls["n"] >= 1, (
            "request_sleep must run classify_and_link_pending as part of the "
            "sleep cycle (the heartbeat no longer schedules it)."
        )
        await eng.close()

    async def test_classify_failure_does_not_abort_sleep(
        self, tmp_path, monkeypatch,
    ):
        """A classifier failure is best-effort — it must not abort the
        whole sleep cycle."""
        eng = await _engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker(),
        )

        async def _boom():
            raise RuntimeError("classifier down")

        monkeypatch.setattr(eng, "classify_and_link_pending", _boom)
        # Should NOT raise — classify is wrapped best-effort inside sleep.
        stats = await eng.request_sleep("regression")
        assert isinstance(stats, dict)
        await eng.close()


# ---------------------------------------------------------------------------
# Bug 2: failed / no-op sleep is not reported as a completed cycle
# ---------------------------------------------------------------------------

class TestSleepFailureNotReportedAsSuccess:
    async def test_request_sleep_raises_on_pipeline_failure(
        self, tmp_path, monkeypatch,
    ):
        """A real pipeline failure must RAISE (not be swallowed into {}),
        so callers can distinguish failure from a no-op."""
        eng = await _engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker(),
        )

        import krakey.engines.memory._internal.sleep.sleep_manager as sm

        async def _boom(*a, **k):
            raise RuntimeError("clustering exploded")

        monkeypatch.setattr(sm, "enter_sleep_mode", _boom)
        with pytest.raises(RuntimeError, match="clustering exploded"):
            await eng.request_sleep("regression")
        # The in-flight guard must have been reset despite the raise.
        assert eng._sleeping is False
        await eng.close()

    async def test_request_sleep_noop_returns_empty_when_no_llm(self, tmp_path):
        """No sleep_llm configured → genuine no-op → {} (NOT a raise, NOT a
        fake 'completed' dict)."""
        eng = await _engine(tmp_path, sleep_llm=None)
        assert await eng.request_sleep("regression") == {}
        await eng.close()

    async def test_request_sleep_returns_truthy_on_real_cycle(
        self, tmp_path,
    ):
        """A real (even empty-GM) cycle returns a TRUTHY stats dict so the
        runtime hook can tell 'a cycle ran' from a no-op {}."""
        eng = await _engine(
            tmp_path, sleep_llm=FakeChatLike(), reranker=FakeReranker(),
        )
        stats = await eng.request_sleep("regression")
        assert stats, "a real sleep cycle must return a non-empty stats dict"
        await eng.close()

    async def test_trigger_memory_sleep_surfaces_failure(
        self, tmp_path, monkeypatch,
    ):
        """runtime.trigger_memory_sleep must NOT report a failed sleep as a
        completed cycle: on a raise it publishes SleepFailedEvent + a
        corrective system:sleep stimulus, and does NOT bump _sleep_cycles."""
        from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes
        from krakey.runtime.events.event_bus import EventBus

        rt = build_runtime_with_fakes(self_llm=ScriptedLLM())
        await rt.memory.initialize()
        bus = EventBus()
        rt.events = bus
        received = []
        bus.subscribe(received.append)

        async def _boom(reason=""):
            raise RuntimeError("simulated sleep crash")

        monkeypatch.setattr(rt.memory, "request_sleep", _boom)
        before = rt._sleep_cycles
        await rt.trigger_memory_sleep("force-sleep at fatigue 130%")

        # Failure surfaced as an event...
        assert any(getattr(e, "kind", "") == "sleep_failed" for e in received), (
            "a failed sleep must publish SleepFailedEvent"
        )
        # ...and as a corrective stimulus to Self.
        drained = rt.buffer.drain()
        fails = [
            s for s in drained
            if s.type == "system_event" and s.source == "system:sleep"
            and "failed" in s.content.lower()
        ]
        assert fails, "a failed sleep must push a 'Sleep transition failed' stimulus"
        # ...and must NOT count as a completed cycle.
        assert rt._sleep_cycles == before, (
            "a failed sleep must not increment the completed-cycle counter"
        )
        await rt.close()

    async def test_trigger_memory_sleep_noop_is_not_a_completed_cycle(
        self, monkeypatch,
    ):
        """A genuine no-op ({} — no sleep_llm, a coalesced cycle, or a backend
        like MemOS that consolidates internally) must NOT be reported as a
        completed cycle: no SleepStart/SleepDone events, no _sleep_cycles bump,
        no 'Completed a sleep cycle' stimulus, and the recall session is
        preserved. This is what stops a no-op backend from spamming a phantom
        cycle every beat under force-sleep."""
        from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes
        from krakey.runtime.events.event_bus import EventBus

        rt = build_runtime_with_fakes(self_llm=ScriptedLLM())
        await rt.memory.initialize()
        bus = EventBus()
        rt.events = bus
        received = []
        bus.subscribe(received.append)

        async def _noop(reason=""):
            return {}

        monkeypatch.setattr(rt.memory, "request_sleep", _noop)
        # Spy on recall.new_session to prove the no-op doesn't discard it.
        resets = {"n": 0}
        orig_new_session = rt.recall.new_session

        def _spy_new_session(*a, **k):
            resets["n"] += 1
            return orig_new_session(*a, **k)

        monkeypatch.setattr(rt.recall, "new_session", _spy_new_session)

        before = rt._sleep_cycles
        rt.buffer.drain()  # clear any startup stimuli
        result = await rt.trigger_memory_sleep("force-sleep at fatigue 130%")

        assert result == {}
        # No lifecycle events at all for a no-op...
        kinds = [getattr(e, "kind", "") for e in received]
        assert "sleep_done" not in kinds, "a no-op must not publish SleepDoneEvent"
        assert "sleep_start" not in kinds, "a no-op must not publish SleepStartEvent"
        # ...no completed-cycle counter bump...
        assert rt._sleep_cycles == before, (
            "a no-op must not increment the completed-cycle counter"
        )
        # ...no 'Completed a sleep cycle' stimulus to Self...
        drained = rt.buffer.drain()
        completed = [
            s for s in drained
            if s.type == "system_event" and s.source == "system:sleep"
            and "completed a sleep cycle" in s.content.lower()
        ]
        assert not completed, (
            "a no-op must not push a 'Completed a sleep cycle' stimulus"
        )
        # ...and the recall session is preserved (not thrown away).
        assert resets["n"] == 0, "a no-op must not reset the recall session"
        await rt.close()
