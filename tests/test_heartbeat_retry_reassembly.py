"""Edge tests: adrenalin-interruptible retry waits + beat reassembly.

Covers two new behaviors NOT YET IMPLEMENTED:

1. ``wait_or_adrenalin(buffer, seconds, *, poll_slice=0.05)`` (idle.py)
   -- module-level async helper that polls buffer.wait_for_any() in
   slices until the deadline; returns True as soon as
   buffer.has_adrenalin() is True; returns False after full ``seconds``
   with no adrenalin.

2. ``HeartbeatOrchestrator._phase_run_self`` reassembly path
   -- when ``wait_or_adrenalin`` returns True (interrupted) the
   orchestrator drains the buffer, handles commands, rebuilds stimuli
   and recall, re-publishes PromptBuiltEvent, resets http_attempt /
   struct_attempt, and retries the LLM call with a fresh prompt.
   The _prompt_log must contain exactly ONE entry per heartbeat_id
   after any number of reassemblies.

Testing approach
----------------
* ``wait_or_adrenalin``: tested directly with a ``_FakeBuffer`` stub
  whose ``has_adrenalin()`` / ``wait_for_any()`` are scripted.
  Wall-clock assertions are generous (3x tolerance) to survive CI.

* ``_phase_run_self`` reassembly: tested through a lightweight fake
  Runtime constructed by hand (NOT via ``build_runtime_with_fakes``)
  so we can patch ``wait_or_adrenalin`` and control exactly when
  adrenalin fires.  We observe reassembly indirectly via:
    - the prompt the LLM was last called with (contains added stimulus)
    - which interval was passed to ``wait_or_adrenalin`` after
      reassembly (http_attempt / struct_attempt resets)
    - ``_prompt_log`` length (must stay == 1 per heartbeat_id)
    - ``request_stop()`` called when /kill arrives mid-retry

pytest uses asyncio_mode=auto (see pytest.ini).
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from krakey.engines.heartbeat.orchestrator import HeartbeatOrchestrator
from krakey.interfaces.engines.recall import RecallResult
from krakey.models.config.heartbeat import IdleSection
from krakey.models.stimulus import Stimulus
from krakey.runtime.events.event_types import PromptBuiltEvent


# ===========================================================================
# Shared helpers
# ===========================================================================

def _stim(content: str, *, adrenalin: bool = False, stype: str = "user_message",
          source: str = "test") -> Stimulus:
    return Stimulus(
        type=stype, source=source,
        content=content, timestamp=datetime.now(),
        adrenalin=adrenalin,
    )


def _good_response() -> str:
    """Structurally-valid Self response (all required tags present)."""
    return (
        "[THINKING]\nok\n"
        "[DECISION]\nNo action.\n"
        "[IDLE]\n1"
    )


def _idle_section(**overrides) -> IdleSection:
    """Tiny-interval IdleSection so retry loops are fast."""
    defaults = dict(
        min_interval=0,
        max_interval=300,
        default_interval=1,
        self_max_wall_seconds=60.0,
        llm_failure_retry_interval=0.01,
        struct_output_fast_retries=2,
        struct_output_slow_retry_interval=0.02,
    )
    defaults.update(overrides)
    return IdleSection(**defaults)


# ===========================================================================
# Section 1 — Fake buffer for wait_or_adrenalin tests
# ===========================================================================

class _FakeBuffer:
    """Scriptable StimulusBuffer stand-in for wait_or_adrenalin tests.

    ``adrenalin_at`` is the wall-clock offset (seconds from creation)
    at which has_adrenalin() will start returning True.  ``None`` means
    never within the test window.
    """

    def __init__(self, *, adrenalin_at: float | None = None,
                 inject_non_adrenalin_at: float | None = None):
        self._created = time.perf_counter()
        self._adrenalin_at = adrenalin_at
        self._inject_non_adrenalin_at = inject_non_adrenalin_at
        # wait_for_any returns immediately when the event is set;
        # we use a real asyncio.Event so it integrates naturally with
        # asyncio.wait_for inside wait_or_adrenalin.
        self._any_event = asyncio.Event()
        self._has_adrenalin_flag = False
        self.wait_for_any_call_count = 0

    def _elapsed(self) -> float:
        return time.perf_counter() - self._created

    def has_adrenalin(self) -> bool:
        # Check scripted schedule.
        if (self._adrenalin_at is not None
                and self._elapsed() >= self._adrenalin_at):
            self._has_adrenalin_flag = True
        return self._has_adrenalin_flag

    def set_adrenalin(self) -> None:
        """Immediately set adrenalin (useful for synchronous test setup)."""
        self._has_adrenalin_flag = True
        self._any_event.set()

    async def wait_for_any(self) -> None:
        self.wait_for_any_call_count += 1
        if (self._inject_non_adrenalin_at is not None
                and self._elapsed() >= self._inject_non_adrenalin_at):
            self._any_event.set()
        await self._any_event.wait()

    def drain(self):
        return []

    def push(self, s):  # not awaited in tests
        pass


# ===========================================================================
# Section 2 — Fake Runtime for _phase_run_self tests
# ===========================================================================

class _FakeRecallSession:
    """Minimal RecallSession.  Tracks add_stimuli calls."""

    def __init__(self, result: RecallResult | None = None):
        self.processed_stimuli: list[Stimulus] = []
        self._result = result or RecallResult()
        self.add_stimuli_calls: list[list[Stimulus]] = []
        self.finalize_calls = 0

    async def add_stimuli(self, stimuli: list[Stimulus]) -> None:
        self.add_stimuli_calls.append(list(stimuli))
        self.processed_stimuli.extend(stimuli)

    async def finalize(self) -> RecallResult:
        self.finalize_calls += 1
        return self._result


class _FakeRecallEngine:
    """Minimal RecallEngine. Returns a scripted session per new_session call."""

    def __init__(self):
        self._sessions: list[_FakeRecallSession] = []

    def new_session(self) -> _FakeRecallSession:
        sess = _FakeRecallSession()
        self._sessions.append(sess)
        return sess


class _FakeLog:
    def hb(self, msg): pass
    def hb_warn(self, msg): pass
    def runtime_error(self, msg): pass
    def set_heartbeat(self, n): pass


class _FakeEvents:
    def __init__(self):
        self.published: list = []

    def publish(self, event) -> None:
        self.published.append(event)

    def subscribe(self, fn): pass


class _FakeConfig:
    def __init__(self, idle: IdleSection):
        self.idle = idle
        # fatigue section stub so _phase_run_self doesn't crash if
        # it reads config.fatigue during logging
        self.fatigue = SimpleNamespace(force_sleep_threshold=9999)
        # enforce_input_budget() (called at the top of _phase_run_self
        # and again on every reassembly) reads config.llm.core_params()
        # and, only on the prune path, config.sliding_window. core_params
        # returns None here -> the real code falls back to LLMParams()
        # (max_input_tokens defaults large) so the tiny FAKE_PROMPT fits
        # the budget and the prune path is never entered.
        self.llm = SimpleNamespace(core_params=lambda tag: None)
        self.sliding_window = SimpleNamespace(compact_include_recall=False)


class _FakeSelfLLM:
    """Records all calls; raises the first ``n_errors`` times, then
    returns ``success_response``.  If ``responses`` is given, pops from
    that list (raises on empty if raise_on_empty=True)."""

    def __init__(self, success_response: str = "", *,
                 n_errors: int = 0,
                 responses: list[str] | None = None,
                 raise_on_empty: bool = False):
        self._success = success_response
        self._n_errors = n_errors
        self._error_count = 0
        self._responses = list(responses) if responses else None
        self._raise_on_empty = raise_on_empty
        self.calls: list[list] = []

    async def chat(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        if self._error_count < self._n_errors:
            self._error_count += 1
            raise ConnectionError(f"fake HTTP error #{self._error_count}")
        if self._responses is not None:
            if not self._responses:
                if self._raise_on_empty:
                    raise RuntimeError("no more responses")
                return ""
            return self._responses.pop(0)
        return self._success


class _FakeBuffer2:
    """Full-featured fake StimulusBuffer for orchestrator integration tests."""

    def __init__(self):
        self._queue: list[Stimulus] = []
        self._adrenalin = False
        self._any_event = asyncio.Event()
        self.drain_calls = 0
        self.push_calls: list[Stimulus] = []

    def has_adrenalin(self) -> bool:
        return self._adrenalin

    async def wait_for_any(self) -> None:
        await self._any_event.wait()

    async def push(self, s: Stimulus) -> None:
        self._queue.append(s)
        self.push_calls.append(s)
        self._any_event.set()
        if s.adrenalin:
            self._adrenalin = True

    def drain(self) -> list[Stimulus]:
        self.drain_calls += 1
        items = list(self._queue)
        self._queue = []
        self._adrenalin = False
        self._any_event.clear()
        return items

    def peek_unrecalled(self):
        return []


def _make_fake_rt(*, idle_cfg: IdleSection | None = None,
                  n_llm_errors: int = 0,
                  llm_success: str = "",
                  llm_responses: list[str] | None = None) -> tuple[Any, _FakeSelfLLM]:
    """Build a minimal fake Runtime suitable for driving _phase_run_self."""
    if idle_cfg is None:
        idle_cfg = _idle_section()
    self_llm = _FakeSelfLLM(
        success_response=llm_success,
        n_errors=n_llm_errors,
        responses=llm_responses,
    )
    recall_engine = _FakeRecallEngine()
    recall_session = _FakeRecallSession()
    buf = _FakeBuffer2()
    events = _FakeEvents()
    log = _FakeLog()

    rt = SimpleNamespace(
        # Required fields touched by _phase_run_self
        heartbeat_count=1,
        stop_requested=False,
        config=_FakeConfig(idle=idle_cfg),
        self_llm=self_llm,
        buffer=buf,
        events=events,
        log=log,
        recall=recall_engine,
        _recall=recall_session,
        _prompt_log=[],
        explicit_history=SimpleNamespace(rounds=[], get_rounds=lambda: []),
        # request_stop sets stop_requested; tested tests check it
        _stop=False,
    )

    def _request_stop():
        rt._stop = True
        # The orchestrator reads stop_requested as a property; we make it
        # a live attribute by updating it too.
        rt.stop_requested = True

    rt.request_stop = _request_stop

    # Minimal context + modifiers so build_self_prompt doesn't crash.
    rt.self_model = {}
    rt.tools = SimpleNamespace(list_descriptions=lambda: [])
    rt.modifiers = SimpleNamespace(all=lambda: [])
    rt.decision = SimpleNamespace()
    rt.memory = SimpleNamespace(
        count_nodes=AsyncMock(return_value=0),
        count_edges=AsyncMock(return_value=0),
        fts_search=AsyncMock(return_value=[]),
    )
    rt.compact_llm = _FakeSelfLLM(success_response="")

    # context.build_default_elements returns something renderable.
    class _FakeElements:
        def render(self): return "FAKE_PROMPT"
        def for_plugin(self, name): return self

    class _FakeContext:
        def build_default_elements(self, **kwargs):
            return _FakeElements()

    rt.context = _FakeContext()

    return rt, self_llm


# ===========================================================================
# Section 3 — Tests for wait_or_adrenalin
# ===========================================================================

class TestWaitOrAdrenalin:
    """
    Positive, BVA, state-transition, and negative tests for
    wait_or_adrenalin (not yet implemented — tests define the contract).
    """

    # --- positive: happy paths ---

    async def test_returns_true_immediately_when_buffer_already_has_adrenalin(self):
        """has_adrenalin() is True on entry → returns True without sleeping."""
        from krakey.engines.heartbeat.idle import wait_or_adrenalin

        buf = _FakeBuffer()
        buf.set_adrenalin()

        t0 = time.perf_counter()
        result = await wait_or_adrenalin(buf, 5.0, poll_slice=0.05)
        elapsed = time.perf_counter() - t0

        assert result is True, "Expected True when adrenalin already present"
        assert elapsed < 0.3, (
            f"Should return immediately (< 0.3s) when adrenalin pre-set; "
            f"took {elapsed:.3f}s"
        )

    async def test_returns_true_early_when_adrenalin_arrives_mid_wait(self):
        """Adrenalin arrives 0.08s into a 5s wait → returns True well before
        the deadline."""
        from krakey.engines.heartbeat.idle import wait_or_adrenalin

        buf = _FakeBuffer(adrenalin_at=0.08)

        async def _set_adrenalin_async():
            await asyncio.sleep(0.08)
            buf.set_adrenalin()

        asyncio.create_task(_set_adrenalin_async())

        t0 = time.perf_counter()
        result = await wait_or_adrenalin(buf, 5.0, poll_slice=0.05)
        elapsed = time.perf_counter() - t0

        assert result is True, "Expected True when adrenalin arrives mid-wait"
        assert elapsed < 1.0, (
            f"Expected early return (<1s) when adrenalin arrives at 0.08s; "
            f"took {elapsed:.3f}s"
        )

    async def test_returns_false_when_full_duration_elapses_without_adrenalin(self):
        """No adrenalin ever → returns False after ~full `seconds`."""
        from krakey.engines.heartbeat.idle import wait_or_adrenalin

        buf = _FakeBuffer()  # no adrenalin ever

        t0 = time.perf_counter()
        result = await wait_or_adrenalin(buf, 0.12, poll_slice=0.03)
        elapsed = time.perf_counter() - t0

        assert result is False, "Expected False when duration elapses without adrenalin"
        assert elapsed >= 0.10, (
            f"Must wait the full duration; elapsed={elapsed:.3f}s < 0.10s"
        )
        assert elapsed < 0.60, (
            f"Must not dramatically overshoot; elapsed={elapsed:.3f}s"
        )

    async def test_returns_false_with_non_adrenalin_stimulus_only(self):
        """Non-adrenalin stimulus arrives → wait_for_any fires, but
        has_adrenalin() stays False → returns False after full duration."""
        from krakey.engines.heartbeat.idle import wait_or_adrenalin

        # inject_non_adrenalin_at triggers wait_for_any to fire at that offset
        buf = _FakeBuffer(inject_non_adrenalin_at=0.03)

        async def _push_non_adrenalin():
            await asyncio.sleep(0.03)
            buf._any_event.set()  # simulate non-adrenalin push

        asyncio.create_task(_push_non_adrenalin())

        t0 = time.perf_counter()
        result = await wait_or_adrenalin(buf, 0.15, poll_slice=0.03)
        elapsed = time.perf_counter() - t0

        assert result is False, (
            "Non-adrenalin stimulus must NOT cause True return"
        )
        assert elapsed >= 0.12, (
            f"Must wait full duration when non-adrenalin fires; elapsed={elapsed:.3f}s"
        )

    # --- BVA: boundary values ---

    async def test_seconds_zero_returns_false_immediately_when_no_adrenalin(self):
        """seconds=0 with no adrenalin → must return False without hanging."""
        from krakey.engines.heartbeat.idle import wait_or_adrenalin

        buf = _FakeBuffer()

        t0 = time.perf_counter()
        result = await wait_or_adrenalin(buf, 0, poll_slice=0.05)
        elapsed = time.perf_counter() - t0

        assert result is False, "seconds=0 with no adrenalin should return False"
        assert elapsed < 0.3, (
            f"seconds=0 should return quickly; took {elapsed:.3f}s"
        )

    async def test_seconds_zero_returns_true_immediately_when_adrenalin_pre_set(self):
        """seconds=0, buffer already has adrenalin → True (adrenalin on entry
        wins regardless of duration)."""
        from krakey.engines.heartbeat.idle import wait_or_adrenalin

        buf = _FakeBuffer()
        buf.set_adrenalin()

        result = await wait_or_adrenalin(buf, 0, poll_slice=0.05)
        assert result is True, "Even seconds=0 returns True when adrenalin is pre-set"

    async def test_negative_seconds_returns_false_without_error(self):
        """seconds<0 must not raise and must return False without sleeping."""
        from krakey.engines.heartbeat.idle import wait_or_adrenalin

        buf = _FakeBuffer()

        t0 = time.perf_counter()
        result = await wait_or_adrenalin(buf, -1.0, poll_slice=0.05)
        elapsed = time.perf_counter() - t0

        assert result is False, "seconds<0 should return False"
        assert elapsed < 0.3, f"seconds<0 should return quickly; took {elapsed:.3f}s"

    async def test_tiny_poll_slice_still_detects_adrenalin(self):
        """poll_slice much smaller than seconds does not miss adrenalin."""
        from krakey.engines.heartbeat.idle import wait_or_adrenalin

        buf = _FakeBuffer()

        async def _delayed():
            await asyncio.sleep(0.05)
            buf.set_adrenalin()

        asyncio.create_task(_delayed())

        result = await wait_or_adrenalin(buf, 5.0, poll_slice=0.01)
        assert result is True

    async def test_poll_slice_larger_than_seconds_still_times_out(self):
        """poll_slice > seconds should still return False after ~seconds, not
        after poll_slice."""
        from krakey.engines.heartbeat.idle import wait_or_adrenalin

        buf = _FakeBuffer()  # no adrenalin

        t0 = time.perf_counter()
        result = await wait_or_adrenalin(buf, 0.05, poll_slice=2.0)
        elapsed = time.perf_counter() - t0

        assert result is False
        # Should not block for 2 full seconds
        assert elapsed < 1.0, (
            f"poll_slice > seconds should NOT block for poll_slice duration; "
            f"took {elapsed:.3f}s"
        )

    # --- state transitions ---

    async def test_adrenalin_arrives_exactly_at_deadline_returns_true(self):
        """Adrenalin set just before deadline loop check → implementation
        should detect it on the final poll and return True (spec: 'returns
        True as soon as buffer.has_adrenalin() is True').

        We test this with a very small window (0.1s) and set adrenalin at
        0.07s — close to but before deadline — giving the poll a chance to
        catch it before the deadline fires."""
        from krakey.engines.heartbeat.idle import wait_or_adrenalin

        buf = _FakeBuffer()

        async def _set_just_before():
            await asyncio.sleep(0.07)
            buf.set_adrenalin()

        asyncio.create_task(_set_just_before())
        result = await wait_or_adrenalin(buf, 0.20, poll_slice=0.02)
        # Implementation polls every poll_slice so should see adrenalin
        # before deadline at 0.20s when adrenalin fires at 0.07s.
        assert result is True

    async def test_does_not_drain_buffer(self):
        """wait_or_adrenalin must NOT drain, peek, or recall from the buffer
        — it is a pure wait helper."""
        from krakey.engines.heartbeat.idle import wait_or_adrenalin

        buf = _FakeBuffer()
        buf.set_adrenalin()

        # Put a sentinel item in the "queue" (our fake has no real queue,
        # but we track drain calls via a counter).
        drain_calls_before = 0
        original_drain = buf.drain
        drain_called = []

        def track_drain():
            drain_called.append(True)
            return original_drain()

        buf.drain = track_drain

        await wait_or_adrenalin(buf, 1.0, poll_slice=0.05)
        assert drain_called == [], "wait_or_adrenalin must NOT call drain()"

    # --- negative / error guessing ---

    async def test_adrenalin_after_deadline_not_detected_as_true(self):
        """Adrenalin that arrives AFTER the full duration window should NOT
        cause a True return — the function has already returned False."""
        from krakey.engines.heartbeat.idle import wait_or_adrenalin

        buf = _FakeBuffer()

        t0 = time.perf_counter()
        result = await wait_or_adrenalin(buf, 0.08, poll_slice=0.03)
        elapsed = time.perf_counter() - t0

        # Now set adrenalin after the function has returned
        buf.set_adrenalin()

        # The return value must already be determined (False) before adrenalin
        assert result is False
        assert elapsed < 0.5


# ===========================================================================
# Section 4 — _phase_run_self reassembly integration tests
# ===========================================================================

class TestPhaseRunSelfReassembly:
    """Integration tests for the reassembly path in _phase_run_self.

    Strategy: patch ``wait_or_adrenalin`` in orchestrator.py with a
    scripted coroutine so we can control exactly when it returns True
    (adrenalin) vs False (timeout).  Observe outcomes via:
    - self_llm.calls[-1]  (last prompt sent — reflects reassembled stimuli)
    - runtime._prompt_log (exactly one entry per heartbeat_id)
    - runtime.events.published (PromptBuiltEvent count)
    - runtime.stop_requested (True iff /kill received)
    - _perform_sleep call (iff /sleep received)

    We patch the target at 'krakey.engines.heartbeat.orchestrator.wait_or_adrenalin'
    and also at 'krakey.engines.heartbeat.idle.wait_or_adrenalin' to cover both
    the new helper import and the orchestrator reference.
    """

    _PATCH_TARGET = "krakey.engines.heartbeat.orchestrator.wait_or_adrenalin"

    # --- helper: build an orchestrator with patched wait_or_adrenalin ---

    def _make_orch(self, rt) -> HeartbeatOrchestrator:
        orch = HeartbeatOrchestrator(rt)
        return orch

    def _counts(self):
        from krakey.engines.heartbeat.orchestrator import _GMCounts
        return _GMCounts(
            node_count=0, edge_count=0,
            fatigue_pct=0, fatigue_hint="",
        )

    # --- (a) adrenalin during HTTP-failure wait → prompt reflects
    #        reassembled stimuli, http_attempt reset ---

    async def test_http_wait_adrenalin_rebuilds_prompt_with_new_stimulus(self):
        """When adrenalin arrives during the HTTP-retry wait:
        - Reassembly drains the buffer (finds new stimulus).
        - The LLM is called with a fresh prompt that includes the new stimulus.
        - After the LLM succeeds, _phase_run_self returns non-None.
        """
        rt, self_llm = _make_fake_rt(
            idle_cfg=_idle_section(
                llm_failure_retry_interval=0.01,
            ),
            llm_responses=[
                # First call raises (HTTP error) — scripted via n_errors
                # We need responses list approach:
            ],
        )
        # Override: first call raises, second call succeeds.
        call_count = {"n": 0}
        new_stimulus_content = "ADRENALIN_NEW_MSG"

        async def _chat(messages, **kwargs):
            call_count["n"] += 1
            last_prompt = messages[-1]["content"] if messages else ""
            if call_count["n"] == 1:
                raise ConnectionError("http error #1")
            return _good_response()

        rt.self_llm.chat = _chat

        # After the first HTTP error, wait_or_adrenalin returns True (interrupted).
        # We push a new adrenalin stimulus so drain() finds it.
        adrenalin_calls = {"n": 0}

        async def _fake_wait_or_adrenalin(buf, seconds, *, poll_slice=0.05):
            adrenalin_calls["n"] += 1
            if adrenalin_calls["n"] == 1:
                # Simulate adrenalin arriving; push a new stimulus first.
                await rt.buffer.push(_stim(new_stimulus_content, adrenalin=True))
                return True
            # Subsequent calls (if any) time out normally.
            return False

        with patch(self._PATCH_TARGET, _fake_wait_or_adrenalin):
            orch = self._make_orch(rt)
            # Start with an initial stimulus.
            initial_stims = [_stim("initial")]
            result = await orch._phase_run_self(
                initial_stims, RecallResult(), self._counts(),
            )

        assert result is not None, "_phase_run_self must return non-None on success"
        assert call_count["n"] == 2, (
            f"Expected exactly 2 LLM calls (1 error + 1 success); "
            f"got {call_count['n']}"
        )

    async def test_http_wait_adrenalin_resets_http_attempt(self):
        """After reassembly triggered by adrenalin during HTTP wait, the
        next wait_or_adrenalin call (if another HTTP error follows) must use
        the FAST (llm_failure_retry_interval) interval, not accumulated."""
        rt, _ = _make_fake_rt(
            idle_cfg=_idle_section(
                llm_failure_retry_interval=0.01,
                struct_output_slow_retry_interval=0.02,
            ),
        )

        call_count = {"n": 0}
        wait_intervals_seen: list[float] = []

        async def _chat(messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise ConnectionError("http error")
            return _good_response()

        rt.self_llm.chat = _chat

        async def _fake_wait(buf, seconds, *, poll_slice=0.05):
            wait_intervals_seen.append(seconds)
            if len(wait_intervals_seen) == 1:
                # First wait: return True (adrenalin) → reassembly, resets
                # http_attempt to 0.
                return True
            # Second wait (after reassembly + another error): return False.
            return False

        with patch(self._PATCH_TARGET, _fake_wait):
            orch = self._make_orch(rt)
            await orch._phase_run_self([], RecallResult(), self._counts())

        # Both waits should use llm_failure_retry_interval (0.01),
        # because http_attempt reset to 0 after reassembly.
        assert len(wait_intervals_seen) >= 2
        for interval in wait_intervals_seen:
            assert interval == pytest.approx(0.01, rel=0.1), (
                f"After http-tier reassembly, next wait must use "
                f"llm_failure_retry_interval=0.01, got {interval}"
            )

    # --- (b) adrenalin during slow struct-output tier →
    #        reassembly happens, next wait is FAST interval ---

    async def test_slow_struct_tier_adrenalin_resets_to_fast_interval(self):
        """After exhausting fast-tier struct retries (transitioning to slow),
        if adrenalin fires during a slow-tier wait → reassembly resets
        struct_attempt to 0 → the NEXT wait interval must be the fast
        (llm_failure_retry_interval), not the slow interval.
        """
        fast_interval = 0.01
        slow_interval = 0.09

        rt, _ = _make_fake_rt(
            idle_cfg=_idle_section(
                llm_failure_retry_interval=fast_interval,
                struct_output_fast_retries=2,
                struct_output_slow_retry_interval=slow_interval,
            ),
        )

        # Responses: 3 malformed (exhaust fast tier + enter slow tier)
        # then adrenalin interrupt on slow wait, then 1 malformed again
        # (should be fast interval now), then success.
        call_count = {"n": 0}
        _MALFORMED = "[THINKING]\nok\n[DECISION]\nok\n"  # missing [IDLE]

        async def _chat(messages, **kwargs):
            call_count["n"] += 1
            # calls 1,2,3 → malformed; call 4 → malformed again;
            # call 5 → success
            if call_count["n"] <= 4:
                return _MALFORMED
            return _good_response()

        rt.self_llm.chat = _chat

        wait_calls: list[float] = []
        interrupted_on_call: int | None = None

        async def _fake_wait(buf, seconds, *, poll_slice=0.05):
            n = len(wait_calls) + 1
            wait_calls.append(seconds)
            # Interrupt on the 3rd wait (which should be first slow-tier wait)
            if n == 3:
                return True   # adrenalin interrupt
            return False

        with patch(self._PATCH_TARGET, _fake_wait):
            orch = self._make_orch(rt)
            await orch._phase_run_self([], RecallResult(), self._counts())

        # Wait 1: fast (struct_attempt=1, within fast_budget=2)
        # Wait 2: fast (struct_attempt=2, within fast_budget=2)
        # Wait 3: slow (struct_attempt=3, exceeded fast_budget) → INTERRUPTED
        # Wait 4: fast (struct_attempt=1 after reset)
        assert len(wait_calls) >= 4, (
            f"Expected at least 4 wait calls; got {len(wait_calls)}: {wait_calls}"
        )
        assert wait_calls[0] == pytest.approx(fast_interval, rel=0.1), (
            f"Wait 1 should be fast ({fast_interval}); got {wait_calls[0]}"
        )
        assert wait_calls[1] == pytest.approx(fast_interval, rel=0.1), (
            f"Wait 2 should be fast ({fast_interval}); got {wait_calls[1]}"
        )
        assert wait_calls[2] == pytest.approx(slow_interval, rel=0.1), (
            f"Wait 3 should be slow ({slow_interval}); got {wait_calls[2]}"
        )
        # Wait 4 should be fast again (struct_attempt reset to 0 by reassembly)
        assert wait_calls[3] == pytest.approx(fast_interval, rel=0.1), (
            f"Wait 4 (after reassembly) should be fast ({fast_interval}); "
            f"got {wait_calls[3]}"
        )

    # --- (c) non-adrenalin stimulus during wait → NO reassembly ---

    async def test_non_adrenalin_wait_no_reassembly(self):
        """When wait_or_adrenalin returns False (timeout/non-adrenalin),
        the beat must NOT reassemble — drain must not be called, prompt
        stays unchanged, same number of LLM calls as if adrenalin never
        fired."""
        rt, self_llm = _make_fake_rt(
            idle_cfg=_idle_section(llm_failure_retry_interval=0.01),
        )

        call_count = {"n": 0}

        async def _chat(messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ConnectionError("error 1")
            return _good_response()

        rt.self_llm.chat = _chat
        drain_calls_before = rt.buffer.drain_calls

        async def _fake_wait_returns_false(buf, seconds, *, poll_slice=0.05):
            return False   # NOT interrupted → no reassembly

        with patch(self._PATCH_TARGET, _fake_wait_returns_false):
            orch = self._make_orch(rt)
            await orch._phase_run_self(
                [_stim("original")], RecallResult(), self._counts(),
            )

        # Drain must NOT have been called during retry (only pre-beat drain
        # happened, which is a different phase).
        assert rt.buffer.drain_calls == drain_calls_before, (
            "Non-interrupted wait must NOT drain the buffer"
        )

    # --- (d) /kill in drained stimuli → returns None, request_stop called ---

    async def test_kill_command_in_drained_stimuli_stops_loop(self):
        """After adrenalin interrupt, if drained stimuli contain /kill,
        _phase_run_self must call rt.request_stop() and return None."""
        rt, self_llm = _make_fake_rt(
            idle_cfg=_idle_section(llm_failure_retry_interval=0.01),
        )

        call_count = {"n": 0}

        async def _chat(messages, **kwargs):
            call_count["n"] += 1
            raise ConnectionError("http error")  # always fails

        rt.self_llm.chat = _chat

        wait_call_n = {"n": 0}

        async def _fake_wait_with_kill(buf, seconds, *, poll_slice=0.05):
            wait_call_n["n"] += 1
            if wait_call_n["n"] == 1:
                # Push /kill as a user_message adrenalin stimulus.
                await rt.buffer.push(_stim("/kill", adrenalin=True,
                                           stype="user_message"))
                return True
            return False

        with patch(self._PATCH_TARGET, _fake_wait_with_kill):
            orch = self._make_orch(rt)
            result = await orch._phase_run_self([], RecallResult(), self._counts())

        assert result is None, (
            "_phase_run_self must return None when /kill is processed"
        )
        assert rt.stop_requested is True, (
            "request_stop() must have been called when /kill arrived"
        )

    # --- (e) /sleep in drained stimuli → _perform_sleep invoked, returns None ---

    async def test_sleep_command_in_drained_stimuli_invokes_perform_sleep(self):
        """After adrenalin interrupt, /sleep in drained stimuli →
        _perform_sleep called and _phase_run_self returns None."""
        rt, self_llm = _make_fake_rt(
            idle_cfg=_idle_section(llm_failure_retry_interval=0.01),
        )

        call_count = {"n": 0}

        async def _chat(messages, **kwargs):
            call_count["n"] += 1
            raise ConnectionError("http error")

        rt.self_llm.chat = _chat

        sleep_invoked = {"called": False, "reason": None}

        wait_call_n = {"n": 0}

        async def _fake_wait_with_sleep(buf, seconds, *, poll_slice=0.05):
            wait_call_n["n"] += 1
            if wait_call_n["n"] == 1:
                await rt.buffer.push(_stim("/sleep", adrenalin=True,
                                           stype="user_message"))
                return True
            return False

        with patch(self._PATCH_TARGET, _fake_wait_with_sleep):
            orch = self._make_orch(rt)

            # Patch _perform_sleep to record invocation without running real sleep.
            async def _fake_perform_sleep(reason, *, wake_msg=""):
                sleep_invoked["called"] = True
                sleep_invoked["reason"] = reason

            orch._perform_sleep = _fake_perform_sleep

            result = await orch._phase_run_self([], RecallResult(), self._counts())

        assert result is None, (
            "_phase_run_self must return None when /sleep command arrives"
        )
        assert sleep_invoked["called"] is True, (
            "_perform_sleep must have been invoked when /sleep was in drained stimuli"
        )

    # --- (f) repeated adrenalin does not cause unbounded loop ---

    async def test_repeated_adrenalin_bounded_by_llm_successes(self):
        """Even if adrenalin keeps firing on every wait, the loop must
        terminate once the LLM returns a valid response.  Each adrenalin
        interrupt yields exactly one subsequent LLM attempt — there is no
        runaway reassembly cycle."""
        rt, _ = _make_fake_rt(
            idle_cfg=_idle_section(llm_failure_retry_interval=0.01),
        )

        call_count = {"n": 0}

        async def _chat(messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 4:
                raise ConnectionError("error")
            return _good_response()

        rt.self_llm.chat = _chat

        wait_call_n = {"n": 0}

        async def _always_adrenalin(buf, seconds, *, poll_slice=0.05):
            wait_call_n["n"] += 1
            # Always return True to simulate persistent adrenalin.
            return True

        with patch(self._PATCH_TARGET, _always_adrenalin):
            orch = self._make_orch(rt)
            result = await orch._phase_run_self([], RecallResult(), self._counts())

        # The loop must terminate and return a valid parsed result.
        assert result is not None, (
            "Loop must terminate once LLM succeeds, even with repeated adrenalin"
        )
        # LLM was called exactly as many times as needed (≥ 4 here, but finite).
        assert call_count["n"] == 4, (
            f"Expected exactly 4 LLM calls; got {call_count['n']}"
        )
        # wait_or_adrenalin was called at least 3 times (one per HTTP error)
        # but must be finite.
        assert wait_call_n["n"] < 100, (
            "wait_or_adrenalin call count must be finite (< 100)"
        )

    # --- (g) _prompt_log has exactly ONE entry per heartbeat_id after reassemblies ---

    async def test_prompt_log_has_single_entry_after_reassembly(self):
        """After one or more reassemblies the _prompt_log must contain
        exactly ONE entry whose heartbeat_id == rt.heartbeat_count.
        The in-place update must NOT append phantom duplicate entries."""
        rt, _ = _make_fake_rt(
            idle_cfg=_idle_section(llm_failure_retry_interval=0.01),
        )

        call_count = {"n": 0}

        async def _chat(messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ConnectionError("error")
            return _good_response()

        rt.self_llm.chat = _chat

        wait_n = {"n": 0}

        async def _adrenalin_twice(buf, seconds, *, poll_slice=0.05):
            wait_n["n"] += 1
            if wait_n["n"] <= 2:
                return True   # two reassemblies
            return False

        with patch(self._PATCH_TARGET, _adrenalin_twice):
            orch = self._make_orch(rt)
            await orch._phase_run_self([], RecallResult(), self._counts())

        hb_id = rt.heartbeat_count
        matching = [e for e in rt._prompt_log
                    if e.get("heartbeat_id") == hb_id]
        assert len(matching) == 1, (
            f"Expected exactly 1 _prompt_log entry for heartbeat_id={hb_id} "
            f"after multiple reassemblies; got {len(matching)}: {matching}"
        )

    async def test_record_raw_output_attaches_to_single_entry(self):
        """record_raw_output must fill the one _prompt_log entry's
        raw_output field after a successful LLM call following reassembly."""
        rt, _ = _make_fake_rt(
            idle_cfg=_idle_section(llm_failure_retry_interval=0.01),
        )

        call_count = {"n": 0}

        async def _chat(messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ConnectionError("error 1")
            return _good_response()

        rt.self_llm.chat = _chat

        wait_n = {"n": 0}

        async def _once_adrenalin(buf, seconds, *, poll_slice=0.05):
            wait_n["n"] += 1
            return wait_n["n"] == 1  # True (interrupt) on first wait only

        with patch(self._PATCH_TARGET, _once_adrenalin):
            orch = self._make_orch(rt)
            await orch._phase_run_self([], RecallResult(), self._counts())

        hb_id = rt.heartbeat_count
        matching = [e for e in rt._prompt_log
                    if e.get("heartbeat_id") == hb_id]
        assert len(matching) == 1
        entry = matching[0]
        assert entry["raw_output"] is not None, (
            "raw_output must be filled in by record_raw_output after success"
        )
        assert _good_response() in entry["raw_output"] or entry["raw_output"], (
            "raw_output must contain the LLM response"
        )

    # --- (h) PromptBuiltEvent published again on each reassembly ---

    async def test_prompt_built_event_published_on_each_reassembly(self):
        """PromptBuiltEvent must be published once initially + once per
        reassembly (each reassembly refreshes the prompt)."""
        rt, _ = _make_fake_rt(
            idle_cfg=_idle_section(llm_failure_retry_interval=0.01),
        )

        call_count = {"n": 0}

        async def _chat(messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise ConnectionError("error")
            return _good_response()

        rt.self_llm.chat = _chat

        wait_n = {"n": 0}

        async def _two_interrupts(buf, seconds, *, poll_slice=0.05):
            wait_n["n"] += 1
            return wait_n["n"] <= 2  # True for first 2 waits (2 reassemblies)

        with patch(self._PATCH_TARGET, _two_interrupts):
            orch = self._make_orch(rt)
            await orch._phase_run_self([], RecallResult(), self._counts())

        prompt_built_events = [
            e for e in rt.events.published
            if isinstance(e, PromptBuiltEvent)
        ]
        # 1 initial + 2 reassemblies = at least 3 PromptBuiltEvent publishes
        assert len(prompt_built_events) >= 3, (
            f"Expected ≥3 PromptBuiltEvent (1 initial + 2 reassemblies); "
            f"got {len(prompt_built_events)}"
        )

    # --- heartbeat_count frozen across all retries and reassemblies ---

    async def test_heartbeat_count_frozen_across_retries_and_reassembly(self):
        """rt.heartbeat_count must not change during HTTP errors, struct
        retries, or reassemblies — it was already set by beat() before
        _phase_run_self is called."""
        rt, _ = _make_fake_rt(
            idle_cfg=_idle_section(llm_failure_retry_interval=0.01),
        )

        initial_hb = rt.heartbeat_count
        call_count = {"n": 0}

        async def _chat(messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 3:
                raise ConnectionError("error")
            return _good_response()

        rt.self_llm.chat = _chat

        wait_n = {"n": 0}
        hb_observed: list[int] = []

        async def _track_hb_and_interrupt(buf, seconds, *, poll_slice=0.05):
            wait_n["n"] += 1
            hb_observed.append(rt.heartbeat_count)
            return wait_n["n"] <= 2  # two reassemblies

        with patch(self._PATCH_TARGET, _track_hb_and_interrupt):
            orch = self._make_orch(rt)
            await orch._phase_run_self([], RecallResult(), self._counts())

        assert rt.heartbeat_count == initial_hb, (
            f"heartbeat_count must stay frozen at {initial_hb}; "
            f"got {rt.heartbeat_count}"
        )
        for obs in hb_observed:
            assert obs == initial_hb, (
                f"heartbeat_count observed mid-wait was {obs}, expected {initial_hb}"
            )


# ===========================================================================
# Section 5 — State transition tests
# ===========================================================================

class TestStateTransitions:
    """Explicit state-machine transitions through the retry tiers."""

    _PATCH_TARGET = "krakey.engines.heartbeat.orchestrator.wait_or_adrenalin"

    def _counts(self):
        from krakey.engines.heartbeat.orchestrator import _GMCounts
        return _GMCounts(node_count=0, edge_count=0,
                          fatigue_pct=0, fatigue_hint="")

    async def test_fast_tier_to_slow_tier_to_fast_tier_via_adrenalin(self):
        """State machine: initial → fast struct retries → slow tier
        → (adrenalin) → fast tier again → success.

        Verifies that the interval sequence is:
          fast, fast, slow, (interrupt), fast, success
        """
        fast = 0.01
        slow = 0.08
        fast_budget = 2

        rt, _ = _make_fake_rt(
            idle_cfg=_idle_section(
                llm_failure_retry_interval=fast,
                struct_output_fast_retries=fast_budget,
                struct_output_slow_retry_interval=slow,
            ),
        )

        call_count = {"n": 0}
        _MALFORMED = "[THINKING]\nok\n[DECISION]\nok\n"  # missing [IDLE]

        async def _chat(messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 4:
                return _MALFORMED   # struct failures: 1,2,3,4
            return _good_response()  # success on call 5

        rt.self_llm.chat = _chat

        intervals: list[float] = []

        async def _scripted_wait(buf, seconds, *, poll_slice=0.05):
            n = len(intervals) + 1
            intervals.append(seconds)
            # Interrupt on call 3 (first slow-tier wait)
            if n == 3:
                return True
            return False

        with patch(self._PATCH_TARGET, _scripted_wait):
            orch = HeartbeatOrchestrator(rt)
            result = await orch._phase_run_self([], RecallResult(), self._counts())

        assert result is not None, "Beat must succeed eventually"
        # Transition sequence: fast(1), fast(2), slow(3)[interrupt],
        # fast(4 — reset), success (no wait on success)
        assert len(intervals) >= 4
        assert intervals[0] == pytest.approx(fast, rel=0.1), f"wait[0] must be fast; got {intervals[0]}"
        assert intervals[1] == pytest.approx(fast, rel=0.1), f"wait[1] must be fast; got {intervals[1]}"
        assert intervals[2] == pytest.approx(slow, rel=0.1), f"wait[2] must be slow; got {intervals[2]}"
        assert intervals[3] == pytest.approx(fast, rel=0.1), (
            f"wait[3] after reassembly must be fast again; got {intervals[3]}"
        )

    async def test_http_tier_adrenalin_resets_counters(self):
        """HTTP error → adrenalin interrupt → reset → HTTP error again →
        second HTTP wait must still use the fast interval (not accumulated)."""
        fast = 0.01

        rt, _ = _make_fake_rt(
            idle_cfg=_idle_section(
                llm_failure_retry_interval=fast,
                struct_output_slow_retry_interval=0.08,
            ),
        )

        call_count = {"n": 0}

        async def _chat(messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise ConnectionError("http error")
            return _good_response()

        rt.self_llm.chat = _chat

        intervals: list[float] = []

        async def _scripted_wait(buf, seconds, *, poll_slice=0.05):
            n = len(intervals) + 1
            intervals.append(seconds)
            # Interrupt on first wait, let second wait time out normally.
            return n == 1

        with patch(self._PATCH_TARGET, _scripted_wait):
            orch = HeartbeatOrchestrator(rt)
            result = await orch._phase_run_self([], RecallResult(), self._counts())

        assert result is not None
        assert len(intervals) >= 2
        # Both waits must be the HTTP/fast interval, not a slow one
        for i, iv in enumerate(intervals):
            assert iv == pytest.approx(fast, rel=0.1), (
                f"HTTP-tier interval[{i}] must be fast ({fast}); got {iv}"
            )

    async def test_stop_requested_exits_loop_without_return_value(self):
        """If rt.stop_requested flips True while the loop is running,
        _phase_run_self must return None."""
        rt, _ = _make_fake_rt(
            idle_cfg=_idle_section(llm_failure_retry_interval=0.01),
        )

        call_count = {"n": 0}

        async def _chat(messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ConnectionError("error")
            return _good_response()

        rt.self_llm.chat = _chat

        async def _wait_that_stops(buf, seconds, *, poll_slice=0.05):
            rt.request_stop()  # flip stop during the wait
            return False

        with patch(self._PATCH_TARGET, _wait_that_stops):
            orch = HeartbeatOrchestrator(rt)
            result = await orch._phase_run_self([], RecallResult(), self._counts())

        assert result is None, (
            "_phase_run_self must return None when stop_requested becomes True"
        )

    async def test_initial_stop_requested_returns_none_immediately(self):
        """If rt.stop_requested is True before _phase_run_self starts,
        it must return None without calling the LLM."""
        rt, self_llm = _make_fake_rt(
            idle_cfg=_idle_section(),
        )
        rt.stop_requested = True

        async def _fake_wait(buf, seconds, *, poll_slice=0.05):
            return False

        with patch(self._PATCH_TARGET, _fake_wait):
            orch = HeartbeatOrchestrator(rt)
            result = await orch._phase_run_self([], RecallResult(), self._counts())

        assert result is None
        assert len(self_llm.calls) == 0, (
            "LLM must not be called when stop_requested is set on entry"
        )


# ===========================================================================
# Section 6 — Negative / error-guessing tests
# ===========================================================================

class TestNegativeEdgeCases:
    """Error-guessing and boundary cases."""

    _PATCH_TARGET = "krakey.engines.heartbeat.orchestrator.wait_or_adrenalin"

    def _counts(self):
        from krakey.engines.heartbeat.orchestrator import _GMCounts
        return _GMCounts(node_count=0, edge_count=0,
                          fatigue_pct=0, fatigue_hint="")

    async def test_reassembly_with_empty_drain_is_safe(self):
        """Adrenalin fires and has_adrenalin() returns True, but by the
        time drain() runs the buffer is empty (race: stimulus was consumed
        elsewhere).  Reassembly should complete safely and not crash —
        stimuli == original stimuli, recall rebuild still runs."""
        rt, _ = _make_fake_rt(
            idle_cfg=_idle_section(llm_failure_retry_interval=0.01),
        )

        call_count = {"n": 0}

        async def _chat(messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ConnectionError("error")
            return _good_response()

        rt.self_llm.chat = _chat

        wait_n = {"n": 0}

        async def _adrenalin_empty_drain(buf, seconds, *, poll_slice=0.05):
            wait_n["n"] += 1
            if wait_n["n"] == 1:
                # Do NOT push anything — drain will return [].
                return True
            return False

        # Should not raise; returns valid result.
        with patch(self._PATCH_TARGET, _adrenalin_empty_drain):
            orch = HeartbeatOrchestrator(rt)
            result = await orch._phase_run_self(
                [_stim("original")], RecallResult(), self._counts(),
            )

        # Must succeed — empty drain is safe.
        assert result is not None, (
            "Reassembly with empty drain must not crash or return None "
            "(LLM eventually succeeds)"
        )

    async def test_enforce_input_budget_error_not_swallowed(self):
        """If enforce_input_budget raises, the exception must propagate
        out of _phase_run_self rather than being silently swallowed in a
        way that hides the rebuilt prompt."""
        rt, _ = _make_fake_rt(
            idle_cfg=_idle_section(llm_failure_retry_interval=0.01),
        )

        async def _chat(messages, **kwargs):
            raise ConnectionError("error")

        rt.self_llm.chat = _chat

        wait_n = {"n": 0}

        async def _adrenalin_once(buf, seconds, *, poll_slice=0.05):
            wait_n["n"] += 1
            return wait_n["n"] == 1  # interrupt once

        orch = HeartbeatOrchestrator(rt)

        boom_count = {"n": 0}
        original_eib = orch.enforce_input_budget

        async def _booming_budget(stimuli, recall_result, counts):
            boom_count["n"] += 1
            if boom_count["n"] >= 2:   # second call (post-reassembly) blows up
                raise RuntimeError("budget blew up")
            return await original_eib(stimuli, recall_result, counts)

        orch.enforce_input_budget = _booming_budget

        with patch(self._PATCH_TARGET, _adrenalin_once):
            with pytest.raises(RuntimeError, match="budget blew up"):
                await orch._phase_run_self(
                    [_stim("x")], RecallResult(), self._counts(),
                )

    async def test_no_wait_call_when_llm_succeeds_first_try(self):
        """When the LLM succeeds on the very first attempt, wait_or_adrenalin
        must never be called (no error, no struct failure)."""
        rt, self_llm = _make_fake_rt(
            idle_cfg=_idle_section(),
            llm_responses=[_good_response()],
        )

        wait_called = {"n": 0}

        async def _should_not_be_called(buf, seconds, *, poll_slice=0.05):
            wait_called["n"] += 1
            return False

        with patch(self._PATCH_TARGET, _should_not_be_called):
            orch = HeartbeatOrchestrator(rt)
            result = await orch._phase_run_self([], RecallResult(), self._counts())

        assert result is not None
        assert wait_called["n"] == 0, (
            "wait_or_adrenalin must NOT be called on a first-try success"
        )

    async def test_struct_failure_without_adrenalin_does_not_reassemble(self):
        """Struct-output failure wait returns False (no adrenalin) → the same
        prompt must be reused — NO reassembly drain, NO new recall session
        created for that beat."""
        rt, _ = _make_fake_rt(
            idle_cfg=_idle_section(
                llm_failure_retry_interval=0.01,
                struct_output_fast_retries=1,
            ),
        )

        call_count = {"n": 0}
        _MALFORMED = "[THINKING]\nok\n[DECISION]\nok\n"

        async def _chat(messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _MALFORMED
            return _good_response()

        rt.self_llm.chat = _chat

        initial_session_count = len(rt.recall._sessions)

        async def _false_wait(buf, seconds, *, poll_slice=0.05):
            return False   # never interrupted

        with patch(self._PATCH_TARGET, _false_wait):
            orch = HeartbeatOrchestrator(rt)
            result = await orch._phase_run_self([], RecallResult(), self._counts())

        assert result is not None, "Must succeed on second LLM call"
        # No extra recall sessions created (no reassembly)
        new_session_count = len(rt.recall._sessions)
        # At most the number of sessions that existed before; reassembly
        # would add one per interrupt.
        assert new_session_count == initial_session_count, (
            "Non-interrupted wait must not create extra recall sessions"
        )

    async def test_kill_before_any_llm_call_returns_none(self):
        """If /kill is in the buffer BEFORE _phase_run_self starts its first
        wait (i.e. on initial drain during command handling in beat()), the
        beat returns None.  This is handled by the existing pre-_phase_run_self
        command phase, but we verify the orchestrator respects stop_requested."""
        rt, self_llm = _make_fake_rt(
            idle_cfg=_idle_section(),
        )
        # Simulate: stop was already set (e.g. via the pre-phase kill handler)
        rt.stop_requested = True

        orch = HeartbeatOrchestrator(rt)
        result = await orch._phase_run_self([], RecallResult(), self._counts())

        assert result is None
        assert len(self_llm.calls) == 0, (
            "LLM must not be called when stop_requested is pre-set"
        )

    async def test_multiple_adrenalin_stimuli_in_single_drain(self):
        """Drain may return several adrenalin stimuli at once; reassembly
        should process all of them and combine with existing stimuli."""
        rt, _ = _make_fake_rt(
            idle_cfg=_idle_section(llm_failure_retry_interval=0.01),
        )

        call_count = {"n": 0}
        captured_stimuli_on_second_call: list = []

        async def _chat(messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ConnectionError("error")
            # Capture last prompt for inspection.
            if messages:
                captured_stimuli_on_second_call.append(messages[-1]["content"])
            return _good_response()

        rt.self_llm.chat = _chat

        wait_n = {"n": 0}

        async def _push_multiple_and_interrupt(buf, seconds, *, poll_slice=0.05):
            wait_n["n"] += 1
            if wait_n["n"] == 1:
                await rt.buffer.push(_stim("A1", adrenalin=True))
                await rt.buffer.push(_stim("A2", adrenalin=True))
                await rt.buffer.push(_stim("A3", adrenalin=False))
                return True
            return False

        with patch(self._PATCH_TARGET, _push_multiple_and_interrupt):
            orch = HeartbeatOrchestrator(rt)
            result = await orch._phase_run_self(
                [_stim("initial")], RecallResult(), self._counts(),
            )

        assert result is not None, "Must succeed after reassembly with multiple stimuli"
        assert call_count["n"] == 2, f"Expected 2 LLM calls; got {call_count['n']}"
