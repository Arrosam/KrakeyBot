"""Behavioral tests for the pause-aware run() loop in DefaultHeartbeatEngine.

These tests verify the NEW behavior:
  - Before each beat, the engine calls runtime.poll_pause_file() then
    checks runtime.paused.
  - When paused=True: skip beat, await asyncio.sleep(0.25), continue
    WITHOUT incrementing the beat count.
  - When paused=False: run beat() and increment count as before.
  - The iterations cap is checked AFTER beat(), so paused ticks do NOT
    count toward the iterations limit.
  - stop_requested remains responsive while paused.

Harness follows the existing test_heartbeat_engine.py style exactly:
  - _FakeOrchestrator counts beat() calls via runtime._orchestrator
  - _FakeRuntime extends the existing _FakeRuntime shape, adding
    `paused` and `poll_pause_file()`
  - asyncio.sleep is patched to an async no-op (or driven stub) so
    the 0.25s paused-tick sleeps do not slow the suite.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from krakey.engines.heartbeat.default import DefaultHeartbeatEngine


# ---------------------------------------------------------------------------
# Shared fake collaborators
# ---------------------------------------------------------------------------

class _FakeOrchestrator:
    """Stand-in HeartbeatOrchestrator — counts beat() calls."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.beat_count = 0

    async def beat(self):
        self.beat_count += 1


class _FakeRuntime:
    """Bare runtime for pause-aware run() tests.

    Exposes:
      stop_requested  — loop-exit predicate (bool property)
      request_stop()  — cooperative shutdown
      paused          — bool; whether the engine should skip a beat
      poll_pause_file() — called by the engine each iteration before
                          reading paused; tests can install side effects
                          here to drive dynamic scenarios.
    """

    def __init__(self):
        self._stop = False
        self.paused = False
        self._poll_calls = 0
        self._orchestrator = _FakeOrchestrator(self)
        # Optional hook: if set, called every poll_pause_file() invocation.
        # Signature: (runtime, poll_call_number) -> None
        self._poll_side_effect = None

    @property
    def stop_requested(self) -> bool:
        return self._stop

    def request_stop(self) -> None:
        self._stop = True

    def poll_pause_file(self) -> None:
        self._poll_calls += 1
        if self._poll_side_effect is not None:
            self._poll_side_effect(self, self._poll_calls)


# ---------------------------------------------------------------------------
# Test 1 — Not paused: iterations=3, beat called 3 times, poll called >= 3
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_not_paused_beat_count_and_poll_count():
    """run(iterations=3) with paused=False calls beat exactly 3 times
    and poll_pause_file at least 3 times."""
    eng = DefaultHeartbeatEngine()
    rt = _FakeRuntime()
    rt.paused = False

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await eng.run(rt, iterations=3)

    assert rt._orchestrator.beat_count == 3
    assert rt._poll_calls >= 3
    # The pause-path sleep must NOT have been called (never paused).
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2 — Paused throughout: beat never called, exits on stop_requested
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_paused_throughout_beat_never_called_exits_on_stop():
    """With paused=True for every iteration, beat() is never called.
    The loop must still exit when stop_requested flips True.
    Uses a poll_pause_file side effect to flip stop after K poll calls."""
    eng = DefaultHeartbeatEngine()
    rt = _FakeRuntime()
    rt.paused = True

    sleep_call_count = 0

    async def fast_sleep(seconds):
        nonlocal sleep_call_count
        sleep_call_count += 1

    # Flip stop after 3 paused ticks so the loop exits.
    def stop_after_3(runtime, n):
        if n >= 3:
            runtime.request_stop()

    rt._poll_side_effect = stop_after_3

    with patch("asyncio.sleep", side_effect=fast_sleep):
        await eng.run(rt, iterations=None)

    assert rt._orchestrator.beat_count == 0
    # The paused path must have awaited sleep at least once.
    assert sleep_call_count >= 1


# ---------------------------------------------------------------------------
# Test 3 — iterations not consumed by paused ticks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_iterations_not_consumed_by_paused_ticks():
    """Paused ticks do not count toward the iterations limit.

    Scenario: runtime starts paused for the first 2 poll cycles, then
    becomes unpaused.  run(iterations=2) must call beat exactly 2 times
    and return (not loop forever).
    """
    eng = DefaultHeartbeatEngine()
    rt = _FakeRuntime()
    rt.paused = True  # start paused

    def unpause_after_2(runtime, n):
        if n > 2:
            runtime.paused = False

    rt._poll_side_effect = unpause_after_2

    sleep_call_count = 0

    async def fast_sleep(seconds):
        nonlocal sleep_call_count
        sleep_call_count += 1

    with patch("asyncio.sleep", side_effect=fast_sleep):
        await eng.run(rt, iterations=2)

    # Exactly 2 real beats must have occurred.
    assert rt._orchestrator.beat_count == 2
    # At least 2 paused-tick sleeps occurred before unpausing.
    assert sleep_call_count >= 2
    # Loop must have returned (we reach this line = it didn't hang).


# ---------------------------------------------------------------------------
# Test 4 — poll ordering: poll_pause_file called before paused check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_poll_called_before_paused_check_same_iteration():
    """poll_pause_file() must update the paused state that is read in
    the same iteration — i.e. it is invoked first.

    Setup: paused starts True; the first poll_pause_file() call sets
    paused=False.  If poll is called before the paused check, the very
    first iteration should run beat() (not skip it).
    Assert: beat_count == 1 after run(iterations=1).
    """
    eng = DefaultHeartbeatEngine()
    rt = _FakeRuntime()
    rt.paused = True  # starts paused

    def first_poll_unpauses(runtime, n):
        # On the very first poll call, clear the paused flag.
        if n == 1:
            runtime.paused = False

    rt._poll_side_effect = first_poll_unpauses

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await eng.run(rt, iterations=1)

    # If poll happens before the paused check, paused was False when
    # the engine read it, so beat ran.
    assert rt._orchestrator.beat_count == 1


# ---------------------------------------------------------------------------
# Test 5 — stop_requested short-circuits even when paused
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_requested_at_entry_while_paused_returns_immediately():
    """If stop_requested is already True when run() is entered AND
    paused is True, run() must return immediately with zero beat calls
    and minimal (ideally zero) poll calls."""
    eng = DefaultHeartbeatEngine()
    rt = _FakeRuntime()
    rt.paused = True
    rt.request_stop()  # stop is set before run() is called

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await eng.run(rt, iterations=None)

    assert rt._orchestrator.beat_count == 0
    # The while condition fires before any work — loop body must not execute.
    mock_sleep.assert_not_called()
    # poll_pause_file must not have been called (loop body never entered).
    assert rt._poll_calls == 0
