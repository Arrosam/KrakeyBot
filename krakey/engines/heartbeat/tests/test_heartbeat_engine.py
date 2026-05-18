"""DefaultHeartbeatEngine — Protocol conformance + orchestrator
delegation + iteration loop.

The Engine wraps HeartbeatOrchestrator; tests use a fake runtime
(no real heartbeat work) to exercise the run-loop semantics without
booting an end-to-end runtime."""
from __future__ import annotations

import pytest

from krakey.engines.heartbeat.default import DefaultHeartbeatEngine
from krakey.interfaces.engines import HeartbeatEngine


class _FakeOrchestrator:
    """Stand-in HeartbeatOrchestrator — counts beat() calls."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.beat_count = 0

    async def beat(self):
        self.beat_count += 1


class _FakeRuntime:
    """Bare runtime that exposes ``stop_requested`` (loop exit
    predicate the Engine reads per iteration) + ``request_stop()``
    (cooperative shutdown signal tests flip to end the loop).
    Also a no-op ``poll_pause_file()`` / ``paused`` pair the loop
    consults each iteration (never paused in these tests)."""

    def __init__(self, stop_after: int | None = None):
        self._stop = False
        self._stop_after = stop_after
        self._beat_observed = 0
        self._orchestrator = _FakeOrchestrator(self)

    @property
    def stop_requested(self) -> bool:
        return self._stop

    def request_stop(self) -> None:
        self._stop = True

    def poll_pause_file(self) -> None:
        pass

    @property
    def paused(self) -> bool:
        return False


def test_satisfies_heartbeat_engine_protocol():
    eng = DefaultHeartbeatEngine()
    assert isinstance(eng, HeartbeatEngine)


@pytest.mark.asyncio
async def test_beat_delegates_to_orchestrator():
    eng = DefaultHeartbeatEngine()
    rt = _FakeRuntime()
    await eng.beat(rt)
    assert rt._orchestrator.beat_count == 1


@pytest.mark.asyncio
async def test_beat_reuses_runtime_orchestrator():
    """Engine's _ensure_orchestrator picks up runtime._orchestrator
    on first call so the Engine + facade methods both observe the
    same instance."""
    eng = DefaultHeartbeatEngine()
    rt = _FakeRuntime()
    await eng.beat(rt)
    assert eng._orchestrator is rt._orchestrator


@pytest.mark.asyncio
async def test_run_iterates_until_iterations_reached():
    """run(iterations=3) should call beat() exactly 3 times."""
    eng = DefaultHeartbeatEngine()
    rt = _FakeRuntime()
    await eng.run(rt, iterations=3)
    assert rt._orchestrator.beat_count == 3


@pytest.mark.asyncio
async def test_run_stops_when_runtime_stop_set():
    """If a beat calls runtime.request_stop(), the loop should exit even with
    iterations remaining."""
    eng = DefaultHeartbeatEngine()
    rt = _FakeRuntime()

    # Patch the orchestrator to flip _stop on the 2nd beat.
    original_beat = rt._orchestrator.beat

    async def maybe_stop():
        await original_beat()
        if rt._orchestrator.beat_count == 2:
            rt.request_stop()

    rt._orchestrator.beat = maybe_stop
    await eng.run(rt, iterations=10)
    assert rt._orchestrator.beat_count == 2  # not 10


@pytest.mark.asyncio
async def test_run_with_no_iteration_cap_runs_until_stop():
    """iterations=None means "run forever" — exit only via
    runtime.stop_requested."""
    eng = DefaultHeartbeatEngine()
    rt = _FakeRuntime()

    original_beat = rt._orchestrator.beat

    async def stop_at_5():
        await original_beat()
        if rt._orchestrator.beat_count == 5:
            rt.request_stop()

    rt._orchestrator.beat = stop_at_5
    await eng.run(rt, iterations=None)
    assert rt._orchestrator.beat_count == 5
