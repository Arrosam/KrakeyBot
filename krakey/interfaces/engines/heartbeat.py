"""``HeartbeatEngine`` — per-beat orchestration + main run loop.

The most ambitious Engine slot. The default impl
``DefaultHeartbeatEngine`` runs the canonical 13-phase pipeline
(drain → fatigue → compact → recall → run-Self → save-round →
log-self → auto-ingest → translate-decision → dispatch → schedule-
classify → idle, plus an opening reload-self-model phase). A user
replacing this Engine controls the entire cognitive cadence — phase
order, what counts as a beat, multi-stage thinking, event-driven vs.
timer-driven execution, etc.

The Protocol intentionally exposes only two methods:

  * ``beat(runtime)`` — execute exactly one heartbeat. Useful for
    tests + future schedulers that want fine-grained control.
  * ``run(runtime, iterations)`` — drive the full loop until
    ``runtime.stop_requested`` is set. The default impl loops over ``beat``;
    a custom impl can implement entirely different scheduling.

Custom Engines can subclass ``DefaultHeartbeatEngine`` and override
specific protected ``_phase_*`` methods if they only want to tweak
one phase — that's the recommended path for incremental customization
without rewriting the whole loop.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from krakey.runtime.runtime import Runtime


@runtime_checkable
class HeartbeatEngine(Protocol):
    """The runtime's per-beat orchestrator + main loop driver."""

    async def beat(self, runtime: "Runtime") -> None:
        """Execute exactly one heartbeat. The Engine reads + mutates
        runtime state through the ``runtime`` reference. Pure logic
        with no Engine-side state — counters live on Runtime so tests
        can inspect them directly.
        """
        ...

    async def run(
        self, runtime: "Runtime", iterations: int | None = None,
    ) -> None:
        """Drive the heartbeat loop until ``runtime.stop_requested`` becomes
        True OR ``iterations`` beats have completed (whichever comes
        first). When ``iterations`` is ``None`` runs forever (or
        until stop). The runtime calls this once at startup; the
        Engine is responsible for the entire loop semantics.
        """
        ...
