"""``heartbeat`` Engine — per-beat orchestration + main loop driver.

Default impl ``DefaultHeartbeatEngine`` wraps the canonical
``HeartbeatOrchestrator`` (the 13-phase pipeline) AND owns the
iteration loop body. Setup + teardown around the loop (GM initialize,
channels start_all, buffer stop_all, classify task cancellation) stay
on Runtime — the Engine is the cognitive scheduler, Runtime is the
lifecycle owner.

A user replacing this Engine controls the entire cognitive cadence:
phase order, what counts as a beat, multi-stage thinking, event-driven
vs. timer-driven execution. The Protocol's surface is intentionally
just two methods so the impl has wide latitude to redefine "what a
heartbeat means".

The ``HeartbeatEngine`` Protocol the runtime depends on lives at
``krakey.interfaces.engines.heartbeat``.
"""
from krakey.engines.catalog import EngineImpl
from krakey.engines.heartbeat.default import DefaultHeartbeatEngine

BUILTIN_ENGINES = {
    "phased": EngineImpl(
        cls=DefaultHeartbeatEngine,
        description=(
            "Canonical 13-phase pipeline: drain → fatigue → compact "
            "→ recall → run-Self → save-round → log → auto-ingest → "
            "translate-decision → dispatch → schedule-classify → idle."
        ),
    ),
}

DEFAULT_ENGINE = "phased"

__all__ = ["BUILTIN_ENGINES", "DEFAULT_ENGINE", "DefaultHeartbeatEngine"]
