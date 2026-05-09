"""``heartbeat`` Engine — per-beat orchestration + main loop driver.

Default impl ``DefaultHeartbeatEngine`` (in ``default.py``) wraps
the long-standing ``HeartbeatOrchestrator`` class for the per-beat
phase pipeline AND owns the iteration loop body that previously
lived inline in ``Runtime.run()``. Setup + teardown around the loop
(GM initialize, channels start_all, buffer stop_all, classify task
cancellation) stay on Runtime — the Engine is the cognitive
scheduler, Runtime is the lifecycle owner.

A user replacing this Engine via ``cfg.core_implementations.heartbeat``
controls the entire cognitive cadence: phase order, what counts as
a beat, multi-stage thinking, event-driven vs. timer-driven
execution. The Protocol's surface is intentionally just two methods
so the impl has wide latitude to redefine "what a heartbeat means".

The ``HeartbeatEngine`` Protocol the runtime depends on lives at
``krakey.interfaces.engines.heartbeat``.
"""
from krakey.engines.heartbeat.default import DefaultHeartbeatEngine

__all__ = ["DefaultHeartbeatEngine"]
