"""``DefaultHeartbeatEngine`` — default HeartbeatEngine impl.

Lazy-built wrapper around the long-standing ``HeartbeatOrchestrator``
class. The Engine's ``beat()`` forwards to ``orchestrator.beat()``;
``run()`` is the iteration loop that previously lived inline in
``Runtime.run()``.

The orchestrator's per-phase methods (the 13 ``_phase_*`` private
methods + ``build_self_prompt`` / ``enforce_input_budget`` /
``_perform_sleep``) stay on the orchestrator class — wrapping rather
than rewriting. Custom HeartbeatEngine impls that want to tweak a
single phase can subclass DefaultHeartbeatEngine, build their own
orchestrator subclass, and override ``_make_orchestrator()``.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from krakey.engines.heartbeat.orchestrator import (
        HeartbeatOrchestrator,
    )
    from krakey.runtime.runtime import Runtime


class DefaultHeartbeatEngine:
    """Default HeartbeatEngine — drives the canonical 13-phase
    pipeline via HeartbeatOrchestrator, owns the iteration loop."""

    def __init__(self, *, cfg=None):
        del cfg  # accepted for contract uniformity, unused
        self._orchestrator: "HeartbeatOrchestrator | None" = None

    def _make_orchestrator(
        self, runtime: "Runtime",
    ) -> "HeartbeatOrchestrator":
        """Subclass hook — override to swap in an alternative
        orchestrator (e.g. one with different phase ordering). The
        default implementation builds the canonical class."""
        from krakey.engines.heartbeat.orchestrator import (
            HeartbeatOrchestrator,
        )
        return HeartbeatOrchestrator(runtime)

    def _ensure_orchestrator(
        self, runtime: "Runtime",
    ) -> "HeartbeatOrchestrator":
        if self._orchestrator is not None:
            return self._orchestrator
        # Reuse runtime's pre-built orchestrator if Runtime__init__
        # already constructed one (it does today as ``self._orchestrator``
        # for legacy facade methods like ``runtime._build_self_prompt``).
        # That keeps facade callers + the Engine pointing at the same
        # instance, so tests that reach in directly + production
        # heartbeat both observe consistent state.
        existing = getattr(runtime, "_orchestrator", None)
        if existing is not None:
            self._orchestrator = existing
        else:
            self._orchestrator = self._make_orchestrator(runtime)
        return self._orchestrator

    async def beat(self, runtime: "Runtime") -> None:
        """One heartbeat — delegates to the orchestrator."""
        await self._ensure_orchestrator(runtime).beat()

    async def run(
        self, runtime: "Runtime", iterations: int | None = None,
    ) -> None:
        """Drive the heartbeat loop until ``runtime.stop_requested``
        becomes True OR ``iterations`` beats have completed.

        Setup (gm.initialize, channels.start_all, etc.) + teardown
        (buffer.stop_all, classify-task cancellation) stay on
        ``Runtime.run()`` — the Engine starts after setup and
        returns before teardown.
        """
        count = 0
        while not runtime.stop_requested:
            runtime.poll_pause_file()
            if runtime.paused:
                await asyncio.sleep(0.25)
                continue
            await self.beat(runtime)
            count += 1
            if iterations is not None and count >= iterations:
                return
