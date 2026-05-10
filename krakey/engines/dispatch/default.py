"""``LocalDispatchEngine`` — default DispatchEngine impl.

Thin wrapper around the existing ``DecisionDispatcher`` class. The
Engine builds one DecisionDispatcher lazily on first ``dispatch()``
call (so the dispatcher's collaborator references match the
runtime's current state) then reuses it across beats.

The wrapped dispatcher is built per-Engine, NOT per-beat — its
collaborators (tools registry, batch tracker, buffer, memory engine,
log, event bus) are all long-lived runtime resources that don't
change across beats during a normal session. The lazy construction
just defers building until we have a runtime ref to read from.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from krakey.interfaces.engines.decision import DecisionResult
    from krakey.engines.dispatch.dispatcher import DecisionDispatcher
    from krakey.runtime.runtime import Runtime


class LocalDispatchEngine:
    """Default DispatchEngine — runs every side-effect in-process via
    the long-standing DecisionDispatcher class."""

    def __init__(self, *, cfg=None):
        del cfg  # accepted for contract uniformity, unused
        self._dispatcher: "DecisionDispatcher | None" = None

    def _ensure_dispatcher(
        self, runtime: "Runtime",
    ) -> "DecisionDispatcher":
        """Lazy build — runtime collaborators are stable across the
        runtime's lifetime, so once built the dispatcher is reused."""
        if self._dispatcher is None:
            from krakey.engines.dispatch.dispatcher import (
                DecisionDispatcher,
            )
            self._dispatcher = DecisionDispatcher(
                tools=runtime.tools,
                batch_tracker=runtime.batch_tracker,
                buffer=runtime.buffer,
                gm=runtime.memory,
                log=runtime.log,
                events=runtime.events,
            )
        return self._dispatcher

    async def dispatch(
        self,
        heartbeat_id: int,
        decision_result: "DecisionResult",
        runtime: "Runtime",
        *,
        recall_context: list[dict] | None = None,
    ) -> None:
        d = self._ensure_dispatcher(runtime)
        d.log_summary(heartbeat_id, decision_result)
        await d.dispatch_tool_calls(
            heartbeat_id, decision_result.tool_calls,
        )
        await d.apply_memory_writes(
            decision_result.memory_writes,
            recall_context or [],
            heartbeat_id,
        )
        await d.apply_memory_updates(decision_result.memory_updates)
