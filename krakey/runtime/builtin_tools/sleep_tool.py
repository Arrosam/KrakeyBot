"""``SleepTool`` — built-in reserved tool name that lets Self choose
voluntary sleep without depending on the hypothalamus translator
plugin.

Why built-in: the previous design made ``DecisionResult.sleep=True``
reachable only through ``hypothalamus.translate(...)``'s JSON
output. Disable hypothalamus → Self could no longer choose sleep
under any circumstance. That violates "core capabilities aren't
plugin-gated" — sleep is part of the heartbeat lifecycle, not an
add-on.

Why a Tool: it surfaces in ``[CAPABILITIES]`` through the same
mechanism every other tool uses, so Self learns about sleep
through the prompt rather than a separate teaching layer. The
contract appears identical to other tools from Self's POV:

    <tool_call>{"name": "sleep"}</tool_call>

Why intercepted (not really executed): the heartbeat loop's
sleep transition needs to happen IN the loop AFTER
``_phase_apply_decision`` returns — not as an async fire-and-
forget task fed by ``_dispatcher.dispatch_tool_calls``. So the
orchestrator filters tool calls with ``tool == SLEEP_TOOL_NAME``
out of the dispatch list and sets ``result.sleep = True`` instead.
``SleepTool.execute`` is registered for [CAPABILITIES] visibility
but should never actually run. If it does (orchestrator bug),
return a diagnostic stimulus so the failure is observable rather
than silent.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from krakey.interfaces.tool import Tool
from krakey.models.stimulus import Stimulus


SLEEP_TOOL_NAME = "sleep"


class SleepTool(Tool):
    """Reserved built-in tool. Dispatch is intercepted by the
    heartbeat orchestrator; ``execute`` should never be called."""

    @property
    def name(self) -> str:
        return SLEEP_TOOL_NAME

    @property
    def description(self) -> str:
        return (
            "Voluntarily enter sleep mode at the end of this beat — "
            "runs the full 7-phase Sleep cycle (clustering, GM→KB "
            "migration, KB consolidation/archival, FOCUS clearing, "
            "index rebuild). Use when fatigue is high or you want "
            "to consolidate recent learning. No arguments. "
            "If energy is currently high (fatigue below the minimum "
            "configured fatigue threshold), the request is refused and "
            "you will receive a notice — sleep only takes effect once "
            "fatigue has built up sufficiently."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, intent: str, params: dict[str, Any]) -> Stimulus:
        # The heartbeat orchestrator intercepts SLEEP_TOOL_NAME calls
        # before dispatch and routes them via runtime.trigger_memory_sleep.
        # If execute is ever reached (e.g. a code path that bypasses the
        # intercept), return a benign acknowledgement rather than raising —
        # the real sleep path is runtime.trigger_memory_sleep.
        return Stimulus(
            type="system_event",
            source=f"tool:{SLEEP_TOOL_NAME}",
            content="Sleep requested.",
            timestamp=datetime.now(),
        )
