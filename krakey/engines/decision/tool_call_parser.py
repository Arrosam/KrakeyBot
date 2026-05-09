"""``ToolCallParserDecisionEngine`` — scripted ``<tool_call>`` parser.

Default DecisionEngine impl. Wraps the existing
``parse_tool_calls_with_failures`` util (in
``krakey.runtime.heartbeat.action_executor``) so the same per-block
salvage + ParseFailure surfacing the heartbeat already relies on
keeps working — only the call-site pattern changes (Engine slot
instead of an inline module-fn import).

Scope: turns parsed-decision text into a ``DecisionResult`` containing
``tool_calls`` + ``parse_failures``. ``memory_writes`` / ``memory_updates``
/ ``sleep`` are always empty in this impl — they're the responsibility
of an LLM-based translator like ``HypothalamusDecisionEngine`` that can
extract those signals from Self's free-form text. The default Engine
parses tags only; richer extraction is opt-in.

Note on input scoping (preserved from the original orchestrator
behavior, see commit bb752c8): the parser scans ONLY the
``decision_text`` (i.e. the ``[DECISION]`` section), NOT the full raw
response. Self's ``[NOTE]`` and ``[THINKING]`` sections sometimes
contain quoted format examples for self-correction; scanning the full
raw response would re-parse those examples as real tool calls and
re-fail on the same drift signature. The Engine receives ``raw`` as a
parameter so future impls can scan it if they want, but the default
deliberately doesn't.
"""
from __future__ import annotations

from typing import Any

from krakey.interfaces.engines.decision import (
    DecisionResult,
    ParseFailure,
    ToolCall,
)
from krakey.runtime.heartbeat.action_executor import (
    parse_tool_calls_with_failures,
)


class ToolCallParserDecisionEngine:
    """Default DecisionEngine — scripts tool_call tag parsing.

    Stateless. Construction takes no kwargs (the EngineRegistry will
    call it with no args via ``cls()``).
    """

    async def translate(
        self,
        decision: str,
        raw: str,
        tools: list[dict[str, Any]],
    ) -> DecisionResult:
        # ``tools`` is unused by the script parser (it doesn't
        # validate names against the registry — that's the
        # dispatcher's job, which produces an "Unknown tool" stimulus
        # on a name miss). Argument kept for Protocol compatibility
        # so swap-in alternative impls (e.g. an LLM translator) can
        # use the live tool list without a separate Protocol.
        del raw, tools

        legacy_calls, legacy_failures = parse_tool_calls_with_failures(
            decision,
        )

        # parse_tool_calls_with_failures still produces the legacy
        # (modifier.py) ToolCall + ParseFailure dataclasses. Convert
        # to the new (engines/decision.py) shapes so the heartbeat
        # consumes a single canonical type. Step 14 unifies the
        # legacy types via re-export, eliminating this conversion.
        tool_calls = [
            ToolCall(
                tool=c.tool,
                intent=c.intent,
                params=dict(c.params or {}),
                adrenalin=c.adrenalin,
            )
            for c in legacy_calls
        ]
        parse_failures = [
            ParseFailure(
                payload=f.payload,
                error=f.error,
                block_index=f.block_index,
                salvaged=f.salvaged,
            )
            for f in legacy_failures
        ]
        return DecisionResult(
            tool_calls=tool_calls,
            memory_writes=[],
            memory_updates=[],
            sleep=False,
            parse_failures=parse_failures,
        )
