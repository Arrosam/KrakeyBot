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

from krakey.interfaces.engines.decision import DecisionResult
from krakey.runtime.heartbeat.action_executor import (
    parse_tool_calls_with_failures,
)


class ToolCallParserDecisionEngine:
    """Default DecisionEngine — scripts tool_call tag parsing.

    Stateless. Accepts ``cfg`` + ``factory`` kwargs for signature
    uniformity with other DecisionEngine impls
    (HypothalamusDecisionEngine needs them); the script parser
    ignores both.
    """

    def __init__(self, *, cfg=None, factory=None):
        del cfg, factory

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

        tool_calls, parse_failures = parse_tool_calls_with_failures(
            decision,
        )
        return DecisionResult(
            tool_calls=tool_calls,
            memory_writes=[],
            memory_updates=[],
            sleep=False,
            parse_failures=parse_failures,
        )
