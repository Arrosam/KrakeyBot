"""ToolCallParserDecisionEngine — wraps parse_tool_calls_with_failures
into a DecisionEngine impl. Tests pin Protocol conformance + that the
output shape (DecisionResult with tool_calls + parse_failures + zero
memory writes) matches the contract the heartbeat will consume."""
from __future__ import annotations

import pytest

from krakey.engines.decision._internal.tool_call_parser import (
    ToolCallParserDecisionEngine,
)
from krakey.interfaces.engines import (
    DecisionEngine,
    DecisionResult,
    ParseFailure,
    ToolCall,
)


def test_satisfies_decision_engine_protocol():
    eng = ToolCallParserDecisionEngine()
    assert isinstance(eng, DecisionEngine)


@pytest.mark.asyncio
async def test_empty_decision_yields_empty_result():
    eng = ToolCallParserDecisionEngine()
    result = await eng.translate(decision="", raw="", tools=[])
    assert isinstance(result, DecisionResult)
    assert result.tool_calls == []
    assert result.parse_failures == []
    assert result.memory_writes == []
    assert result.memory_updates == []
    assert result.sleep is False


@pytest.mark.asyncio
async def test_extracts_single_tool_call():
    decision = (
        'Going to search.\n'
        '<tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>'
    )
    eng = ToolCallParserDecisionEngine()
    result = await eng.translate(decision=decision, raw=decision, tools=[])
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert isinstance(call, ToolCall)
    assert call.tool == "search"
    assert call.params == {"q": "x"}


@pytest.mark.asyncio
async def test_parse_failure_surfaces_in_result():
    """Unparseable JSON → ParseFailure recorded; tool_calls empty."""
    decision = '<tool_call>{not json at all</tool_call>'
    eng = ToolCallParserDecisionEngine()
    result = await eng.translate(decision=decision, raw=decision, tools=[])
    assert result.tool_calls == []
    assert len(result.parse_failures) == 1
    f = result.parse_failures[0]
    assert isinstance(f, ParseFailure)
    assert f.salvaged is False


@pytest.mark.asyncio
async def test_salvage_emits_both_call_and_failure():
    """Trailing-junk JSON → call salvaged AND ParseFailure flagged
    so Self gets corrective feedback."""
    decision = (
        '<tool_call>'
        '{"name": "x", "arguments": {}}</arg_value>'
        '</tool_call>'
    )
    eng = ToolCallParserDecisionEngine()
    result = await eng.translate(decision=decision, raw=decision, tools=[])
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool == "x"
    salvaged = [f for f in result.parse_failures if f.salvaged]
    assert len(salvaged) == 1


@pytest.mark.asyncio
async def test_decision_section_only_not_raw():
    """The default impl scans decision text only — text appearing only
    in ``raw`` (e.g. quoted in [NOTE]) does NOT produce calls. Pinned
    behavior from the historical [DECISION]-only scoping decision."""
    decision = "no tags here"
    raw = (
        decision
        + "\n[NOTE] example: <tool_call>"
        + '{"name": "danger"}</tool_call>'
    )
    eng = ToolCallParserDecisionEngine()
    result = await eng.translate(decision=decision, raw=raw, tools=[])
    assert result.tool_calls == []
