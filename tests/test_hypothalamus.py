import json

import pytest

from krakey.plugins.hypothalamus.reflect import (
    HypothalamusReflectImpl as Hypothalamus,
)
from krakey.interfaces.reflect import DecisionResult


class MockLLM:
    """Captures prompt, returns a scripted JSON string."""

    def __init__(self, response):
        self.response = response
        self.last_messages = None

    async def chat(self, messages, **kwargs):
        self.last_messages = messages
        return self.response


def _tools():
    return [{"name": "action", "description": "general-purpose",
             "parameters_schema": {"intent": "string"}}]


async def test_parses_tool_call_non_urgent():
    llm = MockLLM(json.dumps({
        "tool_calls": [{"tool": "action",
                            "intent": "Search apple online",
                            "params": {},
                            "adrenalin": False}],
        "memory_writes": [],
        "memory_updates": [],
        "sleep": False,
    }))
    hypo = Hypothalamus(llm=llm)
    result = await hypo.translate("Search apple online. Not urgent.", _tools())

    assert isinstance(result, DecisionResult)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool == "action"
    assert result.tool_calls[0].adrenalin is False
    assert result.sleep is False


async def test_parses_adrenalin_true():
    llm = MockLLM(json.dumps({
        "tool_calls": [{"tool": "action", "intent": "查一下",
                            "params": {}, "adrenalin": True}],
        "memory_writes": [], "memory_updates": [], "sleep": False,
    }))
    result = await Hypothalamus(llm=llm).translate("快去查一下, 有人在等", _tools())
    assert result.tool_calls[0].adrenalin is True


async def test_parses_memory_writes():
    llm = MockLLM(json.dumps({
        "tool_calls": [],
        "memory_writes": [{"content": "用户不喜欢夸奖", "importance": "high"}],
        "memory_updates": [],
        "sleep": False,
    }))
    result = await Hypothalamus(llm=llm).translate("记住: 用户不喜欢夸奖", _tools())
    assert len(result.memory_writes) == 1
    assert result.memory_writes[0]["content"] == "用户不喜欢夸奖"


async def test_parses_memory_updates_target_completed():
    llm = MockLLM(json.dumps({
        "tool_calls": [],
        "memory_writes": [],
        "memory_updates": [{"node_name": "苹果搜索任务", "new_category": "FACT"}],
        "sleep": False,
    }))
    result = await Hypothalamus(llm=llm).translate("苹果搜索任务已完成", _tools())
    assert result.memory_updates[0]["new_category"] == "FACT"


async def test_sleep_decision():
    llm = MockLLM(json.dumps({
        "tool_calls": [], "memory_writes": [], "memory_updates": [],
        "sleep": True,
    }))
    result = await Hypothalamus(llm=llm).translate("进入睡眠", _tools())
    assert result.sleep is True


async def test_handles_trailing_comma_lenient_parse():
    """Regression: real LLMs often emit trailing commas; sanitizer must
    fix them rather than crash the heartbeat."""
    bad = ('{"tool_calls": [{"tool": "action", "intent": "x",'
           ' "params": {}, "adrenalin": false,}], "memory_writes": [],'
           ' "memory_updates": [], "sleep": false,}')
    llm = MockLLM(bad)
    result = await Hypothalamus(llm=llm).translate("anything", _tools())
    assert len(result.tool_calls) == 1


async def test_handles_smart_quotes_lenient_parse():
    """LLM emits Unicode curly quotes → must normalize to straight."""
    bad = ('{\u201ctool_calls\u201d: [], \u201cmemory_writes\u201d: [],'
           ' \u201cmemory_updates\u201d: [], \u201csleep\u201d: false}')
    llm = MockLLM(bad)
    result = await Hypothalamus(llm=llm).translate("anything", _tools())
    assert result.tool_calls == []
    assert result.sleep is False


async def test_handles_markdown_fenced_json():
    llm = MockLLM("```json\n" + json.dumps({
        "tool_calls": [], "memory_writes": [], "memory_updates": [],
        "sleep": False,
    }) + "\n```")
    result = await Hypothalamus(llm=llm).translate("No action", _tools())
    assert result.tool_calls == []


async def test_system_prompt_includes_tool_list():
    llm = MockLLM(json.dumps({
        "tool_calls": [], "memory_writes": [], "memory_updates": [],
        "sleep": False,
    }))
    await Hypothalamus(llm=llm).translate("Hi", _tools())
    # messages contains system + user; system prompt should mention tool name
    joined = json.dumps(llm.last_messages, ensure_ascii=False)
    assert "action" in joined


async def test_system_prompt_disambiguates_sleep_vs_hibernate():
    """Regression: prompt must tell LLM that 'rest/睡 N 秒/hibernate' ≠ sleep mode."""
    llm = MockLLM(json.dumps({
        "tool_calls": [], "memory_writes": [], "memory_updates": [],
        "sleep": False,
    }))
    await Hypothalamus(llm=llm).translate("anything", _tools())
    system_content = llm.last_messages[0]["content"]
    # Must mention the distinction and the dangerous near-synonyms
    assert "hibernate" in system_content.lower()
    assert "sleep" in system_content.lower()
    # Must explicitly list ambiguous phrases to exclude
    low = system_content.lower()
    assert "rest" in low or "休息" in system_content
    assert "pause" in low or "睡 n 秒" in low or "睡 n 秒" in system_content


async def test_stateless_each_call_independent():
    llm = MockLLM(json.dumps({
        "tool_calls": [], "memory_writes": [], "memory_updates": [],
        "sleep": False,
    }))
    hypo = Hypothalamus(llm=llm)
    await hypo.translate("first", _tools())
    first_messages = llm.last_messages
    await hypo.translate("second", _tools())
    second_messages = llm.last_messages
    # Independent: user message differs, no history carried
    assert first_messages != second_messages
    # Only system + one user per call (no accumulation)
    assert len(second_messages) == len(first_messages)
