"""HypothalamusDecisionEngine — Protocol conformance + LLM-driven
translation behavior. Lifted from the retired hypothalamus plugin
into a DecisionEngine impl.

Mock the LLM via factory injection — we don't need a real model to
verify the prompt construction, JSON parsing, and DecisionResult
shape."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from krakey.engines.decision.hypothalamus import (
    HypothalamusDecisionEngine,
)
from krakey.interfaces.engines import DecisionEngine, DecisionResult
from krakey.models.config import (
    Config,
    LLMSection,
    Provider,
    TagBinding,
)


def _make_cfg() -> Config:
    return Config(llm=LLMSection(
        providers={"P": Provider(
            type="openai_compatible",
            base_url="http://x", api_key="k",
        )},
        tags={"hypo": TagBinding(provider="P/hypo-model")},
        core_purposes={"hypothalamus": "hypo"},
    ))


class _FakeChatClient:
    """Stand-in LLM that returns canned responses recorded per call."""

    def __init__(self, response: str):
        self.response = response
        self.calls: list[list] = []
        self.model = "hypo-model"

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return self.response


def test_satisfies_decision_engine_protocol():
    eng = HypothalamusDecisionEngine(cfg=_make_cfg())
    assert isinstance(eng, DecisionEngine)


def test_requires_cfg_or_factory():
    """Either ``cfg`` or ``factory`` must be supplied — pure no-arg
    construction can't reach an LLM client."""
    with pytest.raises(TypeError, match="factory|cfg"):
        HypothalamusDecisionEngine()


@pytest.mark.asyncio
async def test_translate_parses_llm_json_response():
    """Given a clean JSON response, translate() returns a populated
    DecisionResult."""
    eng = HypothalamusDecisionEngine(cfg=_make_cfg())
    response = json.dumps({
        "tool_calls": [
            {"tool": "search", "intent": "search the web",
             "params": {"q": "krakey"}, "adrenalin": False},
        ],
        "memory_writes": [
            {"content": "the project is called krakey",
             "importance": "normal"},
        ],
        "memory_updates": [],
        "sleep": False,
    })
    fake = _FakeChatClient(response)
    # Bypass the real factory by monkey-patching.
    eng._factory = type("F", (), {
        "client_for_core_purpose": lambda self, purpose: fake,
    })()
    result = await eng.translate(
        decision="search for krakey",
        raw="search for krakey",
        tools=[{
            "name": "search", "description": "search the web",
            "parameters_schema": {"q": "string"},
        }],
    )
    assert isinstance(result, DecisionResult)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool == "search"
    assert result.tool_calls[0].params == {"q": "krakey"}
    assert len(result.memory_writes) == 1
    assert result.sleep is False
    assert result.parse_failures == []  # Engine doesn't surface them


@pytest.mark.asyncio
async def test_translate_raises_when_no_translator_tag_bound():
    """No core_purpose for hypothalamus → translate raises clearly."""
    cfg = Config(llm=LLMSection(
        providers={}, tags={}, core_purposes={},  # empty
    ))
    eng = HypothalamusDecisionEngine(cfg=cfg)
    with pytest.raises(RuntimeError, match="hypothalamus"):
        await eng.translate(decision="x", raw="x", tools=[])


@pytest.mark.asyncio
async def test_translate_raises_on_empty_llm_response():
    eng = HypothalamusDecisionEngine(cfg=_make_cfg())
    eng._factory = type("F", (), {
        "client_for_core_purpose":
            lambda self, purpose: _FakeChatClient(""),
    })()
    with pytest.raises(ValueError, match="empty content"):
        await eng.translate(decision="x", raw="x", tools=[])


@pytest.mark.asyncio
async def test_translate_handles_markdown_fenced_json():
    """LLMs sometimes wrap JSON in ```json ... ``` fences — parser
    must strip them before parsing."""
    response = (
        "```json\n"
        + json.dumps({"tool_calls": [], "memory_writes": [],
                      "memory_updates": [], "sleep": False})
        + "\n```"
    )
    eng = HypothalamusDecisionEngine(cfg=_make_cfg())
    eng._factory = type("F", (), {
        "client_for_core_purpose":
            lambda self, purpose: _FakeChatClient(response),
    })()
    result = await eng.translate(decision="x", raw="x", tools=[])
    assert result.tool_calls == []
    assert result.sleep is False
