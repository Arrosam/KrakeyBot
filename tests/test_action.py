import pytest

from src.models.stimulus import Stimulus
from src.tentacles.action import ActionTentacle


class MockLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list = []

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return self._responses.pop(0)


async def test_execute_returns_stimulus_with_llm_result():
    llm = MockLLM(["I found 3 pages about hello world."])
    tentacle = ActionTentacle(llm=llm, max_context_tokens=4096)

    stim = await tentacle.execute("search hello world", {})

    assert isinstance(stim, Stimulus)
    assert stim.source == "tentacle:action"
    assert stim.type == "tentacle_feedback"
    assert "hello world" in stim.content


async def test_properties():
    tentacle = ActionTentacle(llm=MockLLM([""]))
    assert tentacle.name == "action"
    assert tentacle.description
    assert isinstance(tentacle.parameters_schema, dict)
    assert tentacle.sandboxed is True


async def test_context_inflation_triggers_summary_and_reset():
    llm = MockLLM(["summary: did lots of stuff"])
    tentacle = ActionTentacle(llm=llm, max_context_tokens=5)  # tiny budget

    stim = await tentacle.execute("do some very long piece of work here", {})
    assert "summary" in stim.content.lower()
    # context must reset after summary
    assert tentacle.context == []


async def test_normal_execute_appends_to_context():
    llm = MockLLM(["replyA"])
    tentacle = ActionTentacle(llm=llm, max_context_tokens=4096)
    await tentacle.execute("hi", {})
    assert tentacle.context[-1]["role"] == "assistant"
    assert tentacle.context[-1]["content"] == "replyA"
