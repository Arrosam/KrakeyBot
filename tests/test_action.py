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
    # context resets to fresh state — system prompt only, no user/assistant
    assert len(tentacle.context) == 1
    assert tentacle.context[0]["role"] == "system"


async def test_normal_execute_appends_to_context():
    llm = MockLLM(["replyA"])
    tentacle = ActionTentacle(llm=llm, max_context_tokens=4096)
    await tentacle.execute("hi", {})
    # context: [system, user, assistant]
    assert tentacle.context[0]["role"] == "system"
    assert tentacle.context[-1]["role"] == "assistant"
    assert tentacle.context[-1]["content"] == "replyA"


async def test_system_prompt_injected_at_start():
    """Regression: without a system prompt, the LLM behaves like a generic
    chatbot and refuses 'I cannot operate your local system'."""
    from src.tentacles.action import ACTION_SYSTEM_PROMPT
    llm = MockLLM(["ok"])
    tentacle = ActionTentacle(llm=llm)
    await tentacle.execute("greet user", {})
    msgs = llm.calls[0]
    assert msgs[0]["role"] == "system"
    assert "Krakey" in msgs[0]["content"]
    assert "不要拒绝" in msgs[0]["content"] or "do not refuse" in msgs[0]["content"].lower()
    assert msgs[0]["content"] == ACTION_SYSTEM_PROMPT


async def test_system_prompt_re_added_after_summary_reset():
    """Token-bloat path resets context — must keep system prompt in the
    fresh context so subsequent calls stay framed."""
    llm = MockLLM(["summary text", "next reply"])
    tentacle = ActionTentacle(llm=llm, max_context_tokens=5)  # tiny

    # Force the summary path
    await tentacle.execute("a long enough intent to overflow tokens", {})
    # Context was reset; should still have system at index 0
    assert tentacle.context[0]["role"] == "system"

    await tentacle.execute("hi", {})
    # Last call's messages must again start with system
    assert llm.calls[-1][0]["role"] == "system"
