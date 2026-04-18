"""Integration test for Phase-0 main loop with fully mocked LLMs."""
import asyncio
import json
from datetime import datetime

import pytest

from src.main import Runtime, RuntimeDeps, build_runtime_with_fakes
from src.models.stimulus import Stimulus
from src.runtime.stimulus_buffer import StimulusBuffer


class ScriptedLLM:
    """Returns responses from a queue, records prompts."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list = []

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        if not self._responses:
            return ""
        return self._responses.pop(0)


async def test_single_iteration_user_message_triggers_action_tentacle():
    # Self: replies with intent to use action tentacle
    self_llm = ScriptedLLM([
        "[THINKING]\nuser said hello. reply.\n"
        "[DECISION]\nUse action tentacle to greet the user.\n"
        "[HIBERNATE]\n1"
    ])
    # Hypothalamus: one action call, non-urgent
    hypo_llm = ScriptedLLM([json.dumps({
        "tentacle_calls": [{"tentacle": "action", "intent": "Greet user",
                            "params": {}, "adrenalin": False}],
        "memory_writes": [], "memory_updates": [], "sleep": False,
    })])
    # Action tentacle: canned reply
    action_llm = ScriptedLLM(["Hi there!"])

    runtime = build_runtime_with_fakes(self_llm=self_llm, hypo_llm=hypo_llm,
                                        action_llm=action_llm)

    # Seed a user stimulus before starting
    await runtime.buffer.push(Stimulus(
        type="user_message", source="sensory:cli_input",
        content="hello", timestamp=datetime.now(), adrenalin=True,
    ))

    # Run exactly one heartbeat
    await runtime.run(iterations=1)

    # Action must have been called with greeting intent
    assert len(action_llm.calls) == 1
    # Wait for tentacle task to complete and push feedback
    await asyncio.sleep(0.05)
    remaining = runtime.buffer.drain()
    contents = [s.content for s in remaining]
    assert any("Hi there!" in c for c in contents)


async def test_no_action_decision_runs_no_tentacle():
    self_llm = ScriptedLLM([
        "[DECISION]\nNo action.\n[HIBERNATE]\n1"
    ])
    hypo_llm = ScriptedLLM([json.dumps({
        "tentacle_calls": [], "memory_writes": [], "memory_updates": [],
        "sleep": False,
    })])
    action_llm = ScriptedLLM(["should not be called"])

    runtime = build_runtime_with_fakes(self_llm=self_llm, hypo_llm=hypo_llm,
                                        action_llm=action_llm)
    await runtime.run(iterations=1)
    assert action_llm.calls == []


async def test_adrenalin_inheritance_from_hypothalamus():
    self_llm = ScriptedLLM([
        "[DECISION]\nAct fast, user waiting.\n[HIBERNATE]\n1"
    ])
    hypo_llm = ScriptedLLM([json.dumps({
        "tentacle_calls": [{"tentacle": "action", "intent": "go",
                            "params": {}, "adrenalin": True}],
        "memory_writes": [], "memory_updates": [], "sleep": False,
    })])
    action_llm = ScriptedLLM(["done"])

    runtime = build_runtime_with_fakes(self_llm=self_llm, hypo_llm=hypo_llm,
                                        action_llm=action_llm)
    await runtime.run(iterations=1)

    # Wait for dispatch
    await asyncio.sleep(0.05)
    stims = runtime.buffer.drain()
    tentacle_stims = [s for s in stims if s.type == "tentacle_feedback"]
    assert tentacle_stims
    assert tentacle_stims[0].adrenalin is True  # inherited


async def test_tentacle_feedback_auto_ingested_to_gm():
    """Phase 1: tentacle_feedback stimuli seen on next heartbeat get
    auto_ingested into Graph Memory."""
    self_llm = ScriptedLLM([
        # HB #1: user says hi → dispatch action
        "[DECISION]\nUse action to greet.\n[HIBERNATE]\n1",
        # HB #2: see action feedback → no more work
        "[DECISION]\nNo action.\n[HIBERNATE]\n1",
    ])
    hypo_llm = ScriptedLLM([
        json.dumps({"tentacle_calls": [{"tentacle": "action",
                                         "intent": "greet",
                                         "params": {}, "adrenalin": False}],
                     "memory_writes": [], "memory_updates": [],
                     "sleep": False}),
        json.dumps({"tentacle_calls": [], "memory_writes": [],
                     "memory_updates": [], "sleep": False}),
    ])
    action_llm = ScriptedLLM(["Hello! Nice to meet you."])

    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=hypo_llm, action_llm=action_llm,
    )

    await runtime.buffer.push(Stimulus(
        type="user_message", source="sensory:cli_input",
        content="hi", timestamp=datetime.now(), adrenalin=True,
    ))
    await runtime.run(iterations=2)
    # Graph Memory must now contain at least one node originating from the
    # action tentacle's feedback.
    nodes = await runtime.gm.list_nodes()
    contents = [n["description"] for n in nodes]
    assert any("nice to meet" in c.lower() for c in contents), contents


async def test_batch_complete_stimulus_wakes_next_heartbeat():
    """After dispatch, BatchTracker fires a batch_complete adrenalin
    stimulus that Self sees on the subsequent heartbeat."""
    self_llm = ScriptedLLM([
        "[DECISION]\nUse action.\n[HIBERNATE]\n60",  # long interval
        "[DECISION]\nNo action.\n[HIBERNATE]\n1",
    ])
    hypo_llm = ScriptedLLM([
        json.dumps({"tentacle_calls": [{"tentacle": "action", "intent": "x",
                                         "params": {}, "adrenalin": False}],
                     "memory_writes": [], "memory_updates": [],
                     "sleep": False}),
        json.dumps({"tentacle_calls": [], "memory_writes": [],
                     "memory_updates": [], "sleep": False}),
    ])
    action_llm = ScriptedLLM(["done"])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=hypo_llm, action_llm=action_llm,
        hibernate_min=0.01, hibernate_max=5.0,
    )

    await runtime.buffer.push(Stimulus(
        type="user_message", source="test", content="go",
        timestamp=datetime.now(), adrenalin=True,
    ))
    await asyncio.wait_for(runtime.run(iterations=2), timeout=3.0)

    # Heartbeat #2 should have seen a batch_complete stimulus
    joined = json.dumps([m for m in self_llm.calls], ensure_ascii=False)
    assert "batch_complete" in joined or "All dispatched tentacles" in joined


async def test_explicit_write_from_hypothalamus_memory_writes():
    """Hypothalamus memory_writes trigger GM.explicit_write."""
    self_llm = ScriptedLLM([
        "[DECISION]\n记住: 用户偏好详细解释\n[HIBERNATE]\n1",
    ])
    hypo_llm = ScriptedLLM([
        json.dumps({
            "tentacle_calls": [],
            "memory_writes": [{"content": "user prefers detailed answers",
                                "importance": "high"}],
            "memory_updates": [], "sleep": False,
        })
    ])
    action_llm = ScriptedLLM([])
    # Classify/extractor LLM used by explicit_write
    extract_llm = ScriptedLLM([json.dumps({
        "nodes": [{"name": "user pref verbose",
                   "category": "FACT",
                   "description": "user prefers detailed answers"}],
        "edges": [],
    })])

    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=hypo_llm, action_llm=action_llm,
        classify_llm=extract_llm,
    )

    await runtime.buffer.push(Stimulus(
        type="user_message", source="test", content="please be detailed",
        timestamp=datetime.now(), adrenalin=True,
    ))
    await runtime.run(iterations=1)

    nodes = await runtime.gm.list_nodes()
    names = [n["name"] for n in nodes]
    assert "user pref verbose" in names


async def test_hibernate_interrupts_on_adrenalin_stimulus():
    self_llm = ScriptedLLM([
        "[DECISION]\nNo action.\n[HIBERNATE]\n5",  # long interval
        "[DECISION]\nNo action.\n[HIBERNATE]\n1",
    ])
    hypo_llm = ScriptedLLM([
        json.dumps({"tentacle_calls": [], "memory_writes": [],
                    "memory_updates": [], "sleep": False}),
        json.dumps({"tentacle_calls": [], "memory_writes": [],
                    "memory_updates": [], "sleep": False}),
    ])
    action_llm = ScriptedLLM([])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=hypo_llm, action_llm=action_llm,
        hibernate_min=0.01, hibernate_max=5.0)

    # Push adrenalin stimulus just after first iteration enters hibernate
    async def shouter():
        await asyncio.sleep(0.05)
        await runtime.buffer.push(Stimulus(
            type="user_message", source="test",
            content="urgent!", timestamp=datetime.now(), adrenalin=True,
        ))

    shout_task = asyncio.create_task(shouter())
    await asyncio.wait_for(runtime.run(iterations=2), timeout=2.0)
    await shout_task

    # Second heartbeat must have observed the urgent stimulus
    joined = json.dumps([m for m in self_llm.calls], ensure_ascii=False)
    assert "urgent!" in joined
