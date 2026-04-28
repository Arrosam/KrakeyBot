"""Integration test for Phase-0 main loop with fully mocked LLMs."""
import asyncio
import json
from datetime import datetime

import pytest

from krakey.main import Runtime, RuntimeDeps
from krakey.models.self_model import SelfModelStore, default_self_model
from tests._runtime_helpers import build_runtime_with_fakes
from krakey.models.stimulus import Stimulus
from krakey.runtime.stimuli.stimulus_buffer import StimulusBuffer


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


async def test_single_iteration_user_message_triggers_tool_dispatch():
    self_llm = ScriptedLLM([
        "[THINKING]\nuser said hello. reply.\n"
        "[DECISION]\nUse web_chat_reply to greet the user.\n"
        "[HIBERNATE]\n1"
    ])
    hypo_llm = ScriptedLLM([json.dumps({
        "tool_calls": [{"tool": "web_chat_reply",
                            "intent": "Hi there!",
                            "params": {"text": "Hi there!"},
                            "adrenalin": False}],
        "memory_writes": [], "memory_updates": [], "sleep": False,
    })])

    runtime = build_runtime_with_fakes(self_llm=self_llm, hypo_llm=hypo_llm)

    await runtime.buffer.push(Stimulus(
        type="user_message", source="channel:cli_input",
        content="hello", timestamp=datetime.now(), adrenalin=True,
    ))

    await runtime.run(iterations=1)

    # Wait for tool task to complete and push feedback
    await asyncio.sleep(0.05)
    remaining = runtime.buffer.drain()
    contents = [s.content for s in remaining]
    # web_chat_reply returns scripted "Sent to web chat (N chars)."
    assert any("Sent to web chat" in c for c in contents)


async def test_no_action_decision_runs_no_tool():
    self_llm = ScriptedLLM([
        "[DECISION]\nNo action.\n[HIBERNATE]\n1"
    ])
    hypo_llm = ScriptedLLM([json.dumps({
        "tool_calls": [], "memory_writes": [], "memory_updates": [],
        "sleep": False,
    })])

    runtime = build_runtime_with_fakes(self_llm=self_llm, hypo_llm=hypo_llm)
    await runtime.run(iterations=1)
    await asyncio.sleep(0.05)
    stims = runtime.buffer.drain()
    assert [s for s in stims if s.type == "tool_feedback"] == []


async def test_hypothalamus_error_pushes_system_event_stimulus():
    """When the Hypothalamus LLM returns junk / empty / raises, Self must
    still learn about it via a system_event stimulus on the next heartbeat,
    otherwise failed dispatches look like silent successes."""
    self_llm = ScriptedLLM([
        "[DECISION]\nDo something real.\n[HIBERNATE]\n1",
    ])
    # Invalid JSON → Hypothalamus._parse_json raises → caught → pushed
    hypo_llm = ScriptedLLM(["not json at all"])

    runtime = build_runtime_with_fakes(self_llm=self_llm, hypo_llm=hypo_llm)
    await runtime.run(iterations=1)

    stims = runtime.buffer.drain()
    decision_errs = [s for s in stims
                     if s.type == "system_event"
                     and s.source == "system:decision"]
    assert decision_errs, "no decision-dispatch error stimulus pushed"
    assert decision_errs[0].adrenalin is True
    assert "could not be translated" in decision_errs[0].content.lower()


async def test_tool_feedback_does_not_inherit_adrenalin_from_hypothalamus():
    """Successful tool_feedback is low-priority by design. Even when
    the Hypothalamus flags the dispatch as adrenalin=True (upstream urgency),
    the resulting feedback receipt should NOT wake Self out of hibernate —
    Self has already reacted to the urgent upstream stimulus; the echo is
    just bookkeeping."""
    self_llm = ScriptedLLM([
        "[DECISION]\nAct fast, user waiting.\n[HIBERNATE]\n1"
    ])
    hypo_llm = ScriptedLLM([json.dumps({
        "tool_calls": [{"tool": "web_chat_reply",
                            "intent": "go", "params": {"text": "go"},
                            "adrenalin": True}],
        "memory_writes": [], "memory_updates": [], "sleep": False,
    })])

    runtime = build_runtime_with_fakes(self_llm=self_llm, hypo_llm=hypo_llm)
    await runtime.run(iterations=1)

    await asyncio.sleep(0.05)
    stims = runtime.buffer.drain()
    tool_stims = [s for s in stims if s.type == "tool_feedback"]
    assert tool_stims
    # Feedback returned from a successful web_chat_reply → adrenalin=False.
    # Tool itself returned False; dispatch no longer inherits.
    assert tool_stims[0].adrenalin is False


async def test_heartbeat_with_connected_recall_nodes_does_not_crash():
    """Regression: when GM has edges between recalled nodes, _layer_recall
    must render them without raising KeyError.
    """
    self_llm = ScriptedLLM([
        "[DECISION]\nNo action.\n[HIBERNATE]\n1",
    ])
    hypo_llm = ScriptedLLM([])

    # Embedder maps text → specific vec so recall hits our seeded nodes.
    class MapEmbedder:
        async def __call__(self, text):
            if "apple" in text:
                return [1.0, 0.0]
            return [0.0, 1.0]

    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=hypo_llm,
        embedder=MapEmbedder(),
    )
    # Initialize GM first so we can seed it before running.
    await runtime.gm.initialize()
    a = await runtime.gm.insert_node(name="apple", category="FACT",
                                       description="red fruit",
                                       embedding=[1.0, 0.0])
    f = await runtime.gm.insert_node(name="fruit", category="KNOWLEDGE",
                                       description="category of foods",
                                       embedding=[0.99, 0.14])
    await runtime.gm.insert_edge_with_cycle_check(a, f, "RELATED_TO")

    # Seed a user stimulus that embeds close to apple → recall finds both,
    # plus the edge between them, exercising the builder's edge rendering.
    await runtime.buffer.push(Stimulus(
        type="user_message", source="channel:cli_input",
        content="tell me about apple", timestamp=datetime.now(),
        adrenalin=True,
    ))
    await runtime.run(iterations=1)
    await runtime.close()


async def test_voluntary_sleep_via_hypothalamus_runs_full_sleep(tmp_path):
    """Self → 'enter sleep' → Hypothalamus sleep:true → enter_sleep_mode
    runs end-to-end; FACT migrates to KB; wake-up stimulus pushed."""
    self_llm = ScriptedLLM([
        "[DECISION]\n进入睡眠模式\n[HIBERNATE]\n1",
    ])
    hypo_llm = ScriptedLLM([
        json.dumps({"tool_calls": [], "memory_writes": [],
                     "memory_updates": [], "sleep": True}),
    ])
    # Compact LLM doubles as the community-summary + KB-relations LLM
    sleep_llm = ScriptedLLM([
        "summary",  # community summary
        json.dumps({"edges": []}),  # KB relations (when 1 KB, not called)
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=hypo_llm,
        compact_llm=sleep_llm,
    )
    runtime.sleep_log_dir = str(tmp_path / "logs")
    # Isolate self-model from production workspace/self_model.yaml
    runtime._self_model_store = SelfModelStore(tmp_path / "self_model.yaml")
    runtime.self_model = default_self_model()
    # Pre-seed a FACT so sleep has something to migrate
    await runtime.gm.initialize()
    await runtime.gm.insert_node(
        name="apple", category="FACT", description="red fruit",
        embedding=[1.0, 0.0],
    )

    await runtime.run(iterations=1)

    # FACT migrated → no longer in GM
    facts = await runtime.gm.list_nodes(category="FACT")
    assert facts == []
    # KB created with 1 entry
    kbs = await runtime.kb_registry.list_kbs()
    assert len(kbs) == 1
    # Wake-up stimulus pushed
    drained = runtime.buffer.drain()
    assert any(s.source == "system:sleep" for s in drained)
    # Sleep counter is now in-memory (was self_model.statistics until
    # the 2026-04-25 slim refactor).
    assert runtime._sleep_cycles == 1
    await runtime.close()


async def test_force_sleep_when_fatigue_exceeds_threshold(tmp_path):
    """When GM exceeds force_sleep_threshold, runtime triggers sleep
    immediately and pushes the special '昏睡' stimulus."""
    self_llm = ScriptedLLM([])  # never reached
    hypo_llm = ScriptedLLM([])
    sleep_llm = ScriptedLLM(["summary"] * 10)
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=hypo_llm,
        compact_llm=sleep_llm,
    )
    runtime.sleep_log_dir = str(tmp_path / "logs")
    # Crank fatigue dial: tiny soft_limit, low force threshold
    runtime.config.fatigue.gm_node_soft_limit = 5
    runtime.config.fatigue.force_sleep_threshold = 100
    # Pre-seed enough nodes to push fatigue ≥ 100%
    await runtime.gm.initialize()
    for i in range(6):
        await runtime.gm.insert_node(
            name=f"n{i}", category="FACT", description=f"fact {i}",
            embedding=[1.0, 0.01 * i],
        )

    await runtime.run(iterations=1)

    # Self LLM should have NOT been called (force-sleep short-circuited)
    assert self_llm.calls == []
    # Wake-up stimulus is the special '昏睡' message
    drained = runtime.buffer.drain()
    wake = [s for s in drained if s.source == "system:sleep"]
    assert len(wake) == 1
    assert "昏睡" in wake[0].content
    # FACTs migrated
    assert await runtime.gm.list_nodes(category="FACT") == []
    await runtime.close()


async def test_bootstrap_self_model_update_and_completion(tmp_path):
    """Phase 2.1 end-to-end: Self in Bootstrap mode writes a <self-model>
    update and signals 'bootstrap complete' in [NOTE]; runtime persists the
    update to self_model.yaml and flips bootstrap_complete=True."""
    sm_path = tmp_path / "self_model.yaml"

    # HB #1: partial self-model update
    # HB #2: bootstrap complete signal
    self_llm = ScriptedLLM([
        ('[DECISION]\nNo action.\n'
         '[NOTE]\n<self-model>{"identity":{"name":"Krakey",'
         '"persona":"curious"}}</self-model>'),
        ('[DECISION]\nNo action.\n'
         '[NOTE]\nBootstrap complete'),
    ])
    hypo_llm = ScriptedLLM([])

    # Pre-seed the self-model file with defaults so the runtime + the
    # BootstrapCoordinator both anchor to the SAME file from
    # construction. (Post-construction swap of runtime._self_model_store
    # used to work but broke when the coordinator started holding its
    # own store reference — passing the path up-front is cleaner anyway.)
    sm_path.write_text(
        # Yaml-dump default content; SelfModelStore.load handles missing
        # too but we want a non-empty starting state for the test.
        "identity: {}\nstate: {bootstrap_complete: false}\n",
        encoding="utf-8",
    )
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=hypo_llm,
        skip_bootstrap=False,
    )
    # Re-anchor BOTH runtime + coordinator to the tmp self-model file.
    runtime._self_model_store = SelfModelStore(sm_path)
    runtime.bootstrap._store = runtime._self_model_store
    runtime.self_model = default_self_model()
    runtime.is_bootstrap = True
    # Tighten hibernate so the test isn't slow (bootstrap forces 10s default,
    # but max_interval=5 from fakes already clamps it).
    runtime._max = 0.1

    await runtime.run(iterations=2)
    await runtime.close()

    saved = SelfModelStore(sm_path).load()
    assert saved["identity"]["name"] == "Krakey"
    assert saved["identity"]["persona"] == "curious"
    assert saved["state"]["bootstrap_complete"] is True
    assert runtime.is_bootstrap is False


async def test_command_kill_stops_runtime():
    self_llm = ScriptedLLM([
        # Heartbeat 1 should hit /kill before reaching Self.
        "[DECISION]\nNo action.\n[HIBERNATE]\n1",
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=ScriptedLLM([]),
    )
    await runtime.buffer.push(Stimulus(
        type="user_message", source="channel:cli_input",
        content="/kill", timestamp=datetime.now(), adrenalin=True,
    ))
    await runtime.run(iterations=5)  # should exit before this many iterations
    await runtime.close()
    # Self LLM never called — kill short-circuited
    assert self_llm.calls == []


async def test_command_status_pushes_system_event_for_self(tmp_path):
    """Command result lands in buffer, visible on the *next* heartbeat."""
    self_llm = ScriptedLLM([
        "[DECISION]\nNo action.\n[HIBERNATE]\n1",  # HB #1: handles /status
        "[DECISION]\nNo action.\n[HIBERNATE]\n1",  # HB #2: sees system_event
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=ScriptedLLM([]),
    )
    await runtime.buffer.push(Stimulus(
        type="user_message", source="channel:cli_input",
        content="/status", timestamp=datetime.now(), adrenalin=True,
    ))
    await runtime.run(iterations=2)
    await runtime.close()

    joined = json.dumps(self_llm.calls[1], ensure_ascii=False)
    assert "system:command" in joined
    assert "/status" in joined or "heartbeats=" in joined


async def test_command_sleep_triggers_full_sleep(tmp_path):
    sleep_llm = ScriptedLLM(["summary"] * 5)
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        compact_llm=sleep_llm,
    )
    runtime.sleep_log_dir = str(tmp_path / "logs")
    runtime._self_model_store = SelfModelStore(tmp_path / "self_model.yaml")
    runtime.self_model = default_self_model()
    await runtime.gm.initialize()
    await runtime.gm.insert_node(
        name="apple", category="FACT", description="red fruit",
        embedding=[1.0, 0.0],
    )
    await runtime.buffer.push(Stimulus(
        type="user_message", source="channel:cli_input",
        content="/sleep", timestamp=datetime.now(), adrenalin=True,
    ))
    await runtime.run(iterations=1)
    # Inspect before close (close shuts down the GM connection).
    assert await runtime.gm.list_nodes(category="FACT") == []
    # In-memory counter (2026-04-25 self-model slim).
    assert runtime._sleep_cycles == 1
    await runtime.close()


async def test_normal_text_passes_through_to_self():
    """Sanity: non-/cmd messages still reach Self normally."""
    self_llm = ScriptedLLM([
        "[DECISION]\nNo action.\n[HIBERNATE]\n1",
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=ScriptedLLM([]),
    )
    await runtime.buffer.push(Stimulus(
        type="user_message", source="channel:cli_input",
        content="hello there", timestamp=datetime.now(), adrenalin=True,
    ))
    await runtime.run(iterations=1)
    await runtime.close()
    # Self saw "hello there" in its prompt
    joined = json.dumps(self_llm.calls[0], ensure_ascii=False)
    assert "hello there" in joined


async def test_self_can_dispatch_memory_recall_and_see_feedback():
    """Self → [DECISION] 'recall about apple' → Hypothalamus → memory_recall
    tool → tool_feedback in next heartbeat's stimuli."""
    self_llm = ScriptedLLM([
        # HB #1: Self decides to recall
        "[DECISION]\nRecall what I know about apple.\n[HIBERNATE]\n1",
        # HB #2: see recall result, take no further action
        "[DECISION]\nNo action.\n[HIBERNATE]\n1",
    ])
    hypo_llm = ScriptedLLM([
        json.dumps({
            "tool_calls": [{"tool": "memory_recall",
                                  "intent": "apple",
                                  "params": {"query": "apple"},
                                  "adrenalin": False}],
            "memory_writes": [], "memory_updates": [], "sleep": False,
        }),
        json.dumps({"tool_calls": [], "memory_writes": [],
                     "memory_updates": [], "sleep": False}),
    ])

    class MapEmbed:
        async def __call__(self, text):
            return [1.0, 0.0] if "apple" in text else [0.0, 1.0]

    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=hypo_llm,
        embedder=MapEmbed(),
    )
    # Pre-seed GM with an apple node so recall returns something concrete.
    await runtime.gm.initialize()
    await runtime.gm.insert_node(
        name="apple", category="FACT", description="red fruit",
        embedding=[1.0, 0.0],
    )

    await runtime.run(iterations=2)
    await runtime.close()

    # Heartbeat #2 must have shown a tool_feedback from memory_recall
    joined = json.dumps(self_llm.calls[1], ensure_ascii=False)
    assert "tool:memory_recall" in joined
    assert "apple" in joined


async def test_uncovered_stimulus_push_back_capped_at_one_retry():
    """Regression for the 'stimuli=5' bug: a user message with an empty GM
    finds no recall coverage. It must be pushed back at most ONCE, then
    dropped — otherwise it loops every heartbeat forever.
    """
    self_llm = ScriptedLLM([
        "[DECISION]\nNo action.\n[HIBERNATE]\n1",  # HB #1
        "[DECISION]\nNo action.\n[HIBERNATE]\n1",  # HB #2
        "[DECISION]\nNo action.\n[HIBERNATE]\n1",  # HB #3
    ])
    hypo_llm = ScriptedLLM([])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=hypo_llm,
    )

    # Seed one user stimulus that will never match (GM empty).
    orig = Stimulus(
        type="user_message", source="channel:cli_input",
        content="hello", timestamp=datetime.now(), adrenalin=True,
    )
    await runtime.buffer.push(orig)

    await runtime.run(iterations=3)
    await runtime.close()

    # After 3 heartbeats, the original stim should have retry_count exactly 1
    # (pushed back once, then dropped). Buffer should not keep re-accumulating it.
    assert orig.metadata.get("recall_retries") == 1


async def test_tool_feedback_auto_ingested_to_gm():
    """Phase 1: tool_feedback stimuli seen on next heartbeat get
    auto_ingested into Graph Memory."""
    self_llm = ScriptedLLM([
        "[DECISION]\nUse web_chat_reply to greet.\n[HIBERNATE]\n1",
        "[DECISION]\nNo action.\n[HIBERNATE]\n1",
    ])
    hypo_llm = ScriptedLLM([
        json.dumps({"tool_calls": [{
            "tool": "web_chat_reply",
            "intent": "Hello! Nice to meet you.",
            "params": {"text": "Hello! Nice to meet you."},
            "adrenalin": False,
        }],
                     "memory_writes": [], "memory_updates": [],
                     "sleep": False}),
        json.dumps({"tool_calls": [], "memory_writes": [],
                     "memory_updates": [], "sleep": False}),
    ])

    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=hypo_llm,
    )

    await runtime.buffer.push(Stimulus(
        type="user_message", source="channel:cli_input",
        content="hi", timestamp=datetime.now(), adrenalin=True,
    ))
    await runtime.run(iterations=2)
    nodes = await runtime.gm.list_nodes()
    contents = [n["description"] for n in nodes]
    # web_chat_reply's feedback "Sent to web chat (N chars)." gets auto_ingested
    assert any("sent to web chat" in c.lower() for c in contents), contents


async def test_batch_complete_stimulus_wakes_next_heartbeat():
    """After dispatch, BatchTracker fires a batch_complete adrenalin
    stimulus that Self sees on the subsequent heartbeat."""
    self_llm = ScriptedLLM([
        "[DECISION]\nUse web_chat_reply.\n[HIBERNATE]\n60",  # long interval
        "[DECISION]\nNo action.\n[HIBERNATE]\n1",
    ])
    hypo_llm = ScriptedLLM([
        json.dumps({"tool_calls": [{"tool": "web_chat_reply",
                                         "intent": "x",
                                         "params": {"text": "x"},
                                         "adrenalin": False}],
                     "memory_writes": [], "memory_updates": [],
                     "sleep": False}),
        json.dumps({"tool_calls": [], "memory_writes": [],
                     "memory_updates": [], "sleep": False}),
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=hypo_llm,
        hibernate_min=0.01, hibernate_max=5.0,
    )

    await runtime.buffer.push(Stimulus(
        type="user_message", source="test", content="go",
        timestamp=datetime.now(), adrenalin=True,
    ))
    await asyncio.wait_for(runtime.run(iterations=2), timeout=3.0)

    # Heartbeat #2 should have seen a batch_complete stimulus
    joined = json.dumps([m for m in self_llm.calls], ensure_ascii=False)
    assert "batch_complete" in joined or "All dispatched tools" in joined


async def test_explicit_write_from_hypothalamus_memory_writes():
    """Hypothalamus memory_writes trigger GM.explicit_write."""
    self_llm = ScriptedLLM([
        "[DECISION]\n记住: 用户偏好详细解释\n[HIBERNATE]\n1",
    ])
    hypo_llm = ScriptedLLM([
        json.dumps({
            "tool_calls": [],
            "memory_writes": [{"content": "user prefers detailed answers",
                                "importance": "high"}],
            "memory_updates": [], "sleep": False,
        })
    ])
    # Classify/extractor LLM used by explicit_write
    extract_llm = ScriptedLLM([json.dumps({
        "nodes": [{"name": "user pref verbose",
                   "category": "FACT",
                   "description": "user prefers detailed answers"}],
        "edges": [],
    })])

    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=hypo_llm,
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
        json.dumps({"tool_calls": [], "memory_writes": [],
                    "memory_updates": [], "sleep": False}),
        json.dumps({"tool_calls": [], "memory_writes": [],
                    "memory_updates": [], "sleep": False}),
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=hypo_llm,
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
