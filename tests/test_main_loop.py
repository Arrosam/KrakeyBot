"""Integration test for Phase-0 main loop with fully mocked LLMs."""
import asyncio
import json
from datetime import datetime

import pytest

from src.main import Runtime, RuntimeDeps
from src.models.self_model import SelfModelStore, default_self_model
from tests._runtime_helpers import build_runtime_with_fakes
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


async def test_heartbeat_with_connected_recall_nodes_does_not_crash():
    """Regression: when GM has edges between recalled nodes, _layer_recall
    must render them without raising KeyError.
    """
    self_llm = ScriptedLLM([
        "[DECISION]\nNo action.\n[HIBERNATE]\n1",
    ])
    hypo_llm = ScriptedLLM([])
    action_llm = ScriptedLLM([])

    # Embedder maps text → specific vec so recall hits our seeded nodes.
    class MapEmbedder:
        async def __call__(self, text):
            if "apple" in text:
                return [1.0, 0.0]
            return [0.0, 1.0]

    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=hypo_llm, action_llm=action_llm,
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
        type="user_message", source="sensory:cli_input",
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
        json.dumps({"tentacle_calls": [], "memory_writes": [],
                     "memory_updates": [], "sleep": True}),
    ])
    action_llm = ScriptedLLM([])
    # Compact LLM doubles as the community-summary + KB-relations LLM
    sleep_llm = ScriptedLLM([
        "summary",  # community summary
        json.dumps({"edges": []}),  # KB relations (when 1 KB, not called)
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=hypo_llm, action_llm=action_llm,
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
    # self-model bookkeeping
    assert runtime.self_model["statistics"]["total_sleep_cycles"] == 1
    await runtime.close()


async def test_force_sleep_when_fatigue_exceeds_threshold(tmp_path):
    """When GM exceeds force_sleep_threshold, runtime triggers sleep
    immediately and pushes the special '昏睡' stimulus."""
    self_llm = ScriptedLLM([])  # never reached
    hypo_llm = ScriptedLLM([])
    action_llm = ScriptedLLM([])
    sleep_llm = ScriptedLLM(["summary"] * 10)
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=hypo_llm, action_llm=action_llm,
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
    action_llm = ScriptedLLM([])

    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=hypo_llm, action_llm=action_llm,
        skip_bootstrap=False,
    )
    # Override the self-model path to the tmp file
    runtime._self_model_store = SelfModelStore(sm_path)
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


async def test_override_kill_stops_runtime():
    self_llm = ScriptedLLM([
        # Heartbeat 1 should hit /kill before reaching Self.
        "[DECISION]\nNo action.\n[HIBERNATE]\n1",
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=ScriptedLLM([]), action_llm=ScriptedLLM([]),
    )
    await runtime.buffer.push(Stimulus(
        type="user_message", source="sensory:cli_input",
        content="/kill", timestamp=datetime.now(), adrenalin=True,
    ))
    await runtime.run(iterations=5)  # should exit before this many iterations
    await runtime.close()
    # Self LLM never called — kill short-circuited
    assert self_llm.calls == []


async def test_override_status_pushes_system_event_for_self(tmp_path):
    """Override result lands in buffer, visible on the *next* heartbeat."""
    self_llm = ScriptedLLM([
        "[DECISION]\nNo action.\n[HIBERNATE]\n1",  # HB #1: handles /status
        "[DECISION]\nNo action.\n[HIBERNATE]\n1",  # HB #2: sees system_event
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=ScriptedLLM([]), action_llm=ScriptedLLM([]),
    )
    await runtime.buffer.push(Stimulus(
        type="user_message", source="sensory:cli_input",
        content="/status", timestamp=datetime.now(), adrenalin=True,
    ))
    await runtime.run(iterations=2)
    await runtime.close()

    joined = json.dumps(self_llm.calls[1], ensure_ascii=False)
    assert "system:override" in joined
    assert "/status" in joined or "heartbeats=" in joined


async def test_override_sleep_triggers_full_sleep(tmp_path):
    sleep_llm = ScriptedLLM(["summary"] * 5)
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        action_llm=ScriptedLLM([]), compact_llm=sleep_llm,
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
        type="user_message", source="sensory:cli_input",
        content="/sleep", timestamp=datetime.now(), adrenalin=True,
    ))
    await runtime.run(iterations=1)
    # Inspect before close (close shuts down the GM connection).
    assert await runtime.gm.list_nodes(category="FACT") == []
    assert runtime.self_model["statistics"]["total_sleep_cycles"] == 1
    await runtime.close()


async def test_override_normal_text_passes_through_to_self():
    """Sanity: non-/cmd messages still reach Self normally."""
    self_llm = ScriptedLLM([
        "[DECISION]\nNo action.\n[HIBERNATE]\n1",
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=ScriptedLLM([]), action_llm=ScriptedLLM([]),
    )
    await runtime.buffer.push(Stimulus(
        type="user_message", source="sensory:cli_input",
        content="hello there", timestamp=datetime.now(), adrenalin=True,
    ))
    await runtime.run(iterations=1)
    await runtime.close()
    # Self saw "hello there" in its prompt
    joined = json.dumps(self_llm.calls[0], ensure_ascii=False)
    assert "hello there" in joined


async def test_memory_recall_renders_via_internal_not_chat(tmp_path):
    """Visual contract: memory_recall output must NOT be styled as Krakey's
    outward chat (green). Goes through logger.internal (magenta) instead."""
    self_llm = ScriptedLLM([
        "[DECISION]\nRecall apple.\n[HIBERNATE]\n1",
        "[DECISION]\nNo action.\n[HIBERNATE]\n1",
    ])
    hypo_llm = ScriptedLLM([
        json.dumps({
            "tentacle_calls": [{"tentacle": "memory_recall",
                                  "intent": "apple",
                                  "params": {"query": "apple"},
                                  "adrenalin": False}],
            "memory_writes": [], "memory_updates": [], "sleep": False,
        }),
        json.dumps({"tentacle_calls": [], "memory_writes": [],
                     "memory_updates": [], "sleep": False}),
    ])
    action_llm = ScriptedLLM([])

    class StubLogger:
        def __init__(self):
            self.internal_calls: list[tuple[str, str]] = []
            self.chat_calls: list[tuple[str, str]] = []
            self.heartbeat_id = 0

        def set_heartbeat(self, n): self.heartbeat_id = n
        def hb(self, msg): pass
        def hb_warn(self, msg): pass
        def hb_thought(self, label, text): pass
        def runtime_error(self, msg): pass
        def hypo(self, msg): pass
        def hypo_warn(self, msg): pass
        def dispatch(self, msg): pass
        def chat(self, sender, content):
            self.chat_calls.append((sender, content))
        def internal(self, sender, content):
            self.internal_calls.append((sender, content))

    spy = StubLogger()

    class MapEmbed:
        async def __call__(self, text):
            return [1.0, 0.0]

    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=hypo_llm, action_llm=action_llm,
        embedder=MapEmbed(),
    )
    runtime.log = spy
    await runtime.gm.initialize()
    await runtime.gm.insert_node(
        name="apple", category="FACT", description="red fruit",
        embedding=[1.0, 0.0],
    )

    await runtime.run(iterations=2)
    await runtime.close()

    # memory_recall went to internal, not chat
    assert any(s == "memory_recall" for (s, _) in spy.internal_calls)
    assert not any(s == "memory_recall" for (s, _) in spy.chat_calls)


async def test_self_can_dispatch_memory_recall_and_see_feedback():
    """Self → [DECISION] 'recall about apple' → Hypothalamus → memory_recall
    tentacle → tentacle_feedback in next heartbeat's stimuli."""
    self_llm = ScriptedLLM([
        # HB #1: Self decides to recall
        "[DECISION]\nRecall what I know about apple.\n[HIBERNATE]\n1",
        # HB #2: see recall result, take no further action
        "[DECISION]\nNo action.\n[HIBERNATE]\n1",
    ])
    hypo_llm = ScriptedLLM([
        json.dumps({
            "tentacle_calls": [{"tentacle": "memory_recall",
                                  "intent": "apple",
                                  "params": {"query": "apple"},
                                  "adrenalin": False}],
            "memory_writes": [], "memory_updates": [], "sleep": False,
        }),
        json.dumps({"tentacle_calls": [], "memory_writes": [],
                     "memory_updates": [], "sleep": False}),
    ])
    action_llm = ScriptedLLM([])

    class MapEmbed:
        async def __call__(self, text):
            return [1.0, 0.0] if "apple" in text else [0.0, 1.0]

    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=hypo_llm, action_llm=action_llm,
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

    # Heartbeat #2 must have shown a tentacle_feedback from memory_recall
    joined = json.dumps(self_llm.calls[1], ensure_ascii=False)
    assert "tentacle:memory_recall" in joined
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
    action_llm = ScriptedLLM([])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=hypo_llm, action_llm=action_llm,
    )

    # Seed one user stimulus that will never match (GM empty).
    orig = Stimulus(
        type="user_message", source="sensory:cli_input",
        content="hello", timestamp=datetime.now(), adrenalin=True,
    )
    await runtime.buffer.push(orig)

    await runtime.run(iterations=3)
    await runtime.close()

    # After 3 heartbeats, the original stim should have retry_count exactly 1
    # (pushed back once, then dropped). Buffer should not keep re-accumulating it.
    assert orig.metadata.get("recall_retries") == 1


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
