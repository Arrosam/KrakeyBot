"""Phase 2.3d: full Sleep mode end-to-end."""
import json
from pathlib import Path

import pytest

from src.interfaces.sensory import PushCallback, Sensory
from src.memory.graph_memory import GraphMemory
from src.memory.knowledge_base import KBRegistry
from src.runtime.stimulus_buffer import StimulusBuffer
from src.memory.sleep.sleep_manager import enter_sleep_mode


class FixedEmbed:
    def __init__(self, mapping=None):
        self._m = mapping or {}

    async def __call__(self, text):
        return list(self._m.get(text, [0.5, 0.5]))


class CountingLLM:
    def __init__(self):
        self.responses_idx = 0
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        self.responses_idx += 1
        # KB relations LLM call also goes here; respond with empty edges.
        if "Knowledge Base" in str(messages) or "kb_id" in str(messages):
            return json.dumps({"edges": []})
        return f"summary #{self.responses_idx}"


class _SpySensory(Sensory):
    def __init__(self, name, urgent):
        self._name = name
        self._urgent = urgent
        self.paused = 0
        self.resumed = 0

    @property
    def name(self): return self._name

    @property
    def default_adrenalin(self): return self._urgent

    async def start(self, push: PushCallback):
        self.resumed += 1

    async def stop(self):
        self.paused += 1


async def _setup(tmp_path, with_sensory=False):
    embed = FixedEmbed()
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=embed)
    await gm.initialize()
    reg = KBRegistry(gm, kb_dir=tmp_path / "kbs", embedder=embed)
    # The buffer now also owns the sensory set (SensoryRegistry merged
    # into StimulusBuffer). Pass the buffer to enter_sleep_mode.
    sensories = StimulusBuffer()
    spies = []
    if with_sensory:
        calm = _SpySensory("calm", urgent=False)
        urgent = _SpySensory("urgent", urgent=True)
        sensories.register(calm)
        sensories.register(urgent)
        await sensories.start_all()
        spies = [calm, urgent]
    return gm, reg, sensories, spies


# ---------------- end-to-end ----------------

async def test_full_sleep_migrates_facts_clears_focus_keeps_target(tmp_path):
    gm, reg, sensories, _ = await _setup(tmp_path)
    fa = await gm.insert_node(name="apple", category="FACT", description="",
                                 embedding=[1.0, 0.0])
    fb = await gm.insert_node(name="banana", category="FACT", description="",
                                 embedding=[0.95, 0.31])
    await gm.insert_edge_with_cycle_check(fa, fb, "RELATED_TO")
    await gm.insert_node(name="t1", category="TARGET", description="open task",
                            embedding=[0.0, 1.0])
    await gm.insert_node(name="f1", category="FOCUS", description="bug X",
                            embedding=[0.0, 1.0])

    log_dir = tmp_path / "logs"
    result = await enter_sleep_mode(
        gm, reg, sensories, llm=CountingLLM(), embedder=FixedEmbed(),
        log_dir=log_dir,
    )

    # FACTs gone
    assert await gm.list_nodes(category="FACT") == []
    # FOCUS gone
    assert await gm.list_nodes(category="FOCUS") == []
    # TARGET preserved
    targets = await gm.list_nodes(category="TARGET")
    assert len(targets) == 1
    # KB created with 2 entries
    kbs = await reg.list_kbs()
    assert len(kbs) == 1
    kb = await reg.open_kb(kbs[0]["kb_id"])
    assert await kb.count_entries() == 2
    # Index node added
    knowledge = await gm.list_nodes(category="KNOWLEDGE")
    assert len(knowledge) == 1
    assert knowledge[0]["metadata"].get("is_kb_index") is True

    assert result["facts_migrated"] == 2
    assert result["focus_cleared"] == 1
    assert result["targets_preserved"] == 1
    assert result["kbs_created"] >= 1

    # Daily log written
    log_files = list(log_dir.glob("*.jsonl"))
    assert len(log_files) == 1
    line = log_files[0].read_text(encoding="utf-8").strip().splitlines()[0]
    rec = json.loads(line)
    assert rec["facts_migrated"] == 2
    await reg.close_all()
    await gm.close()


async def test_sleep_pauses_then_resumes_sensories(tmp_path):
    gm, reg, sensories, spies = await _setup(tmp_path, with_sensory=True)
    await gm.insert_node(name="x", category="FACT", description="",
                            embedding=[1.0, 0.0])

    await enter_sleep_mode(gm, reg, sensories, llm=CountingLLM(),
                              embedder=FixedEmbed(), log_dir=tmp_path / "logs")

    calm, urgent = spies
    # calm is non-urgent → paused (stop called)
    assert calm.paused == 1
    # urgent kept running through sleep (default_adrenalin=True)
    assert urgent.paused == 0
    # After resume_all, only previously-paused sensories restart
    assert calm.resumed >= 2  # initial start + resume
    await reg.close_all()
    await gm.close()


async def test_empty_gm_sleep_is_no_op_but_logs(tmp_path):
    gm, reg, sensories, _ = await _setup(tmp_path)
    log_dir = tmp_path / "logs"
    result = await enter_sleep_mode(gm, reg, sensories, llm=CountingLLM(),
                                       embedder=FixedEmbed(), log_dir=log_dir)
    assert result["facts_migrated"] == 0
    assert result["focus_cleared"] == 0
    assert result["targets_preserved"] == 0
    # Log still written
    assert list(log_dir.glob("*.jsonl"))
    await reg.close_all()
    await gm.close()


async def test_completed_target_via_hypothalamus_migration_path(tmp_path):
    """A TARGET that's been re-categorized to FACT (Hypothalamus 'task done')
    should migrate into the KB like any FACT — Phase 4 is implicit."""
    gm, reg, sensories, _ = await _setup(tmp_path)
    nid = await gm.insert_node(name="finished_task", category="TARGET",
                                  description="done", embedding=[1.0, 0.0])
    await gm.update_node_category("finished_task", "FACT")

    await enter_sleep_mode(gm, reg, sensories, llm=CountingLLM(),
                              embedder=FixedEmbed(),
                              log_dir=tmp_path / "logs")
    # Target-turned-FACT migrated → no longer in GM
    assert await gm.get_node(nid) is None
    await reg.close_all()
    await gm.close()
