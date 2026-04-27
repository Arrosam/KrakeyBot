"""Reflect #3 — default_in_mind.

Coverage:
  * State store: missing file / load+save round-trip / corrupted JSON
    fallback / atomic write / now_iso bumped on update.
  * Reflect: read / partial update / explicit clear via empty string /
    None means leave-alone / timestamp updates.
  * Tentacle: dispatch via [ACTION] JSONL → state mutated + feedback
    receipt names what changed.
  * Prompt injection: virtual round appears in [HISTORY] when state
    populated; absent when all fields empty; instructions layer
    present iff in_mind Reflect registered.
  * Zero-plugin invariant: no in_mind Reflect → no virtual round, no
    instructions layer, runtime fine.
  * attach() registers the update_in_mind tentacle; double-attach is
    tolerated.
"""
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.memory.recall import RecallResult
from src.plugins.default_in_mind.reflect import (
    InMindReflectImpl, build_reflect,
)
from src.plugins.default_in_mind.state import (
    InMindState, load, now_iso, save,
)
from src.plugins.default_in_mind.prompt import (
    IN_MIND_INSTRUCTIONS_LAYER, render_virtual_round,
)
from tests._runtime_helpers import (
    NullEmbedder, ScriptedLLM, build_runtime_with_fakes,
)


# ---- state store -----------------------------------------------------


def test_load_missing_file_returns_empty_state(tmp_path):
    s = load(tmp_path / "does_not_exist.json")
    assert s == InMindState()
    assert s.is_empty()


def test_save_then_load_round_trip(tmp_path):
    p = tmp_path / "in_mind.json"
    save(InMindState(thoughts="t", mood="m", focus="f",
                       updated_at="2026-04-25T00:00:00"), p)
    loaded = load(p)
    assert loaded.thoughts == "t"
    assert loaded.mood == "m"
    assert loaded.focus == "f"
    assert loaded.updated_at == "2026-04-25T00:00:00"


def test_load_corrupted_json_returns_empty_with_warning(tmp_path, capsys):
    p = tmp_path / "in_mind.json"
    p.write_text("{not valid json", encoding="utf-8")
    s = load(p)
    assert s == InMindState()
    err = capsys.readouterr().err
    assert "unreadable" in err or "warning" in err


def test_load_non_object_returns_empty_with_warning(tmp_path, capsys):
    p = tmp_path / "in_mind.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")  # valid JSON, wrong shape
    s = load(p)
    assert s == InMindState()


def test_save_creates_parent_directory(tmp_path):
    """First-run safety: ``workspace/data/`` may not exist yet."""
    nested = tmp_path / "deep" / "nested" / "in_mind.json"
    save(InMindState(thoughts="t"), nested)
    assert nested.exists()


def test_save_atomic_no_temp_litter_on_success(tmp_path):
    """The temp file used for the atomic replace must be cleaned up
    on the success path."""
    p = tmp_path / "in_mind.json"
    save(InMindState(thoughts="t"), p)
    siblings = list(p.parent.iterdir())
    assert siblings == [p], (
        f"unexpected leftover temp files: "
        f"{[s.name for s in siblings if s != p]}"
    )


def test_now_iso_returns_parseable_isoformat():
    s = now_iso()
    # Round-trips through datetime.fromisoformat without raising.
    datetime.fromisoformat(s)


def test_state_is_empty_predicate():
    assert InMindState().is_empty()
    assert InMindState(thoughts="x").is_empty() is False
    assert InMindState(mood="x").is_empty() is False
    assert InMindState(focus="x").is_empty() is False
    # updated_at alone doesn't count as content
    assert InMindState(updated_at="2026-01-01T00:00:00").is_empty()


# ---- reflect -- read / update semantics ------------------------------


def test_reflect_starts_empty_when_state_file_absent(tmp_path):
    r = InMindReflectImpl(state_path=tmp_path / "in_mind.json")
    assert r.read() == {
        "thoughts": "", "mood": "", "focus": "", "updated_at": "",
    }


def test_update_partial_only_touches_named_fields(tmp_path):
    p = tmp_path / "in_mind.json"
    r = InMindReflectImpl(state_path=p)
    r.update(thoughts="t1", mood="m1", focus="f1")
    snapshot1 = r.read()
    # Now update only `thoughts`
    r.update(thoughts="t2")
    snap2 = r.read()
    assert snap2["thoughts"] == "t2"
    assert snap2["mood"] == "m1"  # untouched
    assert snap2["focus"] == "f1"
    # updated_at advanced
    assert snap2["updated_at"] >= snapshot1["updated_at"]


def test_update_empty_string_clears_field(tmp_path):
    r = InMindReflectImpl(state_path=tmp_path / "in_mind.json")
    r.update(thoughts="t", mood="m", focus="f")
    r.update(mood="")  # explicit clear
    snap = r.read()
    assert snap["thoughts"] == "t"
    assert snap["mood"] == ""
    assert snap["focus"] == "f"


def test_update_none_field_means_leave_alone(tmp_path):
    r = InMindReflectImpl(state_path=tmp_path / "in_mind.json")
    r.update(thoughts="t", mood="m", focus="f")
    before = r.read()
    r.update(thoughts=None, mood=None, focus=None)
    # Nothing actually passed — shouldn't bump updated_at
    after = r.read()
    assert after["thoughts"] == before["thoughts"]
    assert after["mood"] == before["mood"]
    assert after["focus"] == before["focus"]
    assert after["updated_at"] == before["updated_at"]


def test_update_persists_across_reflect_instances(tmp_path):
    p = tmp_path / "in_mind.json"
    InMindReflectImpl(state_path=p).update(thoughts="persist me")
    # New instance reads from disk
    assert InMindReflectImpl(state_path=p).read()["thoughts"] == "persist me"


def test_build_reflect_factory_signature():
    """build_reflect takes a PluginContext; reads the deps' state
    path override (None for the default behavior)."""
    from src.interfaces.plugin_context import PluginContext
    fake_deps = SimpleNamespace(in_mind_state_path=None)
    ctx = PluginContext(deps=fake_deps, plugin_name="default_in_mind",
                          config={})
    r = build_reflect(ctx)
    assert isinstance(r, InMindReflectImpl)
    assert r.kind == "in_mind"
    assert r.name == "default_in_mind"


# ---- prompt rendering helpers ----------------------------------------


def test_render_virtual_round_skips_when_all_empty():
    assert render_virtual_round(InMindState()) is None


def test_render_virtual_round_includes_only_nonempty_fields():
    out = render_virtual_round(InMindState(thoughts="t", mood="", focus="f"))
    assert out is not None
    assert "Thoughts: t" in out
    assert "Focus: f" in out
    # Empty mood line should NOT be rendered
    assert "Mood:" not in out


def test_instructions_layer_mentions_update_in_mind():
    """Standing instruction must reference the tentacle name so Self
    knows what to call."""
    assert "update_in_mind" in IN_MIND_INSTRUCTIONS_LAYER
    assert "[HISTORY]" in IN_MIND_INSTRUCTIONS_LAYER


# ---- runtime integration: prompt + tentacle dispatch -----------------


def _counts():
    return SimpleNamespace(
        node_count=0, edge_count=0, fatigue_pct=0, fatigue_hint="",
    )


async def test_runtime_prompt_omits_in_mind_layers_when_no_reflect(
    tmp_path,
):
    """No in_mind Reflect registered → no virtual round, no
    instructions layer. Zero-plugin invariant."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
        reflects=[],  # explicitly nothing
    )
    await runtime.gm.initialize()
    runtime._recall = runtime._new_recall()
    prompt = runtime._build_self_prompt(
        stimuli=[], recall_result=RecallResult(), counts=_counts(),
    )
    assert "[IN MIND" not in prompt
    assert "--- Heartbeat #now (in mind) ---" not in prompt


async def test_runtime_prompt_includes_instructions_when_in_mind_active(
    tmp_path,
):
    """in_mind Reflect registered → instructions layer present even
    if the state itself is empty (no virtual round in that case)."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
        reflects=["default_in_mind"],
    )
    await runtime.gm.initialize()
    runtime._recall = runtime._new_recall()
    prompt = runtime._build_self_prompt(
        stimuli=[], recall_result=RecallResult(), counts=_counts(),
    )
    # Standing instructions present
    assert "[IN MIND" in prompt
    # State is empty → no virtual round (the literal "--- Heartbeat
    # #now (in mind) ---" sentinel is what we look for, not the
    # phrase itself, which the instructions text references).
    assert "--- Heartbeat #now (in mind) ---" not in prompt


async def test_runtime_prompt_includes_virtual_round_when_state_set(
    tmp_path,
):
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
        reflects=["default_in_mind"],
    )
    # Mutate the in_mind Reflect's state directly so we don't need to
    # round-trip through the tentacle for this prompt-shape test.
    in_mind_chain = runtime.reflects.by_kind("in_mind")
    assert in_mind_chain
    in_mind_chain[0].update(
        thoughts="thinking about Cython hot loops",
        focus="port the inner loop",
    )
    await runtime.gm.initialize()
    runtime._recall = runtime._new_recall()
    prompt = runtime._build_self_prompt(
        stimuli=[], recall_result=RecallResult(), counts=_counts(),
    )
    assert "Heartbeat #now (in mind)" in prompt
    assert "Thoughts: thinking about Cython hot loops" in prompt
    assert "Focus: port the inner loop" in prompt


async def test_attach_registers_update_in_mind_tentacle(tmp_path):
    """After attach_all runs (called at the end of Runtime.__init__),
    update_in_mind must be in the tentacle registry."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
        reflects=["default_in_mind"],
    )
    assert "update_in_mind" in runtime.tentacles


async def test_attach_tolerates_pre_existing_tentacle(tmp_path):
    """Re-attaching (e.g. in a test that calls attach_all again) must
    not crash — it should log + skip."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
        reflects=["default_in_mind"],
    )
    # Second attach shouldn't raise
    runtime.reflects.attach_all(runtime)
    assert "update_in_mind" in runtime.tentacles


async def test_self_can_dispatch_update_in_mind_via_action_executor(
    tmp_path,
):
    """End-to-end via the ACTION executor path: Self emits an
    [ACTION] block calling update_in_mind, runtime dispatches, state
    file gets written, feedback stimulus arrives in the buffer.
    """
    self_llm = ScriptedLLM([
        '[THINKING]\nshifting focus to debug session.\n'
        '[DECISION]\nUpdate in_mind.\n'
        '[ACTION]\n'
        '{"name": "update_in_mind", "arguments": '
        '{"focus": "debug recall regression", "mood": "curious"}}\n'
        '[/ACTION]\n'
        '[HIBERNATE]\n1'
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
        reflects=["default_in_mind"],  # only in_mind, no hypothalamus
    )
    # The helper has already provisioned a tmpdir state file via
    # RuntimeDeps.in_mind_state_path, so the Reflect's state lives
    # there — not in production. Capture the path for the post-run
    # assertion below.
    in_mind = runtime.reflects.by_kind("in_mind")[0]
    isolated_state_path = in_mind._state_path

    await runtime.run(iterations=1)

    # State got the update
    snap = in_mind.read()
    assert snap["focus"] == "debug recall regression"
    assert snap["mood"] == "curious"
    # Feedback stimulus landed in buffer
    drained = runtime.buffer.drain()
    feedback = [
        s for s in drained
        if s.type == "tentacle_feedback"
        and s.source == "tentacle:update_in_mind"
    ]
    assert feedback, "no feedback stimulus from update_in_mind"
    assert "in_mind updated" in feedback[0].content
    # State file written into the isolated tmp path the helper set up
    saved = json.loads(isolated_state_path.read_text("utf-8"))
    assert saved["focus"] == "debug recall regression"
