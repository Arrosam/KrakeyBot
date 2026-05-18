"""Edge tests for the voluntary-sleep refusal guard.

When Self emits the built-in ``sleep`` tool call but fatigue_pct is
STRICTLY LESS THAN ``min(config.fatigue.thresholds.keys())``, the
orchestrator must:

  - NOT enter sleep this beat (_sleep_cycles stays unchanged).
  - Push a ``system_event`` / ``system:sleep`` feedback Stimulus for
    the next beat with ``adrenalin=False`` whose content mentions the
    refusal (energy high / sleep refused) and includes both the
    current fatigue % and the minimum threshold %.

These tests are written BEFORE implementation and are expected to FAIL
until the guard is implemented.  They treat the orchestrator as a
black box — assertions only use observable state: ``_sleep_cycles``,
the drained ``StimulusBuffer``, and whether non-sleep tool dispatches
completed.

Harness conventions mirrored from the existing test suite:
  - build_runtime_with_fakes  (_runtime_helpers.py)
  - ScriptedLLM               (local copy, same pattern as test_main_loop.py)
  - SelfModelStore / default_self_model  for sleep-cycle isolation
  - runtime.run(iterations=1) to drive exactly one beat
  - runtime.buffer.drain()    to inspect what was pushed for the next beat
  - runtime._sleep_cycles     to detect whether sleep actually happened

Fatigue-pct control:  fatigue_pct = int(node_count / soft_limit * 100).
To get a desired pct, we set ``config.fatigue.gm_node_soft_limit`` to a
convenient value and pre-seed exactly that many GM nodes, mirroring the
approach in test_force_sleep_when_fatigue_exceeds_threshold.
"""
from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from krakey.models.self_model import SelfModelStore, default_self_model
from krakey.models.stimulus import Stimulus
from tests._runtime_helpers import build_runtime_with_fakes


# ---------------------------------------------------------------------------
# Local ScriptedLLM — same pattern as test_main_loop.py
# ---------------------------------------------------------------------------

class ScriptedLLM:
    """Returns responses from a queue; records prompts."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list = []

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        if not self._responses:
            return ""
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sleep_beat_llm() -> ScriptedLLM:
    """Self LLM that emits the built-in sleep tool call."""
    return ScriptedLLM([
        '[THINKING]\n(quiet beat)\n[DECISION]\nTime to rest.\n'
        '<tool_call>{"name": "sleep"}</tool_call>\n[IDLE]\n1',
    ])


def _idle_beat_llm() -> ScriptedLLM:
    """Self LLM that does nothing — clean beat, no sleep."""
    return ScriptedLLM([
        '[THINKING]\n(quiet beat)\n[DECISION]\nNo action.\n[IDLE]\n1',
    ])


async def _seed_nodes(runtime, count: int) -> None:
    """Pre-seed *count* FACT nodes so fatigue_pct == count / soft_limit * 100."""
    await runtime.memory.initialize()
    for i in range(count):
        await runtime.memory.insert_node(
            name=f"node_{i}",
            category="FACT",
            description=f"seeded fact {i}",
            embedding=[1.0, 0.01 * i],
        )


# ---------------------------------------------------------------------------
# Shared threshold config used across most tests.
#
# soft_limit=100, thresholds={50: ..., 75: ...}, force_sleep_threshold=120.
# fatigue_pct = node_count (since soft_limit==100).
# min(thresholds) = 50.
# "low fatigue" zone: fatigue_pct < 50, achieved by seeding < 50 nodes.
# "sufficient fatigue" zone: fatigue_pct >= 50, achieved by seeding >= 50 nodes.
# "force sleep" zone: fatigue_pct >= 120, achieved by seeding >= 120 nodes.
# ---------------------------------------------------------------------------

_THRESHOLDS = {
    50: "(may sleep when not busy)",
    75: "(fatigued; should proactively sleep)",
}
_SOFT_LIMIT = 100
_FORCE_THRESHOLD = 120
_MIN_THRESHOLD = min(_THRESHOLDS)  # 50


def _configure_fatigue(runtime) -> None:
    """Apply the shared fatigue config onto an already-built runtime."""
    runtime.config.fatigue.gm_node_soft_limit = _SOFT_LIMIT
    runtime.config.fatigue.force_sleep_threshold = _FORCE_THRESHOLD
    runtime.config.fatigue.thresholds = dict(_THRESHOLDS)


# ---------------------------------------------------------------------------
# Test 1 — Spec point 1:
#   voluntary sleep REFUSED when fatigue_pct < min(thresholds)
#   → no sleep, system_event/system:sleep queued, adrenalin=False
# ---------------------------------------------------------------------------

async def test_voluntary_sleep_refused_when_fatigue_below_min_threshold(tmp_path):
    """Sleep request rejected: fatigue_pct(30) < min_threshold(50).

    Observable expectations:
    - _sleep_cycles stays at 0 (no sleep transition occurred).
    - A system_event Stimulus is queued for the next beat with:
        type == "system_event"
        source == "system:sleep"
        adrenalin is False
        content mentions energy is high / sleep was refused
        content includes current fatigue % (30)
        content includes the minimum threshold % (50)
    """
    self_llm = _sleep_beat_llm()
    sleep_llm = ScriptedLLM(["summary"] * 5)

    runtime = build_runtime_with_fakes(
        self_llm=self_llm,
        hypo_llm=ScriptedLLM([]),
        compact_llm=sleep_llm,
        modifiers=[],  # no hypothalamus — built-in sleep path only
    )
    _configure_fatigue(runtime)
    runtime.sleep_log_dir = str(tmp_path / "logs")
    runtime._self_model_store = SelfModelStore(tmp_path / "self_model.yaml")
    runtime.self_model = default_self_model()

    # 30 nodes → fatigue_pct = 30 (< 50 = min threshold)
    await _seed_nodes(runtime, 30)

    sleep_cycles_before = runtime._sleep_cycles
    await runtime.run(iterations=1)
    await runtime.close()

    # --- sleep must NOT have happened ---
    assert runtime._sleep_cycles == sleep_cycles_before, (
        f"Sleep occurred despite fatigue_pct(30) < min_threshold({_MIN_THRESHOLD}); "
        f"guard should have refused it."
    )

    # --- refusal stimulus must be queued ---
    drained = runtime.buffer.drain()
    refusal_stims = [
        s for s in drained
        if s.type == "system_event" and s.source == "system:sleep"
    ]
    assert refusal_stims, (
        "Expected a system_event/system:sleep refusal Stimulus in the buffer; "
        f"got {[s.source for s in drained]}"
    )
    stim = refusal_stims[0]

    assert stim.adrenalin is False, (
        f"Refusal stimulus must have adrenalin=False; got {stim.adrenalin}"
    )

    content_lower = stim.content.lower()
    # Content must signal the reason (energy is high / refused / not needed).
    assert any(kw in content_lower for kw in ("energy", "refused", "refus", "high")), (
        f"Refusal Stimulus content must mention high energy or refusal; got:\n{stim.content!r}"
    )
    # Content must embed the current fatigue %.
    assert "30" in stim.content, (
        f"Refusal Stimulus content must include current fatigue %(30); got:\n{stim.content!r}"
    )
    # Content must embed the minimum threshold %.
    assert str(_MIN_THRESHOLD) in stim.content, (
        f"Refusal Stimulus content must include min threshold%({_MIN_THRESHOLD}); "
        f"got:\n{stim.content!r}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Spec point 2:
#   voluntary sleep ALLOWED when fatigue_pct >= min(thresholds)
# ---------------------------------------------------------------------------

async def test_voluntary_sleep_allowed_when_fatigue_at_or_above_min_threshold(tmp_path):
    """Sleep proceeds unblocked: fatigue_pct(60) >= min_threshold(50).

    Observable expectations:
    - _sleep_cycles increments by 1 (sleep ran).
    - No refusal Stimulus (system_event/system:sleep) with refusal content
      is pushed.  The wake-up Stimulus that enter_sleep_mode always pushes
      may be present — we only check there is no refusal-flavoured one.
    """
    self_llm = _sleep_beat_llm()
    sleep_llm = ScriptedLLM(["summary"] * 5)

    runtime = build_runtime_with_fakes(
        self_llm=self_llm,
        hypo_llm=ScriptedLLM([]),
        compact_llm=sleep_llm,
        modifiers=[],
    )
    _configure_fatigue(runtime)
    runtime.sleep_log_dir = str(tmp_path / "logs")
    runtime._self_model_store = SelfModelStore(tmp_path / "self_model.yaml")
    runtime.self_model = default_self_model()

    # 60 nodes → fatigue_pct = 60 (>= 50 = min threshold)
    await _seed_nodes(runtime, 60)

    sleep_cycles_before = runtime._sleep_cycles
    await runtime.run(iterations=1)
    await runtime.close()

    # Sleep MUST have run.
    assert runtime._sleep_cycles == sleep_cycles_before + 1, (
        f"Sleep should have proceeded when fatigue_pct(60) >= min_threshold({_MIN_THRESHOLD}); "
        f"_sleep_cycles={runtime._sleep_cycles}"
    )

    # No refusal Stimulus should be present.
    drained = runtime.buffer.drain()
    refusal_stims = [
        s for s in drained
        if s.type == "system_event"
        and s.source == "system:sleep"
        and any(kw in s.content.lower() for kw in ("refus", "energy is high"))
    ]
    assert not refusal_stims, (
        f"Unexpected refusal Stimulus found even though fatigue was sufficient: "
        f"{[s.content for s in refusal_stims]}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Spec point 3:
#   empty thresholds → NO enforcement — voluntary sleep proceeds even at 0%
# ---------------------------------------------------------------------------

async def test_voluntary_sleep_not_refused_when_thresholds_empty(tmp_path):
    """When config.fatigue.thresholds == {}, the guard does not apply.

    fatigue_pct = 0 (no nodes seeded), but since thresholds is empty there
    is no minimum threshold to enforce, so sleep must be allowed.
    """
    self_llm = _sleep_beat_llm()
    sleep_llm = ScriptedLLM(["summary"] * 5)

    runtime = build_runtime_with_fakes(
        self_llm=self_llm,
        hypo_llm=ScriptedLLM([]),
        compact_llm=sleep_llm,
        modifiers=[],
    )
    # Empty thresholds — guard must be disabled.
    runtime.config.fatigue.gm_node_soft_limit = _SOFT_LIMIT
    runtime.config.fatigue.force_sleep_threshold = _FORCE_THRESHOLD
    runtime.config.fatigue.thresholds = {}

    runtime.sleep_log_dir = str(tmp_path / "logs")
    runtime._self_model_store = SelfModelStore(tmp_path / "self_model.yaml")
    runtime.self_model = default_self_model()

    # 0 nodes → fatigue_pct = 0
    await runtime.memory.initialize()

    sleep_cycles_before = runtime._sleep_cycles
    await runtime.run(iterations=1)
    await runtime.close()

    assert runtime._sleep_cycles == sleep_cycles_before + 1, (
        "With empty thresholds the guard must not block voluntary sleep at 0% fatigue; "
        f"_sleep_cycles={runtime._sleep_cycles}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Spec point 4:
#   forced sleep (fatigue_pct >= force_sleep_threshold) still sleeps;
#   the guard must NOT block it even when low-fatigue condition would apply.
# ---------------------------------------------------------------------------

async def test_forced_sleep_not_blocked_by_guard(tmp_path):
    """Force-sleep fires regardless of the voluntary-sleep guard.

    Config: thresholds={50:...}, force_sleep_threshold=100, soft_limit=100.
    We seed 100 nodes → fatigue_pct = 100 >= force_sleep_threshold.
    Force-sleep is triggered BEFORE Self is called, so Self never emits
    sleep — yet sleep_cycles must still increment.
    """
    self_llm = ScriptedLLM([])  # never reached in force-sleep path
    sleep_llm = ScriptedLLM(["summary"] * 10)

    runtime = build_runtime_with_fakes(
        self_llm=self_llm,
        hypo_llm=ScriptedLLM([]),
        compact_llm=sleep_llm,
        modifiers=[],
    )
    # force_sleep_threshold == soft_limit so seeding 100 nodes gives pct==100
    runtime.config.fatigue.gm_node_soft_limit = 100
    runtime.config.fatigue.force_sleep_threshold = 100
    runtime.config.fatigue.thresholds = {50: "(may sleep)"}

    runtime.sleep_log_dir = str(tmp_path / "logs")
    runtime._self_model_store = SelfModelStore(tmp_path / "self_model.yaml")
    runtime.self_model = default_self_model()

    # 100 nodes → pct=100 >= force_sleep_threshold(100) — triggers forced sleep.
    await _seed_nodes(runtime, 100)

    sleep_cycles_before = runtime._sleep_cycles
    await runtime.run(iterations=1)
    await runtime.close()

    # Force-sleep must have run.
    assert runtime._sleep_cycles == sleep_cycles_before + 1, (
        "Forced sleep must not be blocked by the voluntary-sleep guard; "
        f"_sleep_cycles={runtime._sleep_cycles}"
    )
    # Self was never called (force-sleep short-circuits the beat).
    assert self_llm.calls == [], (
        "Self LLM must not be called when force-sleep fires"
    )


# ---------------------------------------------------------------------------
# Test 5 — Spec point 5:
#   operator /sleep command still sleeps regardless of fatigue level.
# ---------------------------------------------------------------------------

async def test_operator_sleep_command_not_refused_when_energy_high(tmp_path):
    """The /sleep slash-command bypasses the voluntary-sleep guard.

    fatigue_pct = 0 (no nodes), thresholds configured so pct < min would
    normally trigger a refusal — but operator-initiated sleep MUST proceed.
    """
    sleep_llm = ScriptedLLM(["summary"] * 5)

    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]),
        hypo_llm=ScriptedLLM([]),
        compact_llm=sleep_llm,
        modifiers=[],
    )
    _configure_fatigue(runtime)
    runtime.sleep_log_dir = str(tmp_path / "logs")
    runtime._self_model_store = SelfModelStore(tmp_path / "self_model.yaml")
    runtime.self_model = default_self_model()

    # 0 nodes → fatigue_pct = 0 (well below min threshold=50).
    await runtime.memory.initialize()

    # Operator pushes /sleep command — this is not a Self tool call.
    await runtime.buffer.push(Stimulus(
        type="user_message",
        source="channel:cli_input",
        content="/sleep",
        timestamp=datetime.now(),
        adrenalin=True,
    ))

    sleep_cycles_before = runtime._sleep_cycles
    await runtime.run(iterations=1)
    await runtime.close()

    assert runtime._sleep_cycles == sleep_cycles_before + 1, (
        "Operator /sleep command must not be blocked by the voluntary-sleep guard "
        f"even at fatigue_pct=0; _sleep_cycles={runtime._sleep_cycles}"
    )


# ---------------------------------------------------------------------------
# Test 6 — Spec point 6:
#   a non-sleep tool call emitted alongside a refused sleep still dispatches.
# ---------------------------------------------------------------------------

async def test_non_sleep_tool_dispatches_when_sleep_is_refused(tmp_path):
    """Tool calls in the same beat as a refused sleep still run normally.

    Self emits both ``web_chat_reply`` and ``sleep`` in one [DECISION] block.
    With fatigue_pct(10) < min_threshold(50):
    - The sleep call is refused (no sleep transition).
    - The web_chat_reply call is dispatched and produces tool_feedback.
    """
    self_llm = ScriptedLLM([
        '[THINKING]\n(quiet beat)\n[DECISION]\nReply and rest.\n'
        '<tool_call>{"name": "web_chat_reply", "arguments": {"text": "hello"}}</tool_call>\n'
        '<tool_call>{"name": "sleep"}</tool_call>\n[IDLE]\n1',
    ])

    runtime = build_runtime_with_fakes(
        self_llm=self_llm,
        hypo_llm=ScriptedLLM([]),
        compact_llm=ScriptedLLM([]),
        modifiers=["dashboard"],  # provides web_chat_reply tool
    )
    _configure_fatigue(runtime)

    # 10 nodes → fatigue_pct = 10 (< 50 = min threshold)
    await _seed_nodes(runtime, 10)

    sleep_cycles_before = runtime._sleep_cycles
    await runtime.run(iterations=1)
    await asyncio.sleep(0.05)  # allow dispatched tool task to complete
    await runtime.close()

    # Sleep must NOT have occurred.
    assert runtime._sleep_cycles == sleep_cycles_before, (
        "Sleep must have been refused when fatigue_pct(10) < min_threshold(50)"
    )

    # web_chat_reply feedback must be in the buffer.
    drained = runtime.buffer.drain()
    tool_fb = [s for s in drained if s.type == "tool_feedback"]
    assert tool_fb, (
        "Expected tool_feedback from web_chat_reply even though sleep was refused; "
        f"got stimulus types: {[s.type for s in drained]}"
    )
    assert any("web chat" in s.content.lower() for s in tool_fb), (
        f"Expected web_chat_reply feedback but got: {[s.content for s in tool_fb]}"
    )


# ---------------------------------------------------------------------------
# Test 7 — Spec point 7 (BVA):
#   boundary — fatigue_pct exactly == min(thresholds) is NOT refused
#   (strictly-less-than semantics)
# ---------------------------------------------------------------------------

async def test_voluntary_sleep_allowed_at_exactly_min_threshold(tmp_path):
    """Boundary: fatigue_pct == min(thresholds) must NOT be refused.

    The guard fires only when pct is STRICTLY LESS THAN min(thresholds).
    At the boundary (pct == 50 == min threshold) sleep must proceed.
    """
    self_llm = _sleep_beat_llm()
    sleep_llm = ScriptedLLM(["summary"] * 5)

    runtime = build_runtime_with_fakes(
        self_llm=self_llm,
        hypo_llm=ScriptedLLM([]),
        compact_llm=sleep_llm,
        modifiers=[],
    )
    _configure_fatigue(runtime)
    runtime.sleep_log_dir = str(tmp_path / "logs")
    runtime._self_model_store = SelfModelStore(tmp_path / "self_model.yaml")
    runtime.self_model = default_self_model()

    # Exactly 50 nodes → fatigue_pct = 50 == min_threshold(50)
    # pct is NOT strictly less than the minimum → must NOT be refused.
    await _seed_nodes(runtime, _MIN_THRESHOLD)  # 50 nodes

    sleep_cycles_before = runtime._sleep_cycles
    await runtime.run(iterations=1)
    await runtime.close()

    assert runtime._sleep_cycles == sleep_cycles_before + 1, (
        f"Sleep at fatigue_pct == min_threshold({_MIN_THRESHOLD}) must NOT be refused "
        f"(strictly-less-than semantics); _sleep_cycles={runtime._sleep_cycles}"
    )

    # Confirm there is no refusal Stimulus either.
    drained = runtime.buffer.drain()
    refusal_stims = [
        s for s in drained
        if s.type == "system_event"
        and s.source == "system:sleep"
        and any(kw in s.content.lower() for kw in ("refus", "energy is high"))
    ]
    assert not refusal_stims, (
        f"No refusal Stimulus should exist at the boundary fatigue_pct == min_threshold; "
        f"got: {[s.content for s in refusal_stims]}"
    )


# ---------------------------------------------------------------------------
# Additional BVA — fatigue_pct one below the boundary (min_threshold - 1)
# ---------------------------------------------------------------------------

async def test_voluntary_sleep_refused_at_one_below_min_threshold(tmp_path):
    """BVA: fatigue_pct == min_threshold - 1 == 49 is refused.

    This is the last integer inside the 'refused' zone — the guard must fire.
    """
    self_llm = _sleep_beat_llm()
    sleep_llm = ScriptedLLM(["summary"] * 5)

    runtime = build_runtime_with_fakes(
        self_llm=self_llm,
        hypo_llm=ScriptedLLM([]),
        compact_llm=sleep_llm,
        modifiers=[],
    )
    _configure_fatigue(runtime)
    runtime.sleep_log_dir = str(tmp_path / "logs")
    runtime._self_model_store = SelfModelStore(tmp_path / "self_model.yaml")
    runtime.self_model = default_self_model()

    # 49 nodes → fatigue_pct = 49 (one below min_threshold=50)
    await _seed_nodes(runtime, _MIN_THRESHOLD - 1)

    sleep_cycles_before = runtime._sleep_cycles
    await runtime.run(iterations=1)
    await runtime.close()

    assert runtime._sleep_cycles == sleep_cycles_before, (
        f"Sleep must be refused when fatigue_pct({_MIN_THRESHOLD - 1}) "
        f"< min_threshold({_MIN_THRESHOLD})"
    )

    drained = runtime.buffer.drain()
    refusal_stims = [
        s for s in drained
        if s.type == "system_event" and s.source == "system:sleep"
    ]
    assert refusal_stims, (
        f"Expected a refusal Stimulus at fatigue_pct({_MIN_THRESHOLD - 1}) "
        f"< min_threshold({_MIN_THRESHOLD})"
    )
    assert refusal_stims[0].adrenalin is False
    # Both current pct and threshold must appear in the content.
    assert str(_MIN_THRESHOLD - 1) in refusal_stims[0].content
    assert str(_MIN_THRESHOLD) in refusal_stims[0].content


# ---------------------------------------------------------------------------
# Additional positive — single-entry thresholds dict (edge case for min())
# ---------------------------------------------------------------------------

async def test_voluntary_sleep_refused_with_single_threshold_entry(tmp_path):
    """Guard works when thresholds has exactly one key.

    thresholds={40: "..."} → min = 40.
    fatigue_pct = 20 (< 40) → refused.
    """
    self_llm = _sleep_beat_llm()
    sleep_llm = ScriptedLLM(["summary"] * 5)

    runtime = build_runtime_with_fakes(
        self_llm=self_llm,
        hypo_llm=ScriptedLLM([]),
        compact_llm=sleep_llm,
        modifiers=[],
    )
    runtime.config.fatigue.gm_node_soft_limit = 100
    runtime.config.fatigue.force_sleep_threshold = 120
    runtime.config.fatigue.thresholds = {40: "(single threshold)"}

    runtime.sleep_log_dir = str(tmp_path / "logs")
    runtime._self_model_store = SelfModelStore(tmp_path / "self_model.yaml")
    runtime.self_model = default_self_model()

    # 20 nodes → pct = 20 (< 40)
    await _seed_nodes(runtime, 20)

    sleep_cycles_before = runtime._sleep_cycles
    await runtime.run(iterations=1)
    await runtime.close()

    assert runtime._sleep_cycles == sleep_cycles_before, (
        "Guard must fire with a single-entry threshold dict when pct < threshold"
    )

    drained = runtime.buffer.drain()
    refusal_stims = [
        s for s in drained
        if s.type == "system_event" and s.source == "system:sleep"
    ]
    assert refusal_stims, "Expected refusal Stimulus for single-entry thresholds"
    assert refusal_stims[0].adrenalin is False
