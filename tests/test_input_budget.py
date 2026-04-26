"""Tests for overall prompt-input-budget enforcement
(Runtime._enforce_input_budget). The refactor replaced a single
`sliding_window.max_tokens` trigger with two coordinated budgets:
history (a fraction of max_input_tokens) and overall prompt (the full
max_input_tokens). This file exercises the overall-budget path that
prunes oldest rounds into GM when the assembled prompt won't fit.
"""
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from src.memory.recall import RecallResult
from src.models.stimulus import Stimulus
from src.runtime.sliding_window import SlidingWindowRound
from tests._runtime_helpers import (
    NullEmbedder, ScriptedLLM, build_runtime_with_fakes,
)


def _counts():
    """Minimal _GMCounts stand-in for prompt building."""
    return SimpleNamespace(
        node_count=0, edge_count=0, fatigue_pct=0, fatigue_hint="",
    )


async def _seed_recall(runtime):
    """The enforcer needs self._recall set (created by Runtime.run but
    not by the test helper's bare construction)."""
    runtime._recall = runtime._new_recall()


async def test_prompt_fitting_budget_is_left_alone(tmp_path):
    """No budget pressure \u2192 enforcement must not pop rounds or compact."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    await runtime.gm.initialize()
    await _seed_recall(runtime)

    runtime.window.append(SlidingWindowRound(
        heartbeat_id=1, stimulus_summary="hi", decision_text="ok",
        note_text="",
    ))
    empty_recall = RecallResult(nodes=[], edges=[],
                                  covered_stimuli=[], uncovered_stimuli=[])
    prompt, result = await runtime._enforce_input_budget(
        stimuli=[], recall_result=empty_recall, counts=_counts(),
    )
    assert len(runtime.window.rounds) == 1  # nothing popped
    assert result is empty_recall  # no re-recall triggered
    assert isinstance(prompt, str) and prompt  # something was built


async def test_oversized_prompt_prunes_oldest_round(tmp_path):
    """When the assembled prompt exceeds max_input_tokens, the enforcer
    pops oldest rounds into GM until it fits or the window is empty.
    Here we artificially starve the budget so even a single fat round
    has to go."""
    # Compact LLM returns valid (empty) JSON so extraction runs without
    # error; we only care that the round was popped + compact called.
    compact_calls = {"n": 0}

    class CountingCompact:
        async def chat(self, messages, **kwargs):
            compact_calls["n"] += 1
            return json.dumps({"nodes": [], "edges": []})

    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        compact_llm=CountingCompact(),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    await runtime.gm.initialize()
    await _seed_recall(runtime)

    # Starve the budget artificially. 200 tokens is far below DNA's
    # natural size, guaranteeing the prompt is always "too big" until
    # we've popped every round.
    runtime.config.llm.core_params("self_thinking").max_input_tokens = 200

    # Three fat rounds to make sure at least one gets popped.
    for i in range(3):
        runtime.window.append(SlidingWindowRound(
            heartbeat_id=i, stimulus_summary="x" * 500,
            decision_text="y" * 500, note_text="",
        ))
    empty_recall = RecallResult(nodes=[], edges=[],
                                  covered_stimuli=[], uncovered_stimuli=[])

    before = len(runtime.window.rounds)
    await runtime._enforce_input_budget(
        stimuli=[], recall_result=empty_recall, counts=_counts(),
    )
    after = len(runtime.window.rounds)
    assert after < before, "enforcer did not pop any round under pressure"
    assert compact_calls["n"] >= 1, "compact LLM was never invoked"


async def test_enforcer_tolerates_compact_llm_failure(tmp_path):
    """If the compact LLM blows up mid-enforcement, the round is
    still popped (we must make budget progress) and the next beat
    gets a loud warning but no crash."""
    class FailingCompact:
        async def chat(self, messages, **kwargs):
            raise RuntimeError("upstream 500")

    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        compact_llm=FailingCompact(),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    await runtime.gm.initialize()
    await _seed_recall(runtime)

    runtime.config.llm.core_params("self_thinking").max_input_tokens = 200
    runtime.window.append(SlidingWindowRound(
        heartbeat_id=1, stimulus_summary="x" * 500,
        decision_text="y" * 500, note_text="",
    ))
    empty_recall = RecallResult(nodes=[], edges=[],
                                  covered_stimuli=[], uncovered_stimuli=[])

    # Must not raise, and must pop the round anyway.
    await runtime._enforce_input_budget(
        stimuli=[], recall_result=empty_recall, counts=_counts(),
    )
    assert len(runtime.window.rounds) == 0


async def test_enforcer_stops_when_window_empty(tmp_path):
    """Window empty + prompt still too big \u2192 enforcer returns the
    oversized prompt and logs. Must NOT infinite-loop or raise."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    await runtime.gm.initialize()
    await _seed_recall(runtime)

    runtime.config.llm.core_params("self_thinking").max_input_tokens = 50
    assert len(runtime.window.rounds) == 0
    empty_recall = RecallResult(nodes=[], edges=[],
                                  covered_stimuli=[], uncovered_stimuli=[])
    prompt, result = await runtime._enforce_input_budget(
        stimuli=[], recall_result=empty_recall, counts=_counts(),
    )
    assert prompt  # we still got a prompt back
    assert result is empty_recall
