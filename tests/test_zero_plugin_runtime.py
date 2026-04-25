"""Zero-plugin invariant — runtime must survive with all plugins
disabled.

Set 2026-04-25 as a load-bearing design rule: disabling or removing
any plugin (Reflects, tentacles, sensories) must NOT break the
runtime's core loop. These tests pin the invariant; any future code
that introduces a hard plugin dependency will fail here and have to
add a fallback before merging.

Specific paths exercised:
  * No recall_anchor Reflect registered → ``make_recall`` returns a
    ``NoopRecall``, heartbeat completes with empty ``[GRAPH MEMORY]``.
  * No hypothalamus Reflect registered → ``dispatch_decision`` falls
    through to the action executor (already covered by
    test_reflects.py; sanity-pinned again here).
  * Zero registered tentacles → Self's tentacle calls produce
    ``Unknown tentacle: X`` system events, runtime keeps heartbeating.
  * All three at once → cold runtime + empty stimulus → still
    completes a heartbeat without raising.
"""
import pytest

from src.memory.recall import NoopRecall, RecallResult
from src.reflects import ReflectRegistry
from tests._runtime_helpers import (
    NullEmbedder, ScriptedLLM, build_runtime_with_fakes,
)


# ---- registry-level invariants ---------------------------------------


def test_make_recall_returns_noop_when_no_reflect_registered():
    """Core registry contract: the missing-plugin path returns a
    null-object, not a RuntimeError. This is the load-bearing
    behavior the rest of the runtime depends on."""
    reg = ReflectRegistry()
    recall = reg.make_recall(runtime=None)  # type: ignore[arg-type]
    assert isinstance(recall, NoopRecall)


async def test_noop_recall_satisfies_runtime_lifecycle(tmp_path):
    """The NoopRecall must respond to the same calls IncrementalRecall
    does (add_stimuli, finalize) without errors and without producing
    side effects Self would notice."""
    rec = NoopRecall()
    assert rec.processed_stimuli == []
    await rec.add_stimuli(["fake stimulus 1", "fake stimulus 2"])  # type: ignore[list-item]
    assert len(rec.processed_stimuli) == 2  # tracks dedup, no side effect
    result = await rec.finalize()
    assert isinstance(result, RecallResult)
    assert result.nodes == []
    assert result.edges == []


async def test_dispatch_decision_falls_back_to_executor_with_empty_registry():
    """Already covered by test_reflects.py; pinned here too because
    it's part of the zero-plugin invariant suite — any change that
    makes dispatch_decision require a Reflect breaks the contract."""
    reg = ReflectRegistry()  # nothing registered
    raw = '[ACTION]\n{"name": "noop", "arguments": {}}\n[/ACTION]'
    result = await reg.dispatch_decision(raw, "do nothing", [])
    assert len(result.tentacle_calls) == 1


# ---- runtime end-to-end with all Reflects stripped -------------------


async def test_runtime_heartbeat_survives_all_reflects_unregistered(tmp_path):
    """A heartbeat with zero Reflects, zero stimuli, and the action
    executor finding no [ACTION] blocks must complete without raising.

    This is the core invariant in its strongest form: the runtime can
    breathe in vacuum.
    """
    self_llm = ScriptedLLM([
        # Decision with no [ACTION] block — Self chose not to act.
        "[THINKING]\nQuiet beat.\n[DECISION]\nNo action.\n[HIBERNATE]\n1",
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    # Strip ALL Reflects — both the auto-registered defaults.
    runtime.reflects._by_kind.clear()
    assert runtime.reflects.has_hypothalamus() is False
    assert runtime.reflects.by_kind("recall_anchor") == []

    # Heartbeat completes without raising.
    await runtime.run(iterations=1)
    # No tentacle dispatches happened (Self's decision had no [ACTION]
    # block; runtime has no plugins to dispatch through anyway).
    # The fact we got here without an exception is the assertion.


async def test_runtime_heartbeat_with_no_tentacles_emits_unknown_tentacle(tmp_path):
    """Self emits an [ACTION] call referencing a tentacle name that
    doesn't exist in the registry. Runtime must NOT crash; it must
    push an `Unknown tentacle: ...` system event so Self can correct
    on the next beat."""
    self_llm = ScriptedLLM([
        '[THINKING]\nlet me reply.\n'
        '[DECISION]\nGreet.\n'
        '[ACTION]\n{"name": "nonexistent_tentacle", "arguments": {}}\n[/ACTION]\n'
        '[HIBERNATE]\n1'
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    # Strip the hypothalamus Reflect so dispatch goes via the action
    # executor (the path we want to exercise here).
    runtime.reflects._by_kind.pop("hypothalamus", None)

    # Replace the tentacle registry with an empty one so even a
    # well-formed call lands on a missing tentacle.
    from src.interfaces.tentacle import TentacleRegistry
    runtime.tentacles = TentacleRegistry()

    await runtime.run(iterations=1)
    # The buffer should now contain a system_event explaining the
    # missing tentacle, ready for Self to see next beat.
    drained = runtime.buffer.drain()
    unknown_events = [
        s for s in drained
        if s.type == "system_event"
        and "Unknown tentacle" in s.content
        and "nonexistent_tentacle" in s.content
    ]
    assert unknown_events, (
        "expected an 'Unknown tentacle: nonexistent_tentacle' system "
        "event so Self can self-correct, but none was pushed"
    )


async def test_runtime_construction_works_with_no_reflects(tmp_path):
    """Even Runtime.__init__ should tolerate a state where no Reflects
    end up registered — the registry is built, defaults register, and
    code that wants to remove them later should be free to."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    # Manually clear after construction — this is what config-driven
    # disabling will look like once that lands.
    runtime.reflects._by_kind.clear()
    # Subsequent operations on the registry must still work.
    assert runtime.reflects.has_hypothalamus() is False
    recall = runtime.reflects.make_recall(runtime)
    assert isinstance(recall, NoopRecall)
