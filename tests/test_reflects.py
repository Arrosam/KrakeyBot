"""Reflect protocol + registry skeleton tests.

The 2026-04-25 skeleton wraps the existing Hypothalamus +
IncrementalRecall factory as default built-in Reflects, with no
behavior change. These tests pin:
  * Built-ins implement their kind's Protocol structurally.
  * Registry preserves registration order per kind.
  * Length-1 chain dispatch matches what the equivalent direct call
    would have produced.
  * Length >1 chains raise NotImplementedError until the multi-Reflect
    composition lands with Reflect #1 / #2.
  * Bad Reflects (missing name/kind) are rejected at register-time
    rather than blowing up at dispatch-time.
"""
import json
from datetime import datetime

import pytest

from src.hypothalamus import HypothalamusResult
from src.memory.recall import IncrementalRecall
from src.reflects import (
    HypothalamusReflect, RecallAnchorReflect, Reflect, ReflectRegistry,
)
from src.reflects.builtin import (
    DefaultHypothalamusReflect, DefaultRecallAnchorReflect,
)
from tests._runtime_helpers import (
    NullEmbedder, ScriptedLLM, build_runtime_with_fakes,
)


# ---- protocol compliance ----------------------------------------------


def test_default_hypothalamus_satisfies_protocol():
    r = DefaultHypothalamusReflect(ScriptedLLM([]))
    assert isinstance(r, Reflect)
    assert isinstance(r, HypothalamusReflect)
    assert r.kind == "hypothalamus"
    assert r.name == "default_hypothalamus"


def test_default_recall_anchor_satisfies_protocol():
    r = DefaultRecallAnchorReflect()
    assert isinstance(r, Reflect)
    assert isinstance(r, RecallAnchorReflect)
    assert r.kind == "recall_anchor"
    assert r.name == "default_recall_anchor"


# ---- registry registration --------------------------------------------


def test_registry_groups_by_kind():
    reg = ReflectRegistry()
    h = DefaultHypothalamusReflect(ScriptedLLM([]))
    r = DefaultRecallAnchorReflect()
    reg.register(h)
    reg.register(r)

    assert reg.by_kind("hypothalamus") == [h]
    assert reg.by_kind("recall_anchor") == [r]
    assert reg.by_kind("nonexistent") == []


def test_registry_preserves_registration_order_within_a_kind():
    """Same-kind chain order = registration order. This is the
    contract Samuel locked: config.yaml's order IS execution order."""
    reg = ReflectRegistry()

    class A:
        name, kind = "a", "hypothalamus"
        async def translate(self, decision, tentacles):
            return HypothalamusResult()

    class B:
        name, kind = "b", "hypothalamus"
        async def translate(self, decision, tentacles):
            return HypothalamusResult()

    a, b = A(), B()
    reg.register(a)
    reg.register(b)
    assert reg.by_kind("hypothalamus") == [a, b]


def test_registry_rejects_missing_name_or_kind():
    reg = ReflectRegistry()

    class NoKind:
        name = "x"
        kind = ""

    class NoName:
        name = ""
        kind = "hypothalamus"

    with pytest.raises(ValueError, match="kind"):
        reg.register(NoKind())
    with pytest.raises(ValueError, match="name"):
        reg.register(NoName())


def test_registry_names_listing():
    reg = ReflectRegistry()
    reg.register(DefaultHypothalamusReflect(ScriptedLLM([])))
    reg.register(DefaultRecallAnchorReflect())
    assert set(reg.names()) == {
        "default_hypothalamus", "default_recall_anchor",
    }
    assert reg.names("hypothalamus") == ["default_hypothalamus"]
    assert reg.names("recall_anchor") == ["default_recall_anchor"]


# ---- dispatch ---------------------------------------------------------


async def test_translate_dispatches_through_default_hypothalamus():
    """Length-1 chain: registry.translate() returns exactly what the
    underlying default Reflect would have returned via direct call."""
    reg = ReflectRegistry()
    fake_llm = ScriptedLLM([json.dumps({
        "tentacle_calls": [{"tentacle": "search",
                            "intent": "find weather",
                            "params": {"q": "today"},
                            "adrenalin": False}],
        "memory_writes": [], "memory_updates": [], "sleep": False,
    })])
    reg.register(DefaultHypothalamusReflect(fake_llm))

    result = await reg.translate("check weather", [
        {"name": "search", "description": "web search",
         "parameters_schema": {}},
    ])
    assert len(result.tentacle_calls) == 1
    assert result.tentacle_calls[0].tentacle == "search"


async def test_translate_raises_when_no_hypothalamus_registered():
    reg = ReflectRegistry()
    reg.register(DefaultRecallAnchorReflect())  # wrong kind
    with pytest.raises(RuntimeError, match="hypothalamus"):
        await reg.translate("x", [])


async def test_translate_raises_on_chain_length_above_one():
    """Skeleton-phase guardrail: multi-Reflect composition is
    deliberately deferred. Two same-kind Reflects must be a loud
    failure until Reflect #1 lands proper chain semantics."""
    reg = ReflectRegistry()
    reg.register(DefaultHypothalamusReflect(ScriptedLLM([])))
    reg.register(DefaultHypothalamusReflect(ScriptedLLM([])))
    with pytest.raises(NotImplementedError, match="chain length"):
        await reg.translate("x", [])


async def test_make_recall_dispatches_through_default(tmp_path):
    """Registry.make_recall returns an IncrementalRecall with the same
    knobs the direct factory used to set."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    await runtime.gm.initialize()

    rec = runtime.reflects.make_recall(runtime)
    assert isinstance(rec, IncrementalRecall)
    # Knobs match the runtime's config slots.
    assert rec.per_k == runtime.config.graph_memory.recall_per_stimulus_k
    assert rec.recall_token_budget == (
        runtime.config.llm.roles["self"].params.recall_token_budget
    )
    assert rec.neighbor_depth == (
        runtime.config.graph_memory.neighbor_expand_depth
    )


# ---- runtime integration ---------------------------------------------


async def test_runtime_registers_default_reflects(tmp_path):
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    assert "default_hypothalamus" in runtime.reflects.names()
    assert "default_recall_anchor" in runtime.reflects.names()
    # And only those two — no surprise extras.
    assert set(runtime.reflects.names()) == {
        "default_hypothalamus", "default_recall_anchor",
    }


async def test_runtime_no_longer_holds_hypothalamus_attribute(tmp_path):
    """Regression: Runtime used to expose `self.hypothalamus`. After
    the skeleton refactor it goes through `self.reflects.translate(...)`.
    Tests that imported `runtime.hypothalamus` should fail loudly so
    we notice and migrate."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    assert not hasattr(runtime, "hypothalamus")
