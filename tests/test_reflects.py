"""Reflect protocol + registry tests.

Pins:
  * Built-ins implement the role contract (each declares a unique
    role string).
  * Registry is role-keyed; second registration claiming the same
    role raises.
  * Runtime auto-registers the default reflects.
  * Prompt suppression: [ACTION FORMAT] layer is included iff no
    Reflect has claimed role="hypothalamus".
"""
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from src.interfaces.reflect import (
    HypothalamusReflect, DecisionResult, RecallAnchorReflect,
    Reflect, ReflectRegistry,
)
from src.plugins.default_hypothalamus.reflect import (
    DefaultHypothalamusReflect,
)
from src.plugins.recall_anchor.reflect import RecallAnchorReflectImpl
from tests._runtime_helpers import (
    NullEmbedder, ScriptedLLM, build_runtime_with_fakes,
)


# ---- protocol compliance ----------------------------------------------


def _stub_recall_anchor() -> RecallAnchorReflectImpl:
    """Build a RecallAnchorReflectImpl with placeholder state for
    tests that only inspect protocol attributes (role/name/registry
    behavior). make_recall is never invoked on these stubs."""
    return RecallAnchorReflectImpl(
        gm=None, embedder=None, reranker=None,  # type: ignore[arg-type]
        per_stimulus_k=0, neighbor_depth=0, recall_token_budget=0,
    )


def test_default_hypothalamus_satisfies_protocol():
    r = DefaultHypothalamusReflect(ScriptedLLM([]))
    assert isinstance(r, Reflect)
    assert isinstance(r, HypothalamusReflect)
    assert r.role == "hypothalamus"
    assert r.name == "default_hypothalamus"


def test_recall_anchor_satisfies_protocol():
    r = _stub_recall_anchor()
    assert isinstance(r, Reflect)
    assert isinstance(r, RecallAnchorReflect)
    assert r.role == "recall_anchor"
    assert r.name == "recall_anchor"


# ---- registry --------------------------------------------------------


def test_registry_keys_by_role():
    reg = ReflectRegistry()
    h = DefaultHypothalamusReflect(ScriptedLLM([]))
    r = _stub_recall_anchor()
    reg.register(h)
    reg.register(r)

    assert reg.by_role("hypothalamus") is h
    assert reg.by_role("recall_anchor") is r
    assert reg.by_role("nonexistent") is None
    assert reg.has_role("hypothalamus") is True
    assert reg.has_role("nonexistent") is False


def test_registry_rejects_duplicate_role():
    """Two Reflects claiming the same role is a startup error — the
    runtime can't pick which one to use."""
    reg = ReflectRegistry()

    class A:
        name, role = "a", "hypothalamus"

    class B:
        name, role = "b", "hypothalamus"

    reg.register(A())
    with pytest.raises(ValueError, match="role.*already claimed"):
        reg.register(B())


def test_registry_rejects_missing_name_or_role():
    reg = ReflectRegistry()

    class NoRole:
        name = "x"
        role = ""

    class NoName:
        name = ""
        role = "hypothalamus"

    with pytest.raises(ValueError, match="role"):
        reg.register(NoRole())
    with pytest.raises(ValueError, match="name"):
        reg.register(NoName())


def test_registry_iteration_in_registration_order():
    reg = ReflectRegistry()
    h = DefaultHypothalamusReflect(ScriptedLLM([]))
    r = _stub_recall_anchor()
    reg.register(r)
    reg.register(h)
    assert reg.roles() == ["recall_anchor", "hypothalamus"]
    assert reg.all() == [r, h]
    assert reg.names() == ["recall_anchor", "default_hypothalamus"]


# ---- runtime integration --------------------------------------------


async def test_runtime_registers_default_reflects(tmp_path):
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    assert "default_hypothalamus" in runtime.reflects.names()
    assert "recall_anchor" in runtime.reflects.names()
    # Roles registered:
    assert runtime.reflects.has_role("hypothalamus")
    assert runtime.reflects.has_role("recall_anchor")


async def test_runtime_no_longer_holds_hypothalamus_attribute(tmp_path):
    """Regression: Runtime used to expose `self.hypothalamus`. Now it's
    looked up via reflects.by_role("hypothalamus")."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    assert not hasattr(runtime, "hypothalamus")


# ---- prompt suppression --------------------------------------------


async def test_prompt_includes_action_format_when_no_hypothalamus(tmp_path):
    """No translator role registered → Self prompt MUST include the
    [ACTION FORMAT] block teaching the JSONL syntax."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    # Drop the auto-registered hypothalamus.
    runtime.reflects._by_role.pop("hypothalamus", None)
    runtime.reflects._order.remove("hypothalamus")
    assert runtime.reflects.has_role("hypothalamus") is False

    await runtime.gm.initialize()
    runtime._recall = runtime._new_recall()
    from src.memory.recall import RecallResult
    counts = SimpleNamespace(node_count=0, edge_count=0,
                              fatigue_pct=0, fatigue_hint="")
    prompt = runtime._build_self_prompt(
        stimuli=[],
        recall_result=RecallResult(),
        counts=counts,
    )
    assert "[ACTION FORMAT]" in prompt
    assert '{"name":' in prompt or '"name"' in prompt


async def test_prompt_omits_action_format_when_hypothalamus_active(tmp_path):
    """Translator role registered → suppress [ACTION FORMAT]."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    assert runtime.reflects.has_role("hypothalamus") is True

    await runtime.gm.initialize()
    runtime._recall = runtime._new_recall()
    from src.memory.recall import RecallResult
    counts = SimpleNamespace(node_count=0, edge_count=0,
                              fatigue_pct=0, fatigue_hint="")
    prompt = runtime._build_self_prompt(
        stimuli=[],
        recall_result=RecallResult(),
        counts=counts,
    )
    assert "[ACTION FORMAT]" not in prompt
