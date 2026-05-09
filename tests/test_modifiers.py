"""Modifier protocol + registry tests.

Pins:
  * Built-ins implement the role contract (each declares a unique
    role string).
  * Registry is role-keyed; second registration claiming the same
    role raises.
  * Runtime auto-registers the default modifiers.
  * Prompt suppression: [ACTION FORMAT] layer is included iff no
    Modifier has claimed role="hypothalamus".
"""
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from krakey.interfaces.modifier import (
    HypothalamusModifier, DecisionResult,
    Modifier, ModifierRegistry,
)
from krakey.plugins.hypothalamus.modifier import HypothalamusModifierImpl
from tests._runtime_helpers import (
    NullEmbedder, ScriptedLLM, build_runtime_with_fakes,
)


# ---- protocol compliance ----------------------------------------------


def test_hypothalamus_satisfies_protocol():
    r = HypothalamusModifierImpl(ScriptedLLM([]))
    assert isinstance(r, Modifier)
    assert isinstance(r, HypothalamusModifier)
    assert r.role == "hypothalamus"
    assert r.name == "hypothalamus"


# ---- registry --------------------------------------------------------


def test_registry_keys_by_role():
    reg = ModifierRegistry()
    h = HypothalamusModifierImpl(ScriptedLLM([]))

    class _Other:
        name, role = "other", "other"

    other = _Other()
    reg.register(h)
    reg.register(other)

    assert reg.by_role("hypothalamus") is h
    assert reg.by_role("other") is other
    assert reg.by_role("nonexistent") is None
    assert reg.has_role("hypothalamus") is True
    assert reg.has_role("nonexistent") is False


def test_registry_rejects_duplicate_role():
    """Two Modifiers claiming the same role is a startup error — the
    runtime can't pick which one to use."""
    reg = ModifierRegistry()

    class A:
        name, role = "a", "hypothalamus"

    class B:
        name, role = "b", "hypothalamus"

    reg.register(A())
    with pytest.raises(ValueError, match="role.*already claimed"):
        reg.register(B())


def test_registry_rejects_missing_name_or_role():
    reg = ModifierRegistry()

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
    reg = ModifierRegistry()
    h = HypothalamusModifierImpl(ScriptedLLM([]))

    class _Other:
        name, role = "other", "other"

    other = _Other()
    reg.register(other)
    reg.register(h)
    assert reg.roles() == ["other", "hypothalamus"]
    assert reg.all() == [other, h]
    assert reg.names() == ["other", "hypothalamus"]


# ---- runtime integration --------------------------------------------


async def test_runtime_registers_default_modifiers(tmp_path):
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    assert "hypothalamus" in runtime.modifiers.names()
    assert runtime.modifiers.has_role("hypothalamus")


async def test_runtime_no_longer_holds_hypothalamus_attribute(tmp_path):
    """Regression: Runtime used to expose `self.hypothalamus`. Now it's
    looked up via modifiers.by_role("hypothalamus")."""
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
    runtime.modifiers._by_role.pop("hypothalamus", None)
    runtime.modifiers._order.remove("hypothalamus")
    assert runtime.modifiers.has_role("hypothalamus") is False

    await runtime.memory.initialize()
    runtime._recall = runtime._new_recall()
    from krakey.interfaces.engines.recall import RecallResult
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
    assert runtime.modifiers.has_role("hypothalamus") is True

    await runtime.memory.initialize()
    runtime._recall = runtime._new_recall()
    from krakey.interfaces.engines.recall import RecallResult
    counts = SimpleNamespace(node_count=0, edge_count=0,
                              fatigue_pct=0, fatigue_hint="")
    prompt = runtime._build_self_prompt(
        stimuli=[],
        recall_result=RecallResult(),
        counts=counts,
    )
    assert "[ACTION FORMAT]" not in prompt
