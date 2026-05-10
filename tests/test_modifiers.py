"""Modifier protocol + registry tests.

Pins:
  * Registry is role-keyed; second registration claiming the same
    role raises.
  * Prompt suppression: [ACTION FORMAT] layer is included by default
    (scripted DecisionEngine) and dropped when the LLM-translator
    DecisionEngine is wired in.
"""
from types import SimpleNamespace

import pytest

from krakey.interfaces.modifier import Modifier, ModifierRegistry
from tests._runtime_helpers import (
    NullEmbedder, ScriptedLLM, build_runtime_with_fakes,
)


# ---- registry --------------------------------------------------------


def test_registry_keys_by_role():
    reg = ModifierRegistry()

    class _A:
        name, role = "a", "alpha"

    class _B:
        name, role = "b", "beta"

    a, b = _A(), _B()
    reg.register(a)
    reg.register(b)

    assert reg.by_role("alpha") is a
    assert reg.by_role("beta") is b
    assert reg.by_role("nonexistent") is None
    assert reg.has_role("alpha") is True
    assert reg.has_role("nonexistent") is False


def test_registry_rejects_duplicate_role():
    """Two Modifiers claiming the same role is a startup error — the
    runtime can't pick which one to use."""
    reg = ModifierRegistry()

    class A:
        name, role = "a", "alpha"

    class B:
        name, role = "b", "alpha"

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
        role = "alpha"

    with pytest.raises(ValueError, match="role"):
        reg.register(NoRole())
    with pytest.raises(ValueError, match="name"):
        reg.register(NoName())


def test_registry_iteration_in_registration_order():
    reg = ModifierRegistry()

    class _A:
        name, role = "a", "alpha"

    class _B:
        name, role = "b", "beta"

    a, b = _A(), _B()
    reg.register(b)
    reg.register(a)
    assert reg.roles() == ["beta", "alpha"]
    assert reg.all() == [b, a]
    assert reg.names() == ["b", "a"]


# ---- Modifier base Protocol -----------------------------------------


def test_base_protocol_requires_name_and_role():
    class _M:
        name, role = "m", "m"

    assert isinstance(_M(), Modifier)


# ---- prompt suppression --------------------------------------------


async def test_prompt_includes_action_format_with_default_decision_engine(tmp_path):
    """Default DecisionEngine (scripted ``<tool_call>`` parser) → Self
    prompt MUST include the [ACTION FORMAT] block teaching the
    structured JSON syntax + the parser-flavored worked beat
    examples."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    await runtime.memory.initialize()
    runtime._recall = runtime.recall.new_session()
    from krakey.interfaces.engines.recall import RecallResult
    counts = SimpleNamespace(node_count=0, edge_count=0,
                              fatigue_pct=0, fatigue_hint="")
    prompt = runtime._build_self_prompt(
        stimuli=[],
        recall_result=RecallResult(),
        counts=counts,
    )
    assert "[ACTION FORMAT]" in prompt
    # Tool-call JSON syntax must be taught
    assert "<tool_call>" in prompt
    assert '"name"' in prompt
    # Worked examples must surface
    assert "Worked beat examples" in prompt


async def test_prompt_swaps_action_format_when_hypothalamus_decision_engine(tmp_path):
    """``HypothalamusDecisionEngine`` owns the LLM-translator path —
    it overwrites the [ACTION FORMAT] slot with the natural-language
    teaching layer (with NL-flavored worked examples) so Self isn't
    taught two competing dispatch syntaxes."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    # Swap in the LLM translator engine. Same Engine type the user
    # would wire via cfg.core_implementations.decision; we plug it
    # in directly here so the test doesn't need a full config rebuild.
    from krakey.engines.decision.hypothalamus import (
        HypothalamusDecisionEngine,
    )
    runtime.decision = HypothalamusDecisionEngine(
        cfg=runtime.config, factory=runtime.llm_factory,
    )

    await runtime.memory.initialize()
    runtime._recall = runtime.recall.new_session()
    from krakey.interfaces.engines.recall import RecallResult
    counts = SimpleNamespace(node_count=0, edge_count=0,
                              fatigue_pct=0, fatigue_hint="")
    prompt = runtime._build_self_prompt(
        stimuli=[],
        recall_result=RecallResult(),
        counts=counts,
    )
    # Slot is still present, but with NL-flavored teaching.
    assert "[ACTION FORMAT]" in prompt
    assert "natural language" in prompt
    assert "Worked beat examples" in prompt
    # Parser-flavored content must be ABSENT — no <tool_call> JSON
    # being taught when the hypothalamus owns dispatch.
    assert "<tool_call>" not in prompt
