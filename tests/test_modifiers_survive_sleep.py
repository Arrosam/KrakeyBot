"""Diagnostic regression: modifiers must survive a sleep cycle.

User-reported (2026-05-07): after sleep ends, the
``hypothalamus.modify_prompt`` and ``in_mind_note.modify_prompt``
hooks stop firing AND ``apply_decision`` no longer routes through
the hypothalamus translator. Symptom looks like the modifier
registry got mutated during sleep.

This file pins down the contract: the registry is set ONCE at
``Runtime.__init__`` and survives a ``_perform_sleep`` round-trip
unchanged. If this starts failing, sleep DID mutate the registry —
find the offending phase.

The fixture is built to mirror the user's actual symptom: no
hypothalamus (so the script-only dispatch path + the SleepTool
intercept drives sleep), in_mind_note IS registered (so the
modify_prompt hook has something visible to verify post-sleep).
"""
from __future__ import annotations

from krakey.models.self_model import SelfModelStore, default_self_model
from tests._runtime_helpers import build_runtime_with_fakes


class _ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    async def chat(self, messages, **kwargs):
        if not self._responses:
            return ""
        return self._responses.pop(0)


async def test_modifier_registry_unchanged_across_sleep(tmp_path):
    """Snapshot every Modifier instance pre-sleep, run a full sleep
    cycle via the built-in sleep tool, snapshot again. Identity
    must match — same Python objects, same roles, same registration
    order."""
    self_llm = _ScriptedLLM([
        '[DECISION]\nSleep now.\n'
        '<tool_call>{"name": "sleep"}</tool_call>\n[IDLE]\n1',
    ])
    sleep_llm = _ScriptedLLM(["summary"] * 5)
    # No hypothalamus → script-only action_executor path → SleepTool
    # intercept fires. recall + in_mind_note give us real Modifiers
    # to verify across sleep. dashboard supplies web_chat_reply.
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=_ScriptedLLM([]),
        compact_llm=sleep_llm,
        modifiers=["recall", "in_mind_note", "dashboard"],
    )
    runtime.sleep_log_dir = str(tmp_path / "logs")
    runtime._self_model_store = SelfModelStore(tmp_path / "self_model.yaml")
    runtime.self_model = default_self_model()
    await runtime.memory.initialize()
    await runtime.memory.insert_node(
        name="apple", category="FACT", description="red fruit",
        embedding=[1.0, 0.0],
    )

    pre_roles = runtime.modifiers.roles()
    pre_ids = {r: id(runtime.modifiers.by_role(r)) for r in pre_roles}
    pre_modify_present = {
        r: hasattr(runtime.modifiers.by_role(r), "modify_prompt")
        for r in pre_roles
    }
    assert pre_roles, (
        "test setup is broken — no modifiers registered, can't "
        "verify they survive sleep"
    )
    assert "in_mind" in pre_modify_present, (
        f"setup expected in_mind to be registered; got "
        f"{pre_modify_present.keys()}"
    )

    await runtime.run(iterations=1)

    # Sleep ran end-to-end (FACT migrated → no longer in GM).
    assert await runtime.memory.list_nodes(category="FACT") == [], (
        "sleep didn't actually execute — FACT still in GM"
    )
    assert runtime._sleep_cycles == 1

    post_roles = runtime.modifiers.roles()
    post_ids = {r: id(runtime.modifiers.by_role(r)) for r in post_roles}
    post_modify_present = {
        r: hasattr(runtime.modifiers.by_role(r), "modify_prompt")
        for r in post_roles
    }

    assert post_roles == pre_roles, (
        f"modifier role list changed across sleep: "
        f"pre={pre_roles} post={post_roles}"
    )
    assert post_ids == pre_ids, (
        f"modifier INSTANCES changed across sleep (different "
        f"object id): pre={pre_ids} post={post_ids}. Some sleep "
        f"phase silently re-registered modifiers."
    )
    assert post_modify_present == pre_modify_present, (
        f"modify_prompt presence changed across sleep: "
        f"pre={pre_modify_present} post={post_modify_present}"
    )

    await runtime.close()


async def test_hypothalamus_modify_prompt_fires_post_sleep(tmp_path):
    """Verify the user-reported scenario: hypothalamus IS enabled,
    a sleep cycle runs (via translator's sleep:true JSON), and on
    the next beat hypothalamus's modify_prompt MUST still fire
    (the [ACTION FORMAT] layer should be removed from the prompt
    because that's what its modify_prompt does).
    """
    import json
    self_llm = _ScriptedLLM([
        # beat 1: ask for sleep
        '[DECISION]\nEnter sleep mode.\n[IDLE]\n1',
        # beat 2 (post-sleep): no action — we just want to inspect
        # the assembled prompt
        '[DECISION]\nNo action.\n[IDLE]\n1',
    ])
    hypo_llm = _ScriptedLLM([
        json.dumps({"tool_calls": [], "memory_writes": [],
                    "memory_updates": [], "sleep": True}),
        json.dumps({"tool_calls": [], "memory_writes": [],
                    "memory_updates": [], "sleep": False}),
    ])
    sleep_llm = _ScriptedLLM(["summary"] * 5)
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, decision_translator_llm=hypo_llm,
        compact_llm=sleep_llm,
        # Default modifiers list (hypothalamus + recall + dashboard).
    )
    runtime.sleep_log_dir = str(tmp_path / "logs")
    runtime._self_model_store = SelfModelStore(tmp_path / "self_model.yaml")
    runtime.self_model = default_self_model()
    await runtime.memory.initialize()
    await runtime.memory.insert_node(
        name="apple", category="FACT", description="red fruit",
        embedding=[1.0, 0.0],
    )

    await runtime.run(iterations=2)
    assert runtime._sleep_cycles == 1, "sleep didn't run on beat 1"

    recent = runtime.recent_prompts()
    assert recent, "no prompts recorded"
    post_sleep_prompt = recent[0]["full_prompt"]

    # Hypothalamus's modify_prompt deletes the action_format layer.
    # If it stopped firing post-sleep, [ACTION FORMAT] would still
    # be present.
    assert "[ACTION FORMAT]" not in post_sleep_prompt, (
        "hypothalamus.modify_prompt did NOT fire post-sleep — "
        "the [ACTION FORMAT] layer is still in the prompt. "
        "Last 500 chars:\n" + post_sleep_prompt[-500:]
    )

    await runtime.close()


async def test_modify_prompt_fires_on_post_sleep_beat(tmp_path):
    """Behavior-side check: after sleep, the NEXT beat's prompt
    must STILL include the in_mind modifier's injected instructions
    block. If modify_prompt stops firing post-sleep, the prompt
    structure tells us instantly.
    """
    self_llm = _ScriptedLLM([
        '[DECISION]\nSleep.\n<tool_call>{"name": "sleep"}</tool_call>\n[IDLE]\n1',
        '[DECISION]\nNo action.\n[IDLE]\n1',
    ])
    sleep_llm = _ScriptedLLM(["summary"] * 5)
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=_ScriptedLLM([]),
        compact_llm=sleep_llm,
        modifiers=["recall", "in_mind_note", "dashboard"],
    )
    runtime.sleep_log_dir = str(tmp_path / "logs")
    runtime._self_model_store = SelfModelStore(tmp_path / "self_model.yaml")
    runtime.self_model = default_self_model()
    await runtime.memory.initialize()
    await runtime.memory.insert_node(
        name="apple", category="FACT", description="red fruit",
        embedding=[1.0, 0.0],
    )

    await runtime.run(iterations=2)
    # Confirm sleep happened on beat 1.
    assert runtime._sleep_cycles == 1, (
        "sleep didn't run; this test isn't exercising post-sleep behavior"
    )

    recent = runtime.recent_prompts()
    assert recent, "no prompts recorded"
    # recent_prompts is newest-first; index 0 = post-sleep beat.
    post_sleep_prompt = recent[0]["full_prompt"]

    # in_mind_note's modify_prompt sets `in_mind_instructions` to
    # IN_MIND_INSTRUCTIONS_LAYER on every beat. The layer text
    # contains a recognisable header. If the prompt is missing it,
    # in_mind.modify_prompt didn't fire.
    from krakey.plugins.in_mind_note.prompt import IN_MIND_INSTRUCTIONS_LAYER
    header_marker = IN_MIND_INSTRUCTIONS_LAYER.split("\n", 1)[0]
    assert header_marker in post_sleep_prompt, (
        f"in_mind.modify_prompt did NOT fire on the post-sleep beat. "
        f"The instructions layer header {header_marker!r} is absent "
        f"from the prompt. Last 500 chars of prompt:\n"
        f"{post_sleep_prompt[-500:]}"
    )

    await runtime.close()
