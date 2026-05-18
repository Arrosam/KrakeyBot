"""Edge tests: [IDLE] tag is now required in Self's structured output,
UNLESS the raw response contains a built-in sleep tool call
(``<tool_call>{"name":"sleep"}</tool_call>``).

These tests are written BEFORE the implementation exists and are
expected to FAIL until the orchestrator's ``_phase_run_self`` is
updated.

Spec under test
---------------
- ``[IDLE]`` is added to the required-tag set alongside ``[THINKING]``
  and ``[DECISION]``.
- Waiver: if the raw Self response contains a ``<tool_call>`` block
  whose ``name`` JSON field equals ``SLEEP_TOOL_NAME`` ("sleep"), the
  ``[IDLE]`` requirement is waived for that attempt and the beat
  proceeds normally.
- The waiver is SPECIFIC to the sleep tool — no other tool call grants
  the waiver.
- Natural-language "enter sleep mode" with no sleep ``<tool_call>`` and
  no ``[IDLE]`` is NOT waived; it still retries.
- ``[THINKING]`` / ``[DECISION]`` remain required exactly as today.
- Existing retry mechanics (fast/slow tiers, heartbeat_count frozen,
  round not saved while invalid) are unchanged.

Observable assertions (black-box only)
----------------------------------------
- ``self_llm.call_count`` or ``len(self_llm.calls)`` — retries → >1.
- ``runtime.explicit_history.rounds`` — non-empty IFF the beat produced
  a valid response that was saved.
- ``runtime._sleep_cycles`` — incremented IFF sleep actually ran.
- ``runtime.buffer.drain()`` — post-beat stimuli (sleep wake-up, etc.)

Harness / conventions mirrored from existing tests
---------------------------------------------------
- ``build_runtime_with_fakes``  (_runtime_helpers.py)
- Local ``ScriptedLLM`` class (same shape as test_main_loop.py)
- ``runtime.config.idle.*`` knobs to keep retries fast/deterministic
  (``llm_failure_retry_interval=0.01``, ``struct_output_fast_retries=2``,
  ``struct_output_slow_retry_interval=0.02``).
- ``runtime.run(iterations=1)`` to drive exactly one beat.
- ``modifiers=[]`` to exclude hypothalamus (built-in parser path only).
"""
from __future__ import annotations

import asyncio

import pytest

from tests._runtime_helpers import build_runtime_with_fakes


# ---------------------------------------------------------------------------
# Local ScriptedLLM — same pattern as test_main_loop.py
# ---------------------------------------------------------------------------

class ScriptedLLM:
    """Returns responses from a pre-loaded queue; records every call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list = []

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        if not self._responses:
            return ""
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FAST_RETRY_INTERVAL = 0.01
_FAST_RETRY_COUNT = 2
_SLOW_RETRY_INTERVAL = 0.02


def _configure_fast_retries(runtime) -> None:
    """Apply near-zero retry intervals so the struct-output loop does not
    slow tests down.  Mirrors the approach in test_main_loop.py's
    ``test_struct_output_retry_*`` tests."""
    runtime.config.idle.llm_failure_retry_interval = _FAST_RETRY_INTERVAL
    runtime.config.idle.struct_output_fast_retries = _FAST_RETRY_COUNT
    runtime.config.idle.struct_output_slow_retry_interval = _SLOW_RETRY_INTERVAL


def _good_response_with_idle() -> str:
    """Fully-valid Self response: THINKING + DECISION + IDLE present."""
    return (
        "[THINKING]\n(quiet beat)\n"
        "[DECISION]\nNo action.\n"
        "[IDLE]\n1"
    )


def _response_missing_idle() -> str:
    """Response has THINKING + DECISION but NO [IDLE] and no sleep tool."""
    return (
        "[THINKING]\n(quiet beat)\n"
        "[DECISION]\nNo action.\n"
    )


def _response_with_sleep_tool_no_idle() -> str:
    """Response has THINKING + DECISION + sleep tool call but NO [IDLE].
    Under the new spec this is VALID because the sleep tool waives [IDLE]."""
    return (
        "[THINKING]\n(quiet beat)\n"
        "[DECISION]\nTime to rest.\n"
        '<tool_call>{"name": "sleep"}</tool_call>\n'
    )


def _response_nl_sleep_no_idle() -> str:
    """Response says 'enter sleep mode' in natural language with THINKING +
    DECISION but has NO [IDLE] and NO sleep tool call.
    NOT waived — must still retry."""
    return (
        "[THINKING]\n(quiet beat)\n"
        "[DECISION]\nenter sleep mode\n"
    )


def _response_other_tool_no_idle(tool_name: str = "web_chat_reply") -> str:
    """Response has THINKING + DECISION + a non-sleep tool call but NO
    [IDLE].  The non-sleep tool must NOT grant the waiver."""
    return (
        "[THINKING]\n(quiet beat)\n"
        "[DECISION]\nReply to user.\n"
        f'<tool_call>{{"name": "{tool_name}", "arguments": {{"text": "hi"}}}}</tool_call>\n'
    )


# ---------------------------------------------------------------------------
# Test 1 — Spec point 1:
#   [THINKING]+[DECISION], NO [IDLE], no sleep tool → struct-output RETRY
#   fires; recovery with subsequent valid response succeeds.
# ---------------------------------------------------------------------------

async def test_missing_idle_tag_triggers_struct_retry_then_recovers():
    """[THINKING]+[DECISION] but NO [IDLE] and no sleep tool-call.

    Expected:
    - Self is called MORE THAN ONCE (first call malformed, retried).
    - After the first malformed response the beat has NOT yet saved a
      history round (round would only be saved after a valid response).
    - A follow-up valid response (with [IDLE]) lets the beat complete
      successfully → exactly one history round is appended, heartbeat
      count is 1.
    """
    # First response: THINKING + DECISION but missing [IDLE] (malformed)
    # Second response: full valid response so the beat recovers
    self_llm = ScriptedLLM([
        _response_missing_idle(),
        _good_response_with_idle(),
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm,
        hypo_llm=ScriptedLLM([]),
        modifiers=[],
    )
    _configure_fast_retries(runtime)

    await runtime.run(iterations=1)
    await runtime.close()

    # Retry fired: Self was called at least twice.
    assert len(self_llm.calls) >= 2, (
        f"Expected ≥ 2 Self calls (1 malformed + 1 valid); "
        f"got {len(self_llm.calls)}. "
        "The [IDLE]-missing response must trigger a struct-output retry."
    )

    # Beat recovered: exactly 1 heartbeat completed.
    assert runtime.heartbeat_count == 1, (
        f"heartbeat_count should be 1 after one successful beat; "
        f"got {runtime.heartbeat_count}"
    )

    # A history round was saved (the valid response was persisted).
    assert len(runtime.explicit_history.rounds) == 1, (
        f"Expected exactly 1 saved history round after beat recovery; "
        f"got {len(runtime.explicit_history.rounds)}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Spec point 2:
#   [THINKING]+[DECISION]+[IDLE] → valid first try, beat proceeds, no retry.
# ---------------------------------------------------------------------------

async def test_full_valid_response_proceeds_without_retry():
    """[THINKING]+[DECISION]+[IDLE] is valid on the first call.

    Expected:
    - Self called exactly once.
    - History round saved, heartbeat_count == 1.
    """
    self_llm = ScriptedLLM([_good_response_with_idle()])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm,
        hypo_llm=ScriptedLLM([]),
        modifiers=[],
    )
    _configure_fast_retries(runtime)

    await runtime.run(iterations=1)
    await runtime.close()

    assert len(self_llm.calls) == 1, (
        f"Expected exactly 1 Self call for a fully-valid response; "
        f"got {len(self_llm.calls)}"
    )
    assert runtime.heartbeat_count == 1
    assert len(runtime.explicit_history.rounds) == 1


# ---------------------------------------------------------------------------
# Test 3 — Spec point 3:
#   [THINKING]+[DECISION] + sleep tool call, NO [IDLE] → VALID first try
#   (waiver), beat proceeds, sleep is honored.
# ---------------------------------------------------------------------------

async def test_sleep_tool_call_waives_idle_requirement():
    """Sleep tool call present, [IDLE] absent → waiver applies, valid first
    try.

    Expected:
    - Self called exactly once (no retry).
    - Sleep ran: runtime._sleep_cycles == 1.
    - A system:sleep wake-up stimulus is in the buffer.
    - A history round was NOT saved (sleep short-circuits after _phase_run_self).
      (Technically _phase_save_round runs but _perform_sleep exits early via
      'return' — asserting that sleep ran is the primary check; we don't
      assert on round count here because the spec doesn't require the beat
      to be round-less when sleep fires.)
    """
    self_llm = ScriptedLLM([_response_with_sleep_tool_no_idle()])
    sleep_llm = ScriptedLLM(["summary"] * 5)
    runtime = build_runtime_with_fakes(
        self_llm=self_llm,
        hypo_llm=ScriptedLLM([]),
        compact_llm=sleep_llm,
        modifiers=[],
    )
    _configure_fast_retries(runtime)

    await runtime.memory.initialize()
    await runtime.run(iterations=1)
    await runtime.close()

    # No retry — waiver applied on first call.
    assert len(self_llm.calls) == 1, (
        f"Expected exactly 1 Self call (sleep waiver must not cause a "
        f"retry); got {len(self_llm.calls)}"
    )

    # Sleep actually ran.
    assert runtime._sleep_cycles == 1, (
        f"Expected _sleep_cycles == 1 after sleep-waiver beat; "
        f"got {runtime._sleep_cycles}. "
        "The beat must proceed AND honor the sleep tool call."
    )

    # Wake-up stimulus is present.
    drained = runtime.buffer.drain()
    wake_stims = [s for s in drained if s.source == "system:sleep"]
    assert wake_stims, (
        "Expected a system:sleep wake-up Stimulus after the sleep cycle ran; "
        f"got stimulus sources: {[s.source for s in drained]}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Spec point 4:
#   Missing [DECISION] (or [THINKING]) still retries even if [IDLE] present.
#   THINKING/DECISION requirement is unchanged.
# ---------------------------------------------------------------------------

async def test_missing_decision_still_retries_even_if_idle_present():
    """[THINKING]+[IDLE] present but [DECISION] missing → still retries.

    The [IDLE]-waiver for sleep does not affect the THINKING/DECISION
    requirement.  Missing DECISION triggers the existing struct-output retry.

    Recovery: a second response with all three tags lets the beat complete.
    """
    response_no_decision = (
        "[THINKING]\n(quiet beat)\n"
        "[IDLE]\n1"
    )
    self_llm = ScriptedLLM([
        response_no_decision,
        _good_response_with_idle(),
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm,
        hypo_llm=ScriptedLLM([]),
        modifiers=[],
    )
    _configure_fast_retries(runtime)

    await runtime.run(iterations=1)
    await runtime.close()

    assert len(self_llm.calls) >= 2, (
        f"Expected ≥ 2 Self calls when [DECISION] is missing (even with "
        f"[IDLE] present); got {len(self_llm.calls)}"
    )
    assert runtime.heartbeat_count == 1
    assert len(runtime.explicit_history.rounds) == 1


async def test_missing_thinking_still_retries_even_if_idle_present():
    """[DECISION]+[IDLE] present but [THINKING] missing → still retries.

    The THINKING requirement is unchanged by the [IDLE] waiver spec.
    Recovery: a second response with all three tags lets the beat complete.
    """
    response_no_thinking = (
        "[DECISION]\nNo action.\n"
        "[IDLE]\n1"
    )
    self_llm = ScriptedLLM([
        response_no_thinking,
        _good_response_with_idle(),
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm,
        hypo_llm=ScriptedLLM([]),
        modifiers=[],
    )
    _configure_fast_retries(runtime)

    await runtime.run(iterations=1)
    await runtime.close()

    assert len(self_llm.calls) >= 2, (
        f"Expected ≥ 2 Self calls when [THINKING] is missing (even with "
        f"[IDLE] present); got {len(self_llm.calls)}"
    )
    assert runtime.heartbeat_count == 1
    assert len(runtime.explicit_history.rounds) == 1


# ---------------------------------------------------------------------------
# Test 5 — Spec point 5:
#   NL "enter sleep mode" with NO [IDLE] and NO sleep <tool_call> → RETRIES.
#   This is explicitly intended — the waiver only fires on the tool call.
# ---------------------------------------------------------------------------

async def test_nl_only_sleep_text_without_tool_call_still_retries():
    """Natural-language 'enter sleep mode' without a sleep <tool_call> and
    without [IDLE] is NOT a valid response — must still retry.

    Expected:
    - Self called more than once (first response is invalid).
    - After recovery (second response with [IDLE]) the beat completes.
    - Sleep did NOT run (no tool call was present to trigger it).
    """
    self_llm = ScriptedLLM([
        _response_nl_sleep_no_idle(),   # malformed: no [IDLE], no tool
        _good_response_with_idle(),     # recovery
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm,
        hypo_llm=ScriptedLLM([]),
        modifiers=[],
    )
    _configure_fast_retries(runtime)

    await runtime.run(iterations=1)
    await runtime.close()

    # Retry fired (NL sleep hint did NOT grant a waiver).
    assert len(self_llm.calls) >= 2, (
        f"Expected ≥ 2 Self calls: NL 'enter sleep mode' without a "
        f"sleep <tool_call> must NOT waive [IDLE] requirement; "
        f"got {len(self_llm.calls)} call(s)"
    )

    # Sleep must NOT have run (no actual sleep tool call was present).
    assert runtime._sleep_cycles == 0, (
        f"Sleep must not run from NL text alone (no tool call); "
        f"got _sleep_cycles={runtime._sleep_cycles}"
    )

    assert runtime.heartbeat_count == 1
    assert len(runtime.explicit_history.rounds) == 1


# ---------------------------------------------------------------------------
# Test 6 — Spec point 6:
#   <tool_call> for a NON-sleep tool with no [IDLE] → still RETRIES.
#   The waiver is specific to the sleep tool name.
# ---------------------------------------------------------------------------

async def test_non_sleep_tool_call_does_not_waive_idle_requirement():
    """A <tool_call> block for a non-sleep tool with no [IDLE] must NOT
    grant the waiver — [IDLE] is still required.

    Expected:
    - Self called more than once (first response is invalid).
    - After recovery (second response with [IDLE]) the beat completes.
    """
    self_llm = ScriptedLLM([
        _response_other_tool_no_idle("web_chat_reply"),  # malformed
        _good_response_with_idle(),                       # recovery
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm,
        hypo_llm=ScriptedLLM([]),
        modifiers=["dashboard"],  # provides web_chat_reply so no tool-not-found noise
    )
    _configure_fast_retries(runtime)

    await runtime.run(iterations=1)
    await asyncio.sleep(0.05)  # let any dispatched tool task settle
    await runtime.close()

    # Retry fired — non-sleep tool does NOT waive [IDLE].
    assert len(self_llm.calls) >= 2, (
        f"Expected ≥ 2 Self calls: a non-sleep <tool_call> must NOT "
        f"waive the [IDLE] requirement; got {len(self_llm.calls)} call(s)"
    )

    assert runtime.heartbeat_count == 1
    assert len(runtime.explicit_history.rounds) == 1


# ---------------------------------------------------------------------------
# Additional: waiver also works when sleep tool appears after other tool calls
# (i.e. sleep is not the first tool_call block in the response)
# ---------------------------------------------------------------------------

async def test_sleep_tool_waiver_works_when_sleep_not_first_tool_call():
    """Sleep waiver fires even when the sleep <tool_call> is not the first
    block in Self's response — the waiver scans the whole raw response.

    Expected: no retry, sleep runs.
    """
    response_multi_tools_no_idle = (
        "[THINKING]\n(quiet beat)\n"
        "[DECISION]\nReply then sleep.\n"
        '<tool_call>{"name": "web_chat_reply", "arguments": {"text": "ok"}}</tool_call>\n'
        '<tool_call>{"name": "sleep"}</tool_call>\n'
        # deliberately NO [IDLE]
    )
    self_llm = ScriptedLLM([response_multi_tools_no_idle])
    sleep_llm = ScriptedLLM(["summary"] * 5)
    runtime = build_runtime_with_fakes(
        self_llm=self_llm,
        hypo_llm=ScriptedLLM([]),
        compact_llm=sleep_llm,
        modifiers=["dashboard"],  # provides web_chat_reply
    )
    _configure_fast_retries(runtime)

    await runtime.memory.initialize()
    await runtime.run(iterations=1)
    await asyncio.sleep(0.05)
    await runtime.close()

    # No retry.
    assert len(self_llm.calls) == 1, (
        f"Expected exactly 1 Self call (sleep waiver must apply even when "
        f"sleep <tool_call> is not the first block); got {len(self_llm.calls)}"
    )

    # Sleep ran.
    assert runtime._sleep_cycles == 1, (
        f"Expected _sleep_cycles == 1; got {runtime._sleep_cycles}"
    )


# ---------------------------------------------------------------------------
# Additional: the malformed-then-valid round is NOT persisted during retries
# (history round count stays 0 while retrying, only increments on success).
# Validates that the retry loop correctly withholds _phase_save_round.
# ---------------------------------------------------------------------------

async def test_malformed_response_round_not_persisted_during_retry():
    """Each malformed response (missing [IDLE]) must not create a history
    round.  Only the final valid response produces exactly one round.

    This guards against a regression where _phase_save_round is called
    on every struct-retry attempt rather than only after the loop exits
    with a valid response.
    """
    # Two malformed then one valid.
    self_llm = ScriptedLLM([
        _response_missing_idle(),
        _response_missing_idle(),
        _good_response_with_idle(),
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm,
        hypo_llm=ScriptedLLM([]),
        modifiers=[],
    )
    _configure_fast_retries(runtime)

    await runtime.run(iterations=1)
    await runtime.close()

    # Exactly 3 Self calls.
    assert len(self_llm.calls) == 3, (
        f"Expected 3 calls (2 malformed + 1 valid); got {len(self_llm.calls)}"
    )

    # Only one history round: the valid response.
    assert len(runtime.explicit_history.rounds) == 1, (
        f"Expected exactly 1 history round (malformed responses must not "
        f"be saved); got {len(runtime.explicit_history.rounds)}"
    )

    # Heartbeat count is 1 (frozen during retries).
    assert runtime.heartbeat_count == 1


# ---------------------------------------------------------------------------
# Additional: no-tag-marker fallback still retries (regression guard).
# A completely tag-free response has found_tags == frozenset() and lacks
# both THINKING and DECISION, so the existing retry must still fire.
# The new [IDLE] requirement must not break this pre-existing behavior.
# ---------------------------------------------------------------------------

async def test_completely_tagless_response_still_retries():
    """Tagless output (no [TAG] markers at all) still triggers the
    struct-output retry — existing behavior unchanged by the [IDLE] spec.

    Recovery via a valid second response.
    """
    tagless = "Unstructured rambling without any tags."
    self_llm = ScriptedLLM([
        tagless,
        _good_response_with_idle(),
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm,
        hypo_llm=ScriptedLLM([]),
        modifiers=[],
    )
    _configure_fast_retries(runtime)

    await runtime.run(iterations=1)
    await runtime.close()

    assert len(self_llm.calls) >= 2, (
        f"Expected ≥ 2 calls for a tagless response; got {len(self_llm.calls)}"
    )
    assert runtime.heartbeat_count == 1
    assert len(runtime.explicit_history.rounds) == 1


# ---------------------------------------------------------------------------
# Additional BVA: sleep tool with malformed JSON name field — must NOT waive
# (waiver only granted when parsed name == "sleep", not garbled content).
# ---------------------------------------------------------------------------

async def test_malformed_sleep_tool_json_does_not_waive_idle():
    """A <tool_call> block containing 'sleep' but with malformed JSON that
    cannot be cleanly parsed to confirm name == 'sleep' must NOT grant the
    waiver.  [IDLE] remains required.

    This guards against a naive substring-search implementation that would
    grant a waiver on any response containing the substring 'sleep'
    regardless of whether it is a syntactically correct tool call.
    """
    # The tool call block has 'sleep' in the text but the JSON is broken
    # (trailing junk), so a correct parser cannot confirm name == "sleep".
    response_broken_sleep_tool_no_idle = (
        "[THINKING]\n(quiet beat)\n"
        "[DECISION]\nTime to rest.\n"
        '<tool_call>{"name": "sleep"</tool_call>\n'  # missing closing brace
    )
    self_llm = ScriptedLLM([
        response_broken_sleep_tool_no_idle,
        _good_response_with_idle(),
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm,
        hypo_llm=ScriptedLLM([]),
        modifiers=[],
    )
    _configure_fast_retries(runtime)

    await runtime.run(iterations=1)
    await runtime.close()

    assert len(self_llm.calls) >= 2, (
        f"Expected ≥ 2 calls: a broken sleep <tool_call> JSON must NOT "
        f"waive [IDLE]; got {len(self_llm.calls)}"
    )
    assert runtime.heartbeat_count == 1
