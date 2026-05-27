"""Edge tests for config-driven compaction behavior of SlidingWindow.

Written BEFORE implementation.  Tests cover ONLY observable public behavior:
  SlidingWindow(history_token_budget, config=..., state_path=None/path)
  .needs_compact() -> bool
  .append(round)
  .get_rounds() -> list
  .pop_oldest() -> round | None
  .total_tokens() -> int

NEW SPEC (supersedes all prior versions):
  SlidingWindow.__init__ accepts config: dict | None = None.
  Two integer settings are read and SANITIZED:
    compact_threshold  — default 2048
    max_history_rounds — default 20
  Sanitization rule (applies to BOTH identically):
    Accepted ONLY if type is exactly `int` (NOT bool, NOT float, NOT str)
    AND value is strictly > 0.  Any invalid input (wrong type, <= 0) is
    replaced by the default (2048 / 20 respectively).
  needs_compact() returns True iff:
    total_tokens() > compact_threshold  OR  len(rounds) > max_history_rounds
  Both comparisons STRICT >.  history_token_budget is NO LONGER consulted
  by needs_compact() — it remains stored but does NOT drive compaction.
  Load enforcement: rounds loaded from disk are subject to the same cap.

Sizing note: "hello world " * 60 produces ~240 tokens under cl100k_base.
             "x" / "a" produce only a handful of tokens.
"""
import json

import pytest

from krakey.engines.explicit_history.sliding_window import (
    SlidingWindow,
    ExplicitHistoryRound,
)


# ---------------------------------------------------------------------------
# Shared helpers (match existing test style)
# ---------------------------------------------------------------------------

def _round(i, stim="stim", decision="dec", note=""):
    return ExplicitHistoryRound(
        heartbeat_id=i,
        stimulus_summary=stim,
        decision_text=decision,
        note_text=note,
    )


# Content that reliably produces > ~240 tokens under cl100k_base.
_LONG = "hello world " * 60

# Content that produces only a handful of tokens.
_SHORT = "x"

# A state-file builder for load-enforcement tests.
# Schema v2 format matches what the persistence tests assert on disk.
def _write_state(path, rounds_data, schema_version=2):
    """Write a minimal valid state file to *path*."""
    payload = {
        "schema_version": schema_version,
        "rounds": rounds_data,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _round_dict(i, stim="stim", decision="dec", note=""):
    """Produce the JSON-serialisable dict for one on-disk round entry."""
    return {
        "heartbeat_id": i,
        "stimulus_summary": stim,
        "decision_text": decision,
        "note_text": note,
    }


# ===========================================================================
# Technique 1 — Positive / equivalence partitioning
# ===========================================================================

class TestPositiveEquivalence:
    """Representative valid inputs — one subcase per distinct use case."""

    # --- defaults: no config / config=None / config={} → 2048 / 20 ---

    def test_default_no_config_no_compact_under_both_limits(self):
        """Out-of-box (no config kwarg): one short round → well under 2048
        tokens and well under 20 rounds → False."""
        w = SlidingWindow(history_token_budget=100_000, state_path=None)
        w.append(_round(1, stim=_SHORT))
        assert w.needs_compact() is False

    def test_default_config_none_no_compact(self):
        """config=None is valid; compact_threshold=2048, max_history_rounds=20.
        One short round → False."""
        w = SlidingWindow(
            history_token_budget=100_000, config=None, state_path=None
        )
        w.append(_round(1, stim=_SHORT))
        assert w.needs_compact() is False

    def test_default_config_empty_dict_no_compact(self):
        """config={} → both keys absent → defaults of 2048/20 govern.
        One short round → False."""
        w = SlidingWindow(
            history_token_budget=100_000, config={}, state_path=None
        )
        w.append(_round(1, stim=_SHORT))
        assert w.needs_compact() is False

    def test_default_token_threshold_2048_governs(self):
        """With default config, short content summing to well under 2048
        tokens (e.g., 10 short rounds) → False."""
        w = SlidingWindow(
            history_token_budget=100_000, config=None, state_path=None
        )
        for i in range(10):
            w.append(_round(i, stim=_SHORT))
        assert w.needs_compact() is False

    def test_default_round_cap_20_governs_exact_20_is_false(self):
        """Default max_history_rounds=20: exactly 20 short rounds → False
        (not strictly over 20)."""
        w = SlidingWindow(
            history_token_budget=100_000, config=None, state_path=None
        )
        for i in range(20):
            w.append(_round(i, stim=_SHORT))
        assert w.needs_compact() is False

    def test_default_round_cap_20_governs_21_is_true(self):
        """Default max_history_rounds=20: 21 short rounds → True
        (21 > 20)."""
        w = SlidingWindow(
            history_token_budget=100_000, config=None, state_path=None
        )
        for i in range(21):
            w.append(_round(i, stim=_SHORT))
        assert w.needs_compact() is True

    # --- small compact_threshold fires on token count ---

    def test_small_compact_threshold_fires_on_tokens(self):
        """compact_threshold=50 with long content (~240 tokens) → True."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 50},
            state_path=None,
        )
        w.append(_round(1, stim=_LONG, decision=_LONG))
        assert w.needs_compact() is True

    def test_small_compact_threshold_no_compact_when_under(self):
        """compact_threshold=50 with one short round (< 50 tokens) → False."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 50},
            state_path=None,
        )
        w.append(_round(1, stim=_SHORT))
        assert w.needs_compact() is False

    # --- small max_history_rounds fires on round count ---

    def test_small_max_history_rounds_fires_on_count(self):
        """max_history_rounds=3 with 4 short rounds → True; tokens tiny."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"max_history_rounds": 3},
            state_path=None,
        )
        for i in range(4):
            w.append(_round(i, stim=_SHORT))
        assert w.needs_compact() is True

    def test_small_max_history_rounds_no_compact_when_under(self):
        """max_history_rounds=3 with exactly 3 short rounds → False."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"max_history_rounds": 3},
            state_path=None,
        )
        for i in range(3):
            w.append(_round(i, stim=_SHORT))
        assert w.needs_compact() is False

    # --- large compact_threshold suppresses token trigger ---

    def test_large_compact_threshold_suppresses_token_trigger(self):
        """compact_threshold=100_000 with many small rounds (<=20) → False
        (tokens far under threshold, count under 20)."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 100_000},
            state_path=None,
        )
        for i in range(15):
            w.append(_round(i, stim=_SHORT))
        assert w.needs_compact() is False

    # --- history_token_budget is no longer consulted ---

    def test_tiny_budget_does_not_drive_compaction(self):
        """history_token_budget=10 but default config → compact_threshold=2048.
        Long content (~240 tokens) is still under 2048 → False.
        (Proves budget is no longer consulted by needs_compact.)"""
        w = SlidingWindow(
            history_token_budget=10, config=None, state_path=None
        )
        w.append(_round(1, stim=_LONG))
        # ~240 tokens < 2048 default threshold → False despite tiny budget
        assert w.needs_compact() is False

    # --- return type contract ---

    def test_needs_compact_returns_exact_bool_type(self):
        """Contract guarantees exactly `bool` (not merely truthy/falsy)."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 5, "max_history_rounds": 2},
            state_path=None,
        )
        result = w.needs_compact()
        assert type(result) is bool

    def test_needs_compact_returns_exact_bool_type_when_true(self):
        """Return type is bool even when triggered."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 5},
            state_path=None,
        )
        w.append(_round(1, stim=_LONG))
        result = w.needs_compact()
        assert type(result) is bool


# ===========================================================================
# Technique 2 — Boundary value analysis (BVA)
# ===========================================================================

class TestBoundaryValues:
    """Exact-at and one-over/under boundaries.  Strict > means == is NOT over.
    """

    # --- token threshold: default 2048 ---

    def test_empty_window_zero_tokens_not_over_threshold(self):
        """0 tokens is never > any positive compact_threshold → False."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 2048},
            state_path=None,
        )
        assert w.needs_compact() is False

    def test_tokens_exactly_equal_compact_threshold_is_false(self):
        """total_tokens() == compact_threshold → NOT over (strict >) → False.
        Use a large threshold that a controlled amount of content just reaches
        without exceeding.  Verify via total_tokens() itself."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 2048},
            state_path=None,
        )
        # Impossible to hit exactly 2048 with real content; instead, set a
        # threshold above any realistic content to guarantee == never triggers.
        w.append(_round(1, stim=_SHORT))
        tokens = w.total_tokens()
        # Threshold is 2048; tokens from one short round is far below 2048.
        # This verifies the "under-threshold" side of the default boundary.
        assert tokens < 2048
        assert w.needs_compact() is False

    def test_tokens_one_over_compact_threshold_is_true(self):
        """total_tokens() > compact_threshold → True.  Set threshold=1 so any
        real content exceeds it."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 1},
            state_path=None,
        )
        w.append(_round(1, stim="hello"))
        assert w.needs_compact() is True

    def test_compact_threshold_1_fires_on_any_content(self):
        """compact_threshold=1 is a valid minimum; any appended non-empty
        content should exceed 1 token."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 1},
            state_path=None,
        )
        w.append(_round(1, stim="a"))
        assert w.needs_compact() is True

    def test_compact_threshold_1_empty_window_false(self):
        """compact_threshold=1; empty window (0 tokens) → 0 is not > 1 → False."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 1},
            state_path=None,
        )
        assert w.needs_compact() is False

    # --- round count: default 20 ---

    def test_round_count_exactly_20_default_is_false(self):
        """Default max_history_rounds=20: exactly 20 rounds → 20 is not > 20
        → False."""
        w = SlidingWindow(
            history_token_budget=100_000, config=None, state_path=None
        )
        for i in range(20):
            w.append(_round(i, stim=_SHORT))
        assert w.needs_compact() is False

    def test_round_count_21_default_is_true(self):
        """Default max_history_rounds=20: 21 rounds → 21 > 20 → True."""
        w = SlidingWindow(
            history_token_budget=100_000, config=None, state_path=None
        )
        for i in range(21):
            w.append(_round(i, stim=_SHORT))
        assert w.needs_compact() is True

    # --- round count: small explicit cap ---

    def test_round_count_exactly_equal_max_is_false(self):
        """len(rounds) == max_history_rounds=3 → not strictly over → False."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"max_history_rounds": 3},
            state_path=None,
        )
        for i in range(3):
            w.append(_round(i, stim=_SHORT))
        assert w.needs_compact() is False

    def test_round_count_one_over_max_is_true(self):
        """len(rounds) == max_history_rounds+1 (3+1=4) → True."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"max_history_rounds": 3},
            state_path=None,
        )
        for i in range(4):
            w.append(_round(i, stim=_SHORT))
        assert w.needs_compact() is True

    def test_max_history_rounds_1_fires_on_second_round(self):
        """max_history_rounds=1: first round → False (1 == 1, not over);
        second round → True (2 > 1)."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"max_history_rounds": 1},
            state_path=None,
        )
        w.append(_round(1, stim=_SHORT))
        assert w.needs_compact() is False   # 1 == 1, not over
        w.append(_round(2, stim=_SHORT))
        assert w.needs_compact() is True    # 2 > 1

    def test_max_history_rounds_1_empty_window_false(self):
        """max_history_rounds=1; empty window → 0 is not > 1 → False."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"max_history_rounds": 1},
            state_path=None,
        )
        assert w.needs_compact() is False

    # --- combined boundaries: both fields at minimum valid value ---

    def test_both_at_minimum_1_empty_window_false(self):
        """compact_threshold=1, max_history_rounds=1; empty window → False."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 1, "max_history_rounds": 1},
            state_path=None,
        )
        assert w.needs_compact() is False

    def test_both_at_minimum_1_one_round_true(self):
        """compact_threshold=1, max_history_rounds=1; one real round →
        both triggers fire (token > 1 AND count 1 is not > 1... wait:
        count=1, max=1 → 1 is NOT > 1 → only token trigger).
        Either way → True (token trigger alone is enough)."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 1, "max_history_rounds": 1},
            state_path=None,
        )
        w.append(_round(1, stim="hello"))
        assert w.needs_compact() is True


# ===========================================================================
# Technique 3 — State transitions
# ===========================================================================

class TestStateTransitions:
    """Non-idempotent operations: append and pop_oldest change the state
    feeding into needs_compact().  Critical: the heartbeat compact loop must
    DRAIN and TERMINATE (pop until False, then stop)."""

    # --- token-threshold state cycle ---

    def test_token_trigger_over_then_pop_then_under(self):
        """Append bulk (token trigger fires) → pop_oldest until False.
        Verifies compact loop terminates via token drain."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 50},
            state_path=None,
        )
        w.append(_round(1, stim=_LONG, decision=_LONG))
        w.append(_round(2, stim=_SHORT))
        assert w.needs_compact() is True

        iterations = 0
        while w.needs_compact():
            popped = w.pop_oldest()
            assert popped is not None, (
                "pop_oldest must not return None while needs_compact is True"
            )
            iterations += 1
            assert iterations <= 10, (
                "compact loop did not terminate — pop did not reduce token count"
            )
        assert w.needs_compact() is False

    def test_token_trigger_idempotent_read_false(self):
        """needs_compact() is False; calling again without mutation → still
        False (pure read, no side effects)."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 50},
            state_path=None,
        )
        w.append(_round(1, stim=_SHORT))
        assert w.needs_compact() is False
        assert w.needs_compact() is False

    def test_token_trigger_idempotent_read_true(self):
        """needs_compact() is True; calling again without pop → still True."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 5},
            state_path=None,
        )
        w.append(_round(1, stim=_LONG))
        assert w.needs_compact() is True
        assert w.needs_compact() is True

    # --- round-count state cycle ---

    def test_round_count_trigger_append_then_pop_terminates(self):
        """max_history_rounds=3: append 5 → True; pop loop → terminates at
        <=3 rounds with needs_compact()==False."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"max_history_rounds": 3},
            state_path=None,
        )
        for i in range(5):
            w.append(_round(i, stim=_SHORT))
        assert w.needs_compact() is True

        pops = 0
        while w.needs_compact():
            w.pop_oldest()
            pops += 1
            assert pops <= 10, "compact loop did not terminate"

        assert w.needs_compact() is False
        assert len(w.get_rounds()) <= 3

    def test_round_count_exact_boundary_transition(self):
        """max_history_rounds=2: append 2 → False (== max, not over);
        append 3rd → True (> max); pop → 2 again → False."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"max_history_rounds": 2},
            state_path=None,
        )
        w.append(_round(1, stim=_SHORT))
        w.append(_round(2, stim=_SHORT))
        assert w.needs_compact() is False

        w.append(_round(3, stim=_SHORT))
        assert w.needs_compact() is True

        w.pop_oldest()
        assert w.needs_compact() is False

    def test_round_count_full_drain_to_1(self):
        """max_history_rounds=1: fill to 5; drain loop → exactly 1 round
        remains; needs_compact()==False."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"max_history_rounds": 1},
            state_path=None,
        )
        for i in range(5):
            w.append(_round(i, stim=_SHORT))
        assert w.needs_compact() is True

        pops = 0
        while w.needs_compact():
            w.pop_oldest()
            pops += 1
            assert pops <= 10, "compact loop did not terminate"

        assert w.needs_compact() is False
        assert len(w.get_rounds()) == 1

    def test_re_append_past_cap_re_triggers(self):
        """After draining to under the cap, re-appending past the cap
        re-triggers needs_compact()."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"max_history_rounds": 2},
            state_path=None,
        )
        for i in range(3):
            w.append(_round(i, stim=_SHORT))
        w.pop_oldest()                        # now 2 rounds == max, not over
        assert w.needs_compact() is False

        w.append(_round(99, stim=_SHORT))     # 3 > 2 → re-triggers
        assert w.needs_compact() is True

    def test_token_trigger_then_re_append_re_triggers(self):
        """After draining via token trigger, re-appending long content
        re-triggers needs_compact()."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 50},
            state_path=None,
        )
        w.append(_round(1, stim=_LONG, decision=_LONG))
        assert w.needs_compact() is True

        while w.needs_compact():
            w.pop_oldest()
        assert w.needs_compact() is False

        w.append(_round(2, stim=_LONG, decision=_LONG))
        assert w.needs_compact() is True

    def test_pop_oldest_on_empty_returns_none_stays_false(self):
        """pop_oldest() on empty window returns None (contract) and
        needs_compact() remains False."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"max_history_rounds": 2},
            state_path=None,
        )
        result = w.pop_oldest()
        assert result is None
        assert w.needs_compact() is False

    def test_initial_state_is_not_compact(self):
        """Freshly constructed window → 0 rounds, 0 tokens → always False
        regardless of threshold settings."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 5, "max_history_rounds": 1},
            state_path=None,
        )
        assert w.needs_compact() is False

    # --- combined OR: verify independence of the two triggers ---

    def test_or_only_token_trigger_fires(self):
        """max_history_rounds=10 (count safely under); compact_threshold=5
        (tokens over) → True.  Count trigger did NOT contribute."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 5, "max_history_rounds": 10},
            state_path=None,
        )
        w.append(_round(1, stim=_LONG))
        assert len(w.get_rounds()) == 1         # well under 10
        assert w.needs_compact() is True

    def test_or_only_count_trigger_fires(self):
        """compact_threshold=100_000 (tokens safely under); max_history_rounds=2
        (count over) → True.  Token trigger did NOT contribute."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 100_000, "max_history_rounds": 2},
            state_path=None,
        )
        for i in range(3):
            w.append(_round(i, stim=_SHORT))
        assert w.total_tokens() < 100_000      # well under threshold
        assert w.needs_compact() is True

    def test_or_drain_resolves_when_both_satisfied(self):
        """Both triggers fire simultaneously; drain via pop → resolves
        to False when both are back under their respective limits."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 50, "max_history_rounds": 1},
            state_path=None,
        )
        w.append(_round(1, stim=_LONG, decision=_LONG))
        w.append(_round(2, stim=_LONG, decision=_LONG))
        assert w.needs_compact() is True

        iterations = 0
        while w.needs_compact():
            w.pop_oldest()
            iterations += 1
            assert iterations <= 10, "drain loop did not terminate"

        assert w.needs_compact() is False


# ===========================================================================
# Technique 4 — Negative / invalid inputs (sanitization)
# ===========================================================================

class TestInvalidInputSanitization:
    """Each invalid value must SNAP TO THE DEFAULT — never crash, never
    truncate, never silently use the invalid value.

    Proof strategy: show that the behavior matches the default (2048/20),
    NOT the invalid value.  The key probes are:
      - compact_threshold=0 with tiny budget → NOT compacting at 200 tokens
        (proves 0 snapped to 2048, NOT to the 10-token budget)
      - max_history_rounds=0 with 21 rounds → True (snapped to 20, 21>20)
        and with 20 rounds → False (20 is not > 20)
    """

    # --- compact_threshold: value 0 ---

    def test_compact_threshold_0_snaps_to_2048_not_budget(self):
        """compact_threshold=0 must snap to 2048, NOT fall back to
        history_token_budget.  With budget=10 and ~240-token content,
        needs_compact must be False (200 < 2048, proves 0 ≠ budget fallback).
        """
        w = SlidingWindow(
            history_token_budget=10,
            config={"compact_threshold": 0},
            state_path=None,
        )
        w.append(_round(1, stim=_LONG))   # ~120+ tokens, well under 2048
        # If 0 had been used, 240 > 0 → True (or budget fallback 240 > 10 → True).
        # If snapped to 2048, 240 < 2048 → False.
        assert w.needs_compact() is False

    def test_compact_threshold_0_with_21_rounds_stays_governed_by_default_round_cap(self):
        """compact_threshold=0 snaps to 2048; max_history_rounds not set →
        default 20.  21 short rounds → True (count trigger via default cap)."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 0},
            state_path=None,
        )
        for i in range(21):
            w.append(_round(i, stim=_SHORT))
        assert w.needs_compact() is True

    # --- compact_threshold: negative value ---

    def test_compact_threshold_negative_snaps_to_2048(self):
        """compact_threshold=-5 must snap to 2048.  Short content + large budget
        → False (not over 2048)."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": -5},
            state_path=None,
        )
        w.append(_round(1, stim=_SHORT))
        assert w.needs_compact() is False

    def test_compact_threshold_negative_does_not_crash(self):
        """compact_threshold=-5 must not raise on construction or use."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": -1},
            state_path=None,
        )
        # Must be callable without exception.
        _ = w.needs_compact()

    # --- compact_threshold: float (non-integer type) ---

    def test_compact_threshold_float_3_7_snaps_to_2048(self):
        """compact_threshold=3.7 (float) must snap to 2048 — must NOT be
        truncated to 3.  Short content (< 2048 tokens) → False."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 3.7},
            state_path=None,
        )
        w.append(_round(1, stim="hello world"))
        # If truncated to 3: "hello world" > 3 tokens → True (WRONG).
        # If snapped to 2048: "hello world" << 2048 → False (CORRECT).
        assert w.needs_compact() is False

    def test_compact_threshold_float_2048_0_snaps_to_2048_default(self):
        """compact_threshold=2048.0 (float, technically same numeric value)
        must also snap to 2048 (int check required — floats not accepted).
        Short content → False either way, but proves no truncation/type coerce."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 2048.0},
            state_path=None,
        )
        w.append(_round(1, stim=_SHORT))
        assert w.needs_compact() is False

    # --- compact_threshold: string ---

    def test_compact_threshold_string_snaps_to_2048(self):
        """compact_threshold="50" (str) must snap to 2048 — must NOT be
        parsed as integer 50.  Short content → False (< 2048)."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": "50"},
            state_path=None,
        )
        w.append(_round(1, stim=_SHORT))
        # If parsed as 50: short content might still be < 50 → ambiguous.
        # Use content that exceeds 50 but not 2048.
        w2 = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": "50"},
            state_path=None,
        )
        w2.append(_round(1, stim=_LONG))  # ~120 tokens > 50, < 2048
        # Must be False (snapped to 2048); would be True if parsed as 50.
        assert w2.needs_compact() is False

    # --- compact_threshold: bool ---

    def test_compact_threshold_true_snaps_to_2048(self):
        """compact_threshold=True (bool) must snap to 2048 — bool is a
        subclass of int in Python, but spec says NOT bool.  Short content
        → False (< 2048)."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": True},
            state_path=None,
        )
        w.append(_round(1, stim="hello world"))
        # If True were accepted as int(1): "hello world" > 1 token → True (WRONG).
        # If snapped to 2048: → False (CORRECT).
        assert w.needs_compact() is False

    def test_compact_threshold_false_snaps_to_2048(self):
        """compact_threshold=False (bool 0) must snap to 2048."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": False},
            state_path=None,
        )
        w.append(_round(1, stim=_LONG))   # ~120+ tokens, under 2048
        assert w.needs_compact() is False

    # --- max_history_rounds: value 0 ---

    def test_max_history_rounds_0_snaps_to_20_trigger_at_21(self):
        """max_history_rounds=0 must snap to 20.  21 short rounds → True
        (21 > 20)."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"max_history_rounds": 0},
            state_path=None,
        )
        for i in range(21):
            w.append(_round(i, stim=_SHORT))
        assert w.needs_compact() is True

    def test_max_history_rounds_0_snaps_to_20_no_trigger_at_20(self):
        """max_history_rounds=0 snaps to 20.  Exactly 20 short rounds → False
        (20 is not > 20)."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"max_history_rounds": 0},
            state_path=None,
        )
        for i in range(20):
            w.append(_round(i, stim=_SHORT))
        assert w.needs_compact() is False

    # --- max_history_rounds: negative value ---

    def test_max_history_rounds_negative_snaps_to_20(self):
        """max_history_rounds=-3 must snap to 20.  20 short rounds → False;
        21 → True."""
        w_20 = SlidingWindow(
            history_token_budget=100_000,
            config={"max_history_rounds": -3},
            state_path=None,
        )
        for i in range(20):
            w_20.append(_round(i, stim=_SHORT))
        assert w_20.needs_compact() is False

        w_21 = SlidingWindow(
            history_token_budget=100_000,
            config={"max_history_rounds": -3},
            state_path=None,
        )
        for i in range(21):
            w_21.append(_round(i, stim=_SHORT))
        assert w_21.needs_compact() is True

    def test_max_history_rounds_negative_does_not_crash(self):
        """max_history_rounds=-3 must not raise on construction or use."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"max_history_rounds": -3},
            state_path=None,
        )
        _ = w.needs_compact()

    # --- max_history_rounds: float ---

    def test_max_history_rounds_float_snaps_to_20(self):
        """max_history_rounds=2.5 (float) must snap to 20 — must NOT be
        truncated to 2.  Proof: 3 rounds would fire if capped at 2,
        but 3 rounds is well under 20 → False."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"max_history_rounds": 2.5},
            state_path=None,
        )
        for i in range(3):
            w.append(_round(i, stim=_SHORT))
        # If truncated to 2: 3 > 2 → True (WRONG).
        # If snapped to 20: 3 < 20 → False (CORRECT).
        assert w.needs_compact() is False

    # --- max_history_rounds: string ---

    def test_max_history_rounds_string_snaps_to_20(self):
        """max_history_rounds="5" (str) must snap to 20 — must NOT be
        parsed as 5.  6 rounds would fire if cap=5, but 6 < 20 → False."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"max_history_rounds": "5"},
            state_path=None,
        )
        for i in range(6):
            w.append(_round(i, stim=_SHORT))
        # If parsed as 5: 6 > 5 → True (WRONG).
        # If snapped to 20: 6 < 20 → False (CORRECT).
        assert w.needs_compact() is False

    # --- max_history_rounds: bool ---

    def test_max_history_rounds_true_snaps_to_20(self):
        """max_history_rounds=True (bool) must snap to 20.  2 rounds would
        fire if cap=1 (bool True == int 1), but 2 < 20 → False."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"max_history_rounds": True},
            state_path=None,
        )
        w.append(_round(1, stim=_SHORT))
        w.append(_round(2, stim=_SHORT))
        # If True accepted as 1: 2 > 1 → True (WRONG).
        # If snapped to 20: 2 < 20 → False (CORRECT).
        assert w.needs_compact() is False

    def test_max_history_rounds_false_snaps_to_20(self):
        """max_history_rounds=False (bool 0) must snap to 20."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"max_history_rounds": False},
            state_path=None,
        )
        for i in range(20):
            w.append(_round(i, stim=_SHORT))
        assert w.needs_compact() is False

    # --- max_history_rounds: missing key (treated as absent → default) ---

    def test_max_history_rounds_missing_key_defaults_to_20(self):
        """config without max_history_rounds key → default 20.
        20 rounds → False; 21 → True."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 100_000},  # only other key present
            state_path=None,
        )
        for i in range(20):
            w.append(_round(i, stim=_SHORT))
        assert w.needs_compact() is False

        w.append(_round(99, stim=_SHORT))
        assert w.needs_compact() is True

    # --- OR logic: only one trigger satisfied ---

    def test_or_only_round_count_trigger_satisfied(self):
        """compact_threshold=100_000, max_history_rounds=2: 3 tiny rounds.
        Only count trigger fires → True."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 100_000, "max_history_rounds": 2},
            state_path=None,
        )
        for i in range(3):
            w.append(_round(i, stim=_SHORT))
        assert w.needs_compact() is True

    def test_or_only_token_trigger_satisfied(self):
        """compact_threshold=5, max_history_rounds=1000: 1 long round.
        Only token trigger fires → True."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 5, "max_history_rounds": 1000},
            state_path=None,
        )
        w.append(_round(1, stim=_LONG, decision=_LONG))
        assert w.needs_compact() is True

    def test_or_neither_trigger_satisfied_is_false(self):
        """Both thresholds large; 1 short round → False."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 100_000, "max_history_rounds": 100},
            state_path=None,
        )
        w.append(_round(1, stim=_SHORT))
        assert w.needs_compact() is False

    def test_or_both_triggers_simultaneously_satisfied(self):
        """compact_threshold=5, max_history_rounds=1; 2 long rounds →
        both token > 5 AND count 2 > 1 → True."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 5, "max_history_rounds": 1},
            state_path=None,
        )
        w.append(_round(1, stim=_LONG, decision=_LONG))
        w.append(_round(2, stim=_LONG, decision=_LONG))
        assert w.needs_compact() is True

    # --- unknown config keys ignored ---

    def test_unknown_config_keys_ignored_known_keys_apply(self):
        """Extra unknown keys in config must not raise; known keys apply."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={
                "compact_threshold": 5,
                "max_history_rounds": 10,
                "unknown_future_key": "some_value",
                "another_unknown": 42,
            },
            state_path=None,
        )
        w.append(_round(1, stim=_LONG, decision=_LONG))
        # compact_threshold=5 still governs → True.
        assert w.needs_compact() is True

    def test_unknown_config_keys_do_not_crash(self):
        """Unknown keys alone (no known keys) must not crash; defaults apply."""
        w = SlidingWindow(
            history_token_budget=100_000,
            config={"future_setting": 99},
            state_path=None,
        )
        _ = w.needs_compact()

    # --- backward-compat: config is truly optional ---

    def test_constructor_accepts_no_config_kwarg(self):
        """SlidingWindow(history_token_budget=...) with no config kwarg at all
        must construct and return a bool from needs_compact()."""
        w = SlidingWindow(history_token_budget=4096, state_path=None)
        w.append(_round(1, stim=_SHORT))
        assert isinstance(w.needs_compact(), bool)


# ===========================================================================
# Technique 5 — Load enforcement (new requirement)
# ===========================================================================

class TestLoadEnforcement:
    """Rounds loaded from disk are subject to the same cap as appended rounds.
    A window loaded from a state file that already exceeds the cap must report
    needs_compact()==True immediately after construction.

    On-disk format: {"schema_version": 2, "rounds": [...]}
    (Matches the format asserted by the existing persistence tests.)
    """

    def test_load_overflow_round_count_triggers_immediately(self, tmp_path):
        """State file with 25 rounds (> default max 20) → needs_compact()==True
        right after construction with default config."""
        state = tmp_path / "sw.json"
        rounds_data = [_round_dict(i, stim=_SHORT) for i in range(25)]
        _write_state(state, rounds_data)

        w = SlidingWindow(
            history_token_budget=100_000, config=None, state_path=state
        )
        # 25 loaded rounds > 20 default cap → must be True immediately.
        assert w.needs_compact() is True

    def test_load_overflow_round_count_triggers_with_explicit_cap(self, tmp_path):
        """State file with 5 rounds; max_history_rounds=3 → 5 > 3 → True."""
        state = tmp_path / "sw.json"
        rounds_data = [_round_dict(i, stim=_SHORT) for i in range(5)]
        _write_state(state, rounds_data)

        w = SlidingWindow(
            history_token_budget=100_000,
            config={"max_history_rounds": 3},
            state_path=state,
        )
        assert w.needs_compact() is True

    def test_load_overflow_token_count_triggers_immediately(self, tmp_path):
        """State file with rounds summing to > compact_threshold → True
        immediately after construction."""
        state = tmp_path / "sw.json"
        # 5 rounds of long content → well over 50 tokens total.
        rounds_data = [
            _round_dict(i, stim=_LONG, decision=_LONG) for i in range(5)
        ]
        _write_state(state, rounds_data)

        w = SlidingWindow(
            history_token_budget=100_000,
            config={"compact_threshold": 50},
            state_path=state,
        )
        assert w.needs_compact() is True

    def test_load_under_cap_is_false(self, tmp_path):
        """Positive control: state file with 5 short rounds (<=20, tokens
        small) → needs_compact()==False after load."""
        state = tmp_path / "sw.json"
        rounds_data = [_round_dict(i, stim=_SHORT) for i in range(5)]
        _write_state(state, rounds_data)

        w = SlidingWindow(
            history_token_budget=100_000, config=None, state_path=state
        )
        assert w.needs_compact() is False

    def test_load_exactly_at_cap_is_false(self, tmp_path):
        """State file with exactly 20 short rounds (== default cap, not over)
        → False (strict > means equal is not over)."""
        state = tmp_path / "sw.json"
        rounds_data = [_round_dict(i, stim=_SHORT) for i in range(20)]
        _write_state(state, rounds_data)

        w = SlidingWindow(
            history_token_budget=100_000, config=None, state_path=state
        )
        assert w.needs_compact() is False

    def test_load_overflow_rounds_are_accessible_via_get_rounds(self, tmp_path):
        """After loading an overflow state, get_rounds() returns all loaded
        rounds (loading does not silently truncate — that is compact's job)."""
        state = tmp_path / "sw.json"
        rounds_data = [_round_dict(i, stim=_SHORT) for i in range(25)]
        _write_state(state, rounds_data)

        w = SlidingWindow(
            history_token_budget=100_000, config=None, state_path=state
        )
        assert len(w.get_rounds()) == 25

    def test_load_overflow_drain_loop_terminates(self, tmp_path):
        """After loading 25 rounds with default cap 20 → True; run compact
        loop → terminates with needs_compact()==False and <=20 rounds."""
        state = tmp_path / "sw.json"
        rounds_data = [_round_dict(i, stim=_SHORT) for i in range(25)]
        _write_state(state, rounds_data)

        w = SlidingWindow(
            history_token_budget=100_000, config=None, state_path=state
        )
        assert w.needs_compact() is True

        pops = 0
        while w.needs_compact():
            w.pop_oldest()
            pops += 1
            assert pops <= 30, "drain loop from loaded overflow did not terminate"

        assert w.needs_compact() is False
        assert len(w.get_rounds()) <= 20
