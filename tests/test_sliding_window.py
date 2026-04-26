"""Phase 1.4: SlidingWindow (dynamic token-based window)."""
import pytest

from src.runtime.sliding_window import SlidingWindow, SlidingWindowRound


def _round(i, stim="stim", decision="dec", note=""):
    return SlidingWindowRound(heartbeat_id=i, stimulus_summary=stim,
                                decision_text=decision, note_text=note)


def test_append_and_get_rounds():
    w = SlidingWindow(max_tokens=4096)
    w.append(_round(1))
    w.append(_round(2))
    assert [r.heartbeat_id for r in w.get_rounds()] == [1, 2]


def test_needs_compact_false_when_under_limit():
    w = SlidingWindow(max_tokens=10000)
    w.append(_round(1, stim="short"))
    assert w.needs_compact() is False


def test_needs_compact_true_when_over_limit():
    # tight max_tokens + long content
    w = SlidingWindow(max_tokens=10)
    w.append(_round(1, stim="a" * 200, decision="b" * 200))
    assert w.needs_compact() is True


def test_pop_oldest_returns_and_removes():
    w = SlidingWindow(max_tokens=4096)
    w.append(_round(1))
    w.append(_round(2))
    popped = w.pop_oldest()
    assert popped.heartbeat_id == 1
    assert [r.heartbeat_id for r in w.get_rounds()] == [2]


def test_pop_oldest_on_empty_returns_none():
    w = SlidingWindow(max_tokens=4096)
    assert w.pop_oldest() is None


def test_needs_compact_clears_after_popping():
    # Use content long enough to blow past the 50-token cap under the
    # real (tiktoken cl100k_base) estimator. Previously-relied-upon
    # char/4 heuristic undercounted dramatically, so the small round
    # needed bigger payloads than expected.
    w = SlidingWindow(max_tokens=50)
    w.append(_round(1, stim="hello world " * 60,
                      decision="goodbye world " * 60))
    w.append(_round(2, stim="c"))
    assert w.needs_compact() is True
    w.pop_oldest()
    assert w.needs_compact() is False


def test_total_tokens_approximation_scales_with_content():
    w = SlidingWindow(max_tokens=4096)
    small = SlidingWindow(max_tokens=4096)
    w.append(_round(1, stim="x" * 400))
    small.append(_round(1, stim="x" * 40))
    assert w.total_tokens() > small.total_tokens()
