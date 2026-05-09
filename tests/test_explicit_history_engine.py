"""SlidingWindowExplicitHistoryEngine — Protocol conformance + delegation
to the inherited SlidingWindow.

The Engine is a subclass alias of SlidingWindow during the migration
window. Tests pin Protocol conformance + that the inherited methods
(append / get_rounds / pop_oldest / total_tokens / needs_compact) all
still work."""
from __future__ import annotations

from krakey.engines.explicit_history.default import (
    SlidingWindowExplicitHistoryEngine,
)
from krakey.interfaces.engines import ExplicitHistoryEngine
from krakey.runtime.heartbeat.sliding_window import (
    SlidingWindow,
    SlidingWindowRound,
)


def test_satisfies_explicit_history_engine_protocol():
    eng = SlidingWindowExplicitHistoryEngine(history_token_budget=1000)
    assert isinstance(eng, ExplicitHistoryEngine)


def test_subclasses_sliding_window():
    assert issubclass(SlidingWindowExplicitHistoryEngine, SlidingWindow)


def test_history_token_budget_attribute():
    """Protocol asks for a ``history_token_budget`` int attribute —
    SlidingWindow exposes it via __init__."""
    eng = SlidingWindowExplicitHistoryEngine(history_token_budget=5000)
    assert eng.history_token_budget == 5000


def test_append_get_rounds_round_trip():
    eng = SlidingWindowExplicitHistoryEngine(history_token_budget=10_000)
    r = SlidingWindowRound(
        heartbeat_id=1, stimulus_summary="s", decision_text="d",
        note_text="n",
    )
    eng.append(r)
    rounds = eng.get_rounds()
    assert len(rounds) == 1
    assert rounds[0].heartbeat_id == 1


def test_pop_oldest_returns_none_when_empty():
    eng = SlidingWindowExplicitHistoryEngine(history_token_budget=1000)
    assert eng.pop_oldest() is None


def test_pop_oldest_removes_first_appended():
    eng = SlidingWindowExplicitHistoryEngine(history_token_budget=10_000)
    eng.append(SlidingWindowRound(
        heartbeat_id=1, stimulus_summary="a",
        decision_text="", note_text="",
    ))
    eng.append(SlidingWindowRound(
        heartbeat_id=2, stimulus_summary="b",
        decision_text="", note_text="",
    ))
    popped = eng.pop_oldest()
    assert popped is not None and popped.heartbeat_id == 1
    rounds = eng.get_rounds()
    assert len(rounds) == 1 and rounds[0].heartbeat_id == 2


def test_needs_compact_reflects_token_total():
    """Tiny budget + a chunky round → needs_compact is True."""
    eng = SlidingWindowExplicitHistoryEngine(history_token_budget=1)
    eng.append(SlidingWindowRound(
        heartbeat_id=1,
        stimulus_summary="lots and lots of words to push tokens",
        decision_text="more text here", note_text="and more",
    ))
    assert eng.needs_compact() is True
