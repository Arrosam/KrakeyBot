"""``ExplicitHistoryEngine`` — working-memory window management.

The Protocol captures the abstract surface the heartbeat depends on:

  * Append a round of (stimulus, decision, note) text.
  * Yield rounds for prompt assembly.
  * Pop the oldest round (the compactor evicts oldest into GM).
  * Report current token usage + whether compaction is needed.

The default impl ``SlidingWindowExplicitHistoryEngine`` is bounded by
``history_token_budget = self_role.max_input_tokens *
history_token_fraction`` and persists to a JSON file on every mutation.
A custom Engine could maintain a summary instead of raw rounds, score
rounds for retention rather than evict oldest, or back the window with
a remote store — the Protocol stays minimal so all those variants fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class ExplicitHistoryRound:
    """One past heartbeat as it appears in the [HISTORY] layer.

    The data shape is what crosses the boundary; the Engine's
    internal storage strategy is its own business.
    """
    heartbeat_id: int
    stimulus_summary: str
    decision_text: str
    note_text: str


@runtime_checkable
class ExplicitHistoryEngine(Protocol):
    """Bounded buffer of recent heartbeat rounds + token-budget bookkeeping.

    The heartbeat reads ``rounds`` (or calls ``get_rounds()``) for
    prompt assembly, ``append`` after each beat completes, and the
    compactor uses ``needs_compact`` + ``pop_oldest`` to evict overflow
    into long-term storage.

    ``history_token_budget`` is exposed read-only so the heartbeat can
    log + diagnose budget pressure. Engines compute it from the Self
    role's ``max_input_tokens * history_token_fraction`` at construction
    time; the runtime doesn't reset it post-construction.
    """

    history_token_budget: int

    def append(self, r: ExplicitHistoryRound) -> None: ...

    def get_rounds(self) -> list[ExplicitHistoryRound]: ...

    def pop_oldest(self) -> ExplicitHistoryRound | None: ...

    def total_tokens(self) -> int: ...

    def needs_compact(self) -> bool: ...
