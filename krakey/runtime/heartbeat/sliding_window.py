"""Sliding window — dynamic token-bounded recent-heartbeat buffer
(DevSpec §10.1). Each round stores (stimulus_summary, decision, note).

Window size (the threshold that triggers compaction) is no longer
set by a standalone config key. It's derived at runtime from the
Self role's LLMParams::

    history_budget = max_input_tokens * history_token_fraction

The runtime owner (main.Runtime) computes that at startup and passes
it into ``SlidingWindow(history_token_budget=...)``. Compaction code
calls ``needs_compact()`` which compares live token count against
that budget.

Token estimation goes through ``src.utils.tokens.estimate_tokens``
(tiktoken cl100k_base) — replaces the previous char/4 heuristic which
undercounted Chinese text ~4-8×.
"""
from __future__ import annotations

from dataclasses import dataclass

from krakey.utils.tokens import estimate_tokens


@dataclass
class SlidingWindowRound:
    heartbeat_id: int
    stimulus_summary: str
    decision_text: str
    note_text: str


class SlidingWindow:
    """Bounded buffer of recent heartbeat rounds.

    ``history_token_budget`` is the threshold at which compaction
    fires. When ``needs_compact()`` is True the compactor loop pops
    the oldest round and asks the compact LLM to extract GM nodes.

    ``max_tokens`` is kept as a back-compat alias for callers still
    passing the old keyword; new code should use
    ``history_token_budget`` explicitly so budget-math reads the same
    on both sides.
    """

    def __init__(
        self,
        history_token_budget: int | None = None,
        *,
        max_tokens: int | None = None,
    ):
        if history_token_budget is None and max_tokens is None:
            raise ValueError(
                "SlidingWindow needs history_token_budget "
                "(from self_role.max_input_tokens * "
                "history_token_fraction)"
            )
        # Either-or for the deprecation window; history_token_budget
        # wins if both are passed (new name is authoritative).
        self.history_token_budget: int = (
            history_token_budget
            if history_token_budget is not None else int(max_tokens or 0)
        )
        self.rounds: list[SlidingWindowRound] = []

    # Legacy accessor for tests / logging that used `max_tokens`.
    @property
    def max_tokens(self) -> int:
        return self.history_token_budget

    def append(self, r: SlidingWindowRound) -> None:
        self.rounds.append(r)

    def get_rounds(self) -> list[SlidingWindowRound]:
        return list(self.rounds)

    def pop_oldest(self) -> SlidingWindowRound | None:
        if not self.rounds:
            return None
        return self.rounds.pop(0)

    def total_tokens(self) -> int:
        return sum(
            estimate_tokens(r.stimulus_summary)
            + estimate_tokens(r.decision_text)
            + estimate_tokens(r.note_text)
            for r in self.rounds
        )

    def needs_compact(self) -> bool:
        return self.total_tokens() > self.history_token_budget
