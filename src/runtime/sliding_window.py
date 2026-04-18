"""Sliding window — dynamic token-bounded recent-heartbeat buffer
(DevSpec §10.1). Each round stores (stimulus_summary, decision, note).
"""
from __future__ import annotations

from dataclasses import dataclass


def _approx_tokens(text: str) -> int:
    """Rough estimate: 4 chars ≈ 1 token. Deterministic; avoids the
    flakiness of real tokenizers during Phase 1."""
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class SlidingWindowRound:
    heartbeat_id: int
    stimulus_summary: str
    decision_text: str
    note_text: str


class SlidingWindow:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.rounds: list[SlidingWindowRound] = []

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
            _approx_tokens(r.stimulus_summary)
            + _approx_tokens(r.decision_text)
            + _approx_tokens(r.note_text)
            for r in self.rounds
        )

    def needs_compact(self) -> bool:
        return self.total_tokens() > self.max_tokens
