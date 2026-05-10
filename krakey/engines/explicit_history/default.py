"""``SlidingWindowExplicitHistoryEngine`` — default ExplicitHistoryEngine.

Alias for the long-standing ``SlidingWindow`` class; that class
already implements every method the new ``ExplicitHistoryEngine``
Protocol asks for plus the ``history_token_budget`` read-only attr.
Subclassing rather than re-exporting so the Engine class has its
own identity for ``isinstance`` checks in tests.

The constructor signature stays the SlidingWindow one — the
EngineRegistry-supplied kwargs (``history_token_budget``,
``state_path``) are exactly what SlidingWindow expects.
"""
from __future__ import annotations

from krakey.engines.explicit_history.sliding_window import SlidingWindow


class SlidingWindowExplicitHistoryEngine(SlidingWindow):
    """Default ExplicitHistoryEngine — sliding-window-of-rounds
    backed by an atomic JSON mirror at ``state_path``."""
    pass
