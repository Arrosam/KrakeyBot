"""``explicit_history`` Engine — working-memory window.

Default impl ``SlidingWindowExplicitHistoryEngine`` is a token-bounded
ring of recent heartbeat rounds (the canonical [HISTORY] layer
source). Sliding-window-of-rounds is one possible strategy; future
Engines could maintain a summary tree, a relevance-scored LRU cache,
a hierarchical recall buffer, etc. The Protocol stays minimal so all
those variants fit.

The ``ExplicitHistoryEngine`` Protocol the runtime depends on lives
at ``krakey.interfaces.engines.explicit_history``.
"""
from krakey.engines.catalog import EngineImpl
from krakey.engines.explicit_history.default import (
    SlidingWindowExplicitHistoryEngine,
)

BUILTIN_ENGINES = {
    "sliding_window": EngineImpl(
        cls=SlidingWindowExplicitHistoryEngine,
        description=(
            "Token-bounded ring of recent heartbeat rounds; oldest "
            "rounds compact into GM when the budget overflows."
        ),
    ),
}

DEFAULT_ENGINE = "sliding_window"

__all__ = [
    "BUILTIN_ENGINES",
    "DEFAULT_ENGINE",
    "SlidingWindowExplicitHistoryEngine",
]
