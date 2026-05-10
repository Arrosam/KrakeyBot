"""``explicit_history`` Engine — working-memory window.

Default impl ``SlidingWindowExplicitHistoryEngine`` (in ``default.py``)
is the existing ``krakey.engines.explicit_history.sliding_window.SlidingWindow``
exported under the new Engine name. The behavioral surface (append /
get_rounds / pop_oldest / total_tokens / needs_compact + the
``history_token_budget`` attr) is unchanged.

The slot is renamed from ``sliding_window`` to ``explicit_history``
because sliding-window-of-rounds is just one of several possible
working-memory strategies — future Engines could maintain a summary
tree, a relevance-scored LRU cache, a hierarchical recall buffer,
etc. The Protocol stays minimal so all those variants fit.

The ``ExplicitHistoryEngine`` Protocol the runtime depends on lives
at ``krakey.interfaces.engines.explicit_history``.
"""
from krakey.engines.explicit_history.default import (
    SlidingWindowExplicitHistoryEngine,
)

__all__ = ["SlidingWindowExplicitHistoryEngine"]
