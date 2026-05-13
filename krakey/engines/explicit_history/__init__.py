"""``explicit_history`` Engine — working-memory window.

The slot's catalog of impls lives in ``meta.yaml`` next to this file.
The ExplicitHistoryEngine Protocol lives at
``krakey.interfaces.engines.explicit_history``.
"""
from krakey.engines.explicit_history.default import (
    SlidingWindowExplicitHistoryEngine,
)

__all__ = ["SlidingWindowExplicitHistoryEngine"]
