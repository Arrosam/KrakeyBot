"""Built-in default Reflects.

These wrap the existing in-tree ``Hypothalamus`` and ``IncrementalRecall``
factory as default-registered Reflects. Behavior is unchanged from
pre-Reflect Krakey; the Reflect wrappers only exist so the registry
sees something to dispatch through.

User-toggleable Reflects (Reflect #1, #2, #3 from the design doc) will
land in sibling files here over time.
"""
from src.reflects.builtin.default_hypothalamus import (  # noqa: F401
    DefaultHypothalamusReflect,
)
from src.reflects.builtin.default_recall_anchor import (  # noqa: F401
    DefaultRecallAnchorReflect,
)
