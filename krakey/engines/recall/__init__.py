"""``recall`` Engine — per-beat memory recall.

The slot's catalog of impls lives in ``meta.yaml`` next to this file.
The RecallEngine + RecallSession Protocols live at
``krakey.interfaces.engines.recall``.
"""
from krakey.engines.recall.default import IncrementalRecallEngine
from krakey.engines.recall.incremental import IncrementalRecall

__all__ = ["IncrementalRecallEngine", "IncrementalRecall"]
