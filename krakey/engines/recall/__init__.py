"""``recall`` Engine — per-beat memory recall.

Default impl ``IncrementalRecallEngine`` (in ``default.py``) is a
factory wrapper around the ``IncrementalRecall`` driver class
(``incremental.py``). The Engine's ``new_session()`` returns a fresh
``IncrementalRecall`` instance each beat — per-beat state lives on
the session, not the Engine.

The ``RecallEngine`` Protocol the runtime depends on lives at
``krakey.interfaces.engines.recall``. The session shape
(``processed_stimuli`` + ``add_stimuli`` + ``finalize``) is exactly
what ``IncrementalRecall`` satisfies.
"""
from krakey.engines.recall.default import IncrementalRecallEngine
from krakey.engines.recall.incremental import IncrementalRecall

__all__ = ["IncrementalRecallEngine", "IncrementalRecall"]
