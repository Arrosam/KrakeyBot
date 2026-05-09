"""``recall`` Engine — per-beat memory recall.

Default impl ``IncrementalRecallEngine`` (in ``default.py``) is a
factory wrapper around the long-standing ``IncrementalRecall`` class
that lives under ``krakey.plugins.recall.incremental``. The Engine's
``new_session()`` returns a fresh ``IncrementalRecall`` instance each
beat (the per-beat-state lifecycle stays the same as before).

The ``RecallEngine`` Protocol the runtime depends on lives at
``krakey.interfaces.engines.recall``. The session shape
(``processed_stimuli`` + ``add_stimuli`` + ``finalize``) is exactly
what ``IncrementalRecall`` already satisfies.

This is the second plugin → engine uplift (after hypothalamus).
The in-tree ``recall`` plugin stays in place during the migration
window so users with the modifier-role wiring keep working; step 12
+ test rewrites delete the plugin once the engine slot is the only
path.
"""
from krakey.engines.recall.default import IncrementalRecallEngine

__all__ = ["IncrementalRecallEngine"]
