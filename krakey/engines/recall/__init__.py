"""``recall`` Engine — per-beat memory recall.

Default impl ``IncrementalRecallEngine`` is a factory wrapper around
the ``IncrementalRecall`` driver class (``incremental.py``). The
Engine's ``new_session()`` returns a fresh ``IncrementalRecall``
instance each beat — per-beat state lives on the session, not the
Engine.

The ``RecallEngine`` Protocol the runtime depends on lives at
``krakey.interfaces.engines.recall``. The session shape
(``processed_stimuli`` + ``add_stimuli`` + ``finalize``) is exactly
what ``IncrementalRecall`` satisfies.
"""
from krakey.engines.catalog import EngineImpl
from krakey.engines.recall.default import IncrementalRecallEngine
from krakey.engines.recall.incremental import IncrementalRecall

BUILTIN_ENGINES = {
    "incremental": EngineImpl(
        cls=IncrementalRecallEngine,
        description=(
            "Per-stimulus vec_search → rerank → token-budget cut + "
            "covered/uncovered partition for the next beat."
        ),
    ),
}

DEFAULT_ENGINE = "incremental"

__all__ = [
    "BUILTIN_ENGINES",
    "DEFAULT_ENGINE",
    "IncrementalRecallEngine",
    "IncrementalRecall",
]
