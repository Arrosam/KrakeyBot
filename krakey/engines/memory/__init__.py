"""``memory`` Engine — unified GM CRUD + KB management + sleep cycle.

Default impl ``GraphMemoryEngine`` extends the bundled ``GraphMemory``
class (now at ``krakey.engines.memory._internal.graph_memory``) with
KB management methods (delegating to an internal ``KBRegistry``) and a
``sleep_cycle`` method (wrapping the ``enter_sleep_mode`` pipeline
under ``_internal/sleep/``).

The ``MemoryEngine`` Protocol that callers depend on lives at
``krakey.interfaces.engines.memory``. It's a single flat surface —
GM CRUD + KB management + sleep_cycle on one Engine — so the runtime
treats memory as one swappable component rather than three loosely
coupled services.

Everything under ``_internal/`` (``gm/``, ``knowledge_base/``,
``sleep/``, ``writer.py``, ``_db.py``, ``tools/``, ``schemas.sql``)
is private to this Engine. A user replacing the ``memory`` slot
can ignore those modules entirely and ship their own backend that
satisfies the ``MemoryEngine`` Protocol.
"""
from krakey.engines.catalog import EngineImpl
from krakey.engines.memory.default import GraphMemoryEngine

BUILTIN_ENGINES = {
    "graph_memory": EngineImpl(
        cls=GraphMemoryEngine,
        description=(
            "SQLite-backed graph memory + per-topic KB SQLite files "
            "+ enter_sleep_mode consolidation pipeline."
        ),
    ),
}

DEFAULT_ENGINE = "graph_memory"

__all__ = ["BUILTIN_ENGINES", "DEFAULT_ENGINE", "GraphMemoryEngine"]
