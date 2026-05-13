"""``memory`` Engine — unified GM CRUD + KB management + sleep cycle.

The slot's catalog of impls lives in ``meta.yaml`` next to this file.
Everything under ``_internal/`` is private to the default impl;
users replacing the slot ignore those modules entirely and satisfy
the MemoryEngine Protocol declared at
``krakey.interfaces.engines.memory``.
"""
from krakey.engines.memory.default import GraphMemoryEngine

__all__ = ["GraphMemoryEngine"]
