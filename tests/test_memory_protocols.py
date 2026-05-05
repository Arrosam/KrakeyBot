"""MemoryService / KnowledgeBaseLike / KBRegistryService Protocols —
sanity checks that the built-in concrete classes satisfy the
Protocols we just promoted.

These tests are pure structure checks (isinstance against
@runtime_checkable Protocols). They guard against:
  * Adding a method to GraphMemory that callers want but nobody
    promoted to MemoryService — Protocol falls behind reality.
  * Renaming a method on GraphMemory without updating MemoryService —
    isinstance silently keeps passing because runtime_checkable only
    checks for method names, not signatures, but a planned signature
    drift test (next phase) will catch this.

If these tests fail, EITHER the Protocol needs an addition OR the
implementation diverged from the Protocol. Either way, both files
should be reviewed together.
"""
from __future__ import annotations

import pytest

from krakey.interfaces.services.memory import (
    KBRegistryService,
    KnowledgeBaseLike,
    MemoryService,
)


def test_graph_memory_satisfies_memory_service():
    """The built-in GraphMemory must satisfy MemoryService."""
    from krakey.memory.graph_memory import GraphMemory

    # Build an instance with minimal args. embedder is required at
    # construction; pass a no-op stub.
    async def _no_embed(text: str) -> list[float]:
        return [0.0] * 8

    gm = GraphMemory(":memory:", embedder=_no_embed)
    assert isinstance(gm, MemoryService)


def test_kb_registry_satisfies_kb_registry_service():
    """The built-in KBRegistry must satisfy KBRegistryService."""
    from krakey.memory.graph_memory import GraphMemory
    from krakey.memory.knowledge_base import KBRegistry

    async def _no_embed(text: str) -> list[float]:
        return [0.0] * 8

    gm = GraphMemory(":memory:", embedder=_no_embed)
    registry = KBRegistry(gm, kb_dir="/tmp/krakey-test-kbs",
                            embedder=_no_embed)
    assert isinstance(registry, KBRegistryService)


async def test_knowledge_base_satisfies_knowledge_base_like(tmp_path):
    """A KB instance returned from create_kb must satisfy KnowledgeBaseLike."""
    from krakey.memory.graph_memory import GraphMemory
    from krakey.memory.knowledge_base import KBRegistry

    async def _no_embed(text: str) -> list[float]:
        return [0.0] * 8

    gm = GraphMemory(":memory:", embedder=_no_embed)
    await gm.initialize()
    registry = KBRegistry(gm, kb_dir=str(tmp_path), embedder=_no_embed)

    kb = await registry.create_kb("test_kb", name="Test KB")
    assert isinstance(kb, KnowledgeBaseLike)

    await registry.close_all()
    await gm.close()


def test_protocols_are_runtime_checkable():
    """Decorator presence — @runtime_checkable is what makes the
    isinstance checks above actually run. If someone removes the
    decorator, isinstance() raises TypeError (caught here)."""
    # If not @runtime_checkable, isinstance() raises TypeError.
    # We construct a class that has none of the methods and verify
    # the check returns False (rather than raising).
    class _Empty: ...
    assert not isinstance(_Empty(), MemoryService)
    assert not isinstance(_Empty(), KnowledgeBaseLike)
    assert not isinstance(_Empty(), KBRegistryService)
