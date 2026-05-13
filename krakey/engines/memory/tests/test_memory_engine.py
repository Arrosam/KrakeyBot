"""GraphMemoryEngine — Protocol conformance + KB delegation +
inheritance from GraphMemory + sleep_cycle wiring.

Step 5a: the Engine class exists but isn't yet wired through the
runtime composition root (that's step 5b). Tests here prove the
class is a valid MemoryEngine impl in isolation.
"""
from __future__ import annotations

import pytest

from krakey.engines.memory.default import GraphMemoryEngine
from krakey.interfaces.engines import KnowledgeBaseLike, MemoryEngine


async def _no_embed(text: str) -> list[float]:
    return [0.0] * 8


# --------------------------------------------------------------------
# Protocol conformance
# --------------------------------------------------------------------


def test_satisfies_memory_engine_protocol(tmp_path):
    """The Engine must satisfy the new MemoryEngine Protocol — every
    method (GM CRUD + KB management + sleep_cycle) is reachable."""
    eng = GraphMemoryEngine(
        db_path=":memory:", embedder=_no_embed,
        kb_dir=str(tmp_path),
    )
    assert isinstance(eng, MemoryEngine)


# --------------------------------------------------------------------
# Lifecycle — initialize builds KB registry; close tears it down
# --------------------------------------------------------------------


async def test_initialize_builds_kb_registry(tmp_path):
    eng = GraphMemoryEngine(
        db_path=":memory:", embedder=_no_embed,
        kb_dir=str(tmp_path),
    )
    # Pre-init: KB methods raise (better signal than NoneType errors).
    with pytest.raises(RuntimeError, match="initialize"):
        await eng.create_kb("x", name="X")

    await eng.initialize()
    # Post-init: KB methods work.
    kb = await eng.create_kb("kb1", name="One")
    assert isinstance(kb, KnowledgeBaseLike)
    await eng.close()


async def test_close_tears_down_kbs(tmp_path):
    """close() should close every open KB before closing GM."""
    eng = GraphMemoryEngine(
        db_path=":memory:", embedder=_no_embed,
        kb_dir=str(tmp_path),
    )
    await eng.initialize()
    await eng.create_kb("a", name="A")
    await eng.create_kb("b", name="B")
    # close() should not raise even with multiple KBs open.
    await eng.close()


# --------------------------------------------------------------------
# KB management — happy paths via the engine surface
# --------------------------------------------------------------------


async def test_list_kbs_after_create(tmp_path):
    eng = GraphMemoryEngine(
        db_path=":memory:", embedder=_no_embed,
        kb_dir=str(tmp_path),
    )
    await eng.initialize()
    await eng.create_kb("kb1", name="First")
    await eng.create_kb("kb2", name="Second")
    kbs = await eng.list_kbs()
    ids = sorted(k["kb_id"] for k in kbs)
    assert ids == ["kb1", "kb2"]
    await eng.close()


async def test_open_kb_returns_existing(tmp_path):
    eng = GraphMemoryEngine(
        db_path=":memory:", embedder=_no_embed,
        kb_dir=str(tmp_path),
    )
    await eng.initialize()
    kb1 = await eng.create_kb("kbX", name="Test")
    kb2 = await eng.open_kb("kbX")
    # Same registry instance returns the cached KB on re-open.
    assert kb1 is kb2
    await eng.close()


async def test_set_archived_and_list_filters(tmp_path):
    eng = GraphMemoryEngine(
        db_path=":memory:", embedder=_no_embed,
        kb_dir=str(tmp_path),
    )
    await eng.initialize()
    await eng.create_kb("active1", name="A")
    await eng.create_kb("archived1", name="X")
    await eng.set_archived("archived1", True)

    active_only = await eng.list_kbs(include_archived=False)
    assert {k["kb_id"] for k in active_only} == {"active1"}

    all_kbs = await eng.list_kbs(include_archived=True)
    assert {k["kb_id"] for k in all_kbs} == {"active1", "archived1"}
    await eng.close()


async def test_delete_kb_removes_entry(tmp_path):
    eng = GraphMemoryEngine(
        db_path=":memory:", embedder=_no_embed,
        kb_dir=str(tmp_path),
    )
    await eng.initialize()
    await eng.create_kb("kb_drop", name="To Drop")
    assert {k["kb_id"] for k in await eng.list_kbs()} == {"kb_drop"}
    await eng.delete_kb("kb_drop")
    assert await eng.list_kbs() == []
    await eng.close()


async def test_list_kbs_before_initialize_returns_empty(tmp_path):
    """list_kbs should soft-fail (return []) pre-init rather than
    raise, since the dashboard polls it on startup."""
    eng = GraphMemoryEngine(
        db_path=":memory:", embedder=_no_embed,
        kb_dir=str(tmp_path),
    )
    assert await eng.list_kbs() == []


# --------------------------------------------------------------------
# Inherited GM behavior still works
# --------------------------------------------------------------------


async def test_inherited_node_crud(tmp_path):
    eng = GraphMemoryEngine(
        db_path=":memory:", embedder=_no_embed,
        kb_dir=str(tmp_path),
    )
    await eng.initialize()
    nid = await eng.insert_node(
        name="n1", category="FACT", description="hello",
    )
    assert nid > 0
    assert await eng.count_nodes() == 1
    await eng.close()
