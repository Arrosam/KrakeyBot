"""KB lifecycle: index vector + importance + consolidate + archive + revive."""
from __future__ import annotations

import pytest

from krakey.memory.graph_memory import GraphMemory
from krakey.memory.knowledge_base import KBRegistry
from krakey.memory.sleep.kb_lifecycle import (
    archive_excess_kbs, compute_kb_importance, compute_kb_index_embedding,
    consolidate_kbs, find_revive_target, revive_kb,
)


class StubEmbed:
    async def __call__(self, text):
        return [0.0]


async def _new(tmp_path):
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=StubEmbed())
    await gm.initialize()
    reg = KBRegistry(gm, kb_dir=tmp_path / "kbs", embedder=StubEmbed())
    return gm, reg


# ---------------- compute helpers ----------------

async def test_compute_index_embedding_averages_member_vectors(tmp_path):
    _, reg = await _new(tmp_path)
    kb = await reg.create_kb("k1", name="K1")
    await kb.write_entry("a", embedding=[1.0, 0.0, 0.0])
    await kb.write_entry("b", embedding=[0.0, 1.0, 0.0])
    emb = await compute_kb_index_embedding(kb)
    assert emb == pytest.approx([0.5, 0.5, 0.0])


async def test_compute_index_embedding_none_when_no_vectors(tmp_path):
    _, reg = await _new(tmp_path)
    kb = await reg.create_kb("k1", name="K1")
    await kb.write_entry("no embedding here")
    assert await compute_kb_index_embedding(kb) is None


async def test_compute_importance_zero_when_empty(tmp_path):
    _, reg = await _new(tmp_path)
    kb = await reg.create_kb("k1", name="K1")
    assert await compute_kb_importance(kb) == 0.0


async def test_compute_importance_count_times_mean(tmp_path):
    _, reg = await _new(tmp_path)
    kb = await reg.create_kb("k1", name="K1")
    await kb.write_entry("a", importance=2.0)
    await kb.write_entry("b", importance=4.0)
    # 2 entries × mean(2,4)=3.0 → 6.0
    assert await compute_kb_importance(kb) == pytest.approx(6.0)


# ---------------- consolidation ----------------

async def test_consolidate_merges_similar_kbs(tmp_path):
    _, reg = await _new(tmp_path)
    a = await reg.create_kb("a", name="A")
    b = await reg.create_kb("b", name="B")
    # Both KBs have entries with parallel embeddings (cosine ≈ 1.0)
    await a.write_entry("a1", embedding=[1.0, 0.0])
    await a.write_entry("a2", embedding=[1.0, 0.001])
    await b.write_entry("b1", embedding=[1.0, 0.0])

    stats = await consolidate_kbs(reg, threshold=0.9)
    assert stats["merged"] == 1
    remaining = await reg.list_kbs()
    assert len(remaining) == 1
    # The smaller one (b, 1 entry) is absorbed into the larger (a, 2 entries)
    survivor = remaining[0]
    kb = await reg.open_kb(survivor["kb_id"])
    assert await kb.count_entries() == 3  # a1 + a2 + b1 copied in


async def test_consolidate_keeps_orthogonal_kbs(tmp_path):
    _, reg = await _new(tmp_path)
    a = await reg.create_kb("a", name="A")
    b = await reg.create_kb("b", name="B")
    await a.write_entry("a1", embedding=[1.0, 0.0])
    await b.write_entry("b1", embedding=[0.0, 1.0])  # orthogonal → cos=0

    stats = await consolidate_kbs(reg, threshold=0.5)
    assert stats["merged"] == 0
    assert len(await reg.list_kbs()) == 2


# ---------------- archive ----------------

async def test_archive_does_nothing_under_max(tmp_path):
    gm, reg = await _new(tmp_path)
    await reg.create_kb("a", name="A")
    stats = await archive_excess_kbs(reg, gm, max_count=10, archive_pct=10)
    assert stats["archived"] == 0


async def test_archive_drops_lowest_importance_pct(tmp_path):
    gm, reg = await _new(tmp_path)
    # Create 10 KBs with descending importance
    for i in range(10):
        kb = await reg.create_kb(f"k{i}", name=f"K{i}")
        # importance = entries × mean(importance); fewer entries = lower
        for j in range(i + 1):
            await kb.write_entry(f"e{j}", embedding=[float(i), 0.0],
                                  importance=1.0)

    stats = await archive_excess_kbs(reg, gm, max_count=5, archive_pct=20)
    # 20% of 10 = 2 archived
    assert stats["archived"] == 2
    active = await reg.list_kbs(include_archived=False)
    assert len(active) == 8
    # k0 + k1 (lowest importance) archived
    archived_ids = {k["kb_id"] for k in await reg.list_kbs(include_archived=True)
                    if k["is_archived"]}
    assert archived_ids == {"k0", "k1"}


async def test_archive_drops_gm_index_node(tmp_path):
    gm, reg = await _new(tmp_path)
    # Pre-create a fake GM index node for kb 'doomed'
    await gm.upsert_node({
        "name": "KB:doomed", "category": "KNOWLEDGE",
        "description": "x", "embedding": [0.0],
        "metadata": {"is_kb_index": True, "kb_id": "doomed"},
    })
    # Two KBs so we exceed max=1
    a = await reg.create_kb("a", name="A")
    await a.write_entry("a1", embedding=[1.0, 0.0], importance=10.0)
    d = await reg.create_kb("doomed", name="Doomed")
    await d.write_entry("d1", embedding=[1.0, 0.0], importance=0.1)

    await archive_excess_kbs(reg, gm, max_count=1, archive_pct=50)
    # The 'doomed' KB index node should be gone
    nodes = await gm.list_nodes(category="KNOWLEDGE")
    names = {n["name"] for n in nodes}
    assert "KB:doomed" not in names


# ---------------- revive ----------------

async def test_find_revive_target_returns_match(tmp_path):
    _, reg = await _new(tmp_path)
    kb = await reg.create_kb("old", name="Old")
    await reg.set_index_embedding("old", [1.0, 0.0])
    await reg.set_archived("old", True)

    # Highly similar query → returns "old"
    match = await find_revive_target(reg, [1.0, 0.001], threshold=0.9)
    assert match == "old"


async def test_find_revive_target_misses_below_threshold(tmp_path):
    _, reg = await _new(tmp_path)
    await reg.create_kb("old", name="Old")
    await reg.set_index_embedding("old", [1.0, 0.0])
    await reg.set_archived("old", True)

    match = await find_revive_target(reg, [0.0, 1.0], threshold=0.5)
    assert match is None


async def test_find_revive_target_skips_active_kbs(tmp_path):
    _, reg = await _new(tmp_path)
    await reg.create_kb("active", name="A")
    await reg.set_index_embedding("active", [1.0, 0.0])
    # not archived
    match = await find_revive_target(reg, [1.0, 0.0], threshold=0.5)
    assert match is None


async def test_revive_kb_clears_archived_flag(tmp_path):
    _, reg = await _new(tmp_path)
    await reg.create_kb("k", name="K")
    await reg.set_archived("k", True)
    kb = await revive_kb(reg, "k")
    metas = await reg.list_kbs(include_archived=False)
    assert any(m["kb_id"] == "k" for m in metas)


# ---------------- list filtering ----------------

async def test_list_kbs_default_excludes_archived(tmp_path):
    _, reg = await _new(tmp_path)
    await reg.create_kb("active", name="A")
    await reg.create_kb("dead", name="D")
    await reg.set_archived("dead", True)

    default = await reg.list_kbs()
    assert {k["kb_id"] for k in default} == {"active"}

    full = await reg.list_kbs(include_archived=True)
    assert {k["kb_id"] for k in full} == {"active", "dead"}


async def test_delete_kb_removes_file_and_row(tmp_path):
    _, reg = await _new(tmp_path)
    kb = await reg.create_kb("k", name="K")
    path = kb.path
    assert path.exists()
    await reg.delete_kb("k")
    assert not path.exists()
    assert await reg.list_kbs(include_archived=True) == []
