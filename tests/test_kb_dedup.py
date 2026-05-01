"""KB dedup primitives: vec_search returning scored pairs + merge_entry.

These exercise the building blocks the sleep-migration dedup pass uses;
the migration end-to-end wiring is covered separately in
``test_migration.py``.
"""
import pytest

from krakey.memory.knowledge_base import KBRegistry
from krakey.memory.graph_memory import GraphMemory


class FixedEmbed:
    def __init__(self, mapping=None):
        self._m = mapping or {}

    async def __call__(self, text):
        return list(self._m.get(text, [0.5, 0.5]))


async def _setup(tmp_path):
    embed = FixedEmbed()
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=embed)
    await gm.initialize()
    reg = KBRegistry(gm, kb_dir=tmp_path / "kbs", embedder=embed)
    kb = await reg.create_kb("test_kb", name="test", description="t")
    return gm, reg, kb


# -------- vec_search --------

async def test_vec_search_empty_kb_returns_empty(tmp_path):
    gm, reg, kb = await _setup(tmp_path)
    out = await kb.vec_search([1.0, 0.0], top_k=5)
    assert out == []
    await reg.close_all()
    await gm.close()


async def test_vec_search_returns_pairs_sorted_by_cosine(tmp_path):
    gm, reg, kb = await _setup(tmp_path)
    await kb.write_entry("close", embedding=[1.0, 0.05], importance=1.0)
    await kb.write_entry("middle", embedding=[0.7, 0.7], importance=1.0)
    await kb.write_entry("far", embedding=[0.0, 1.0], importance=1.0)

    pairs = await kb.vec_search([1.0, 0.0], top_k=5, min_similarity=0.0)
    assert len(pairs) == 3
    contents = [e["content"] for (e, _s) in pairs]
    assert contents == ["close", "middle", "far"]
    # Each pair carries a similarity float; descending.
    sims = [s for (_e, s) in pairs]
    assert sims == sorted(sims, reverse=True)
    assert all(isinstance(s, float) for s in sims)
    await reg.close_all()
    await gm.close()


async def test_vec_search_filters_inactive_entries(tmp_path):
    gm, reg, kb = await _setup(tmp_path)
    eid = await kb.write_entry("hide me", embedding=[1.0, 0.0])
    await kb.write_entry("keep me", embedding=[0.99, 0.0])

    # Deactivate the first entry directly.
    db = kb._require()
    await db.execute("UPDATE kb_entries SET is_active=0 WHERE id=?", (eid,))
    await db.commit()

    pairs = await kb.vec_search([1.0, 0.0], top_k=5)
    contents = [e["content"] for (e, _s) in pairs]
    assert contents == ["keep me"]
    await reg.close_all()
    await gm.close()


# -------- merge_entry --------

async def test_merge_entry_replaces_content_and_sums_importance(tmp_path):
    gm, reg, kb = await _setup(tmp_path)
    eid = await kb.write_entry("old text", tags=["A"],
                                  embedding=[1.0, 0.0], importance=1.5)

    await kb.merge_entry(
        eid, new_content="new text",
        new_embedding=[0.0, 1.0],
        new_importance=2.0,
        new_tags=["B"],
    )

    got = await kb.get_entry(eid)
    assert got is not None
    assert got["content"] == "new text"
    assert got["embedding"] == [0.0, 1.0]
    assert got["importance"] == pytest.approx(3.5)
    assert got["tags"] == ["A", "B"]
    await reg.close_all()
    await gm.close()


async def test_merge_entry_unions_and_dedupes_tags(tmp_path):
    gm, reg, kb = await _setup(tmp_path)
    eid = await kb.write_entry("c", tags=["FACT", "apple"],
                                  embedding=[1.0, 0.0], importance=1.0)

    await kb.merge_entry(
        eid, new_content="c2",
        new_embedding=[1.0, 0.0],
        new_importance=0.5,
        new_tags=["FACT", "fruit"],
    )

    got = await kb.get_entry(eid)
    assert got["tags"] == ["FACT", "apple", "fruit"]   # sorted, deduped
    assert got["importance"] == pytest.approx(1.5)
    await reg.close_all()
    await gm.close()


async def test_merge_entry_raises_on_unknown_id(tmp_path):
    gm, reg, kb = await _setup(tmp_path)
    with pytest.raises(KeyError):
        await kb.merge_entry(
            9999, new_content="x", new_embedding=None,
            new_importance=1.0, new_tags=[],
        )
    await reg.close_all()
    await gm.close()
