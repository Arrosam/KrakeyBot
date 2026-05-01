"""KnowledgeBase + KBRegistry — long-term per-topic SQLite stores."""
from pathlib import Path

import pytest

from krakey.memory.graph_memory import GraphMemory
from krakey.memory.knowledge_base import KBRegistry, KnowledgeBase


class FixedEmbed:
    def __init__(self, mapping=None):
        self._m = mapping or {}

    async def __call__(self, text):
        if text in self._m:
            return list(self._m[text])
        # default: orthogonal-ish vector keyed off length
        return [1.0, 0.0]


class FailingEmbed:
    async def __call__(self, text):
        raise RuntimeError("embed down")


async def _gm(tmp_path):
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=FixedEmbed())
    await gm.initialize()
    return gm


# ---------------- KnowledgeBase basics ----------------

async def test_kb_initialize_creates_schema(tmp_path):
    kb = KnowledgeBase("astro", tmp_path / "astro.sqlite", embedder=FixedEmbed())
    await kb.initialize()
    # Should be able to write and read
    eid = await kb.write_entry("Sun is a star", embedding=[1.0, 0.0])
    assert eid > 0
    await kb.close()


async def test_kb_write_entry_persists(tmp_path):
    kb = KnowledgeBase("k", tmp_path / "k.sqlite", embedder=FixedEmbed())
    await kb.initialize()
    eid = await kb.write_entry("hello world", tags=["greet"],
                                  embedding=[1.0, 0.0], importance=2.5)
    row = await kb.get_entry(eid)
    assert row["content"] == "hello world"
    assert row["importance"] == 2.5
    assert "greet" in (row["tags"] or [])
    await kb.close()


async def test_kb_write_edge_with_check_constraint(tmp_path):
    kb = KnowledgeBase("k", tmp_path / "k.sqlite", embedder=FixedEmbed())
    await kb.initialize()
    a = await kb.write_entry("a", embedding=[1.0, 0.0])
    b = await kb.write_entry("b", embedding=[0.0, 1.0])
    info = await kb.write_edge(a, b, "RELATED_TO")
    assert info["written"] is True
    # Reverse order auto-normalized
    info2 = await kb.write_edge(b, a, "CAUSES")
    assert info2["written"] is True
    await kb.close()


async def test_kb_vec_search_returns_top_matches(tmp_path):
    embed = FixedEmbed({"q": [1.0, 0.0]})
    kb = KnowledgeBase("k", tmp_path / "k.sqlite", embedder=embed)
    await kb.initialize()
    await kb.write_entry("apple", embedding=[1.0, 0.0])
    await kb.write_entry("banana", embedding=[0.95, 0.31])
    await kb.write_entry("car", embedding=[0.0, 1.0])

    hits = await kb.search("q", top_k=2)
    contents = [h["content"] for h in hits]
    assert contents[0] == "apple"
    assert "car" not in contents
    await kb.close()


async def test_kb_falls_back_to_fts_when_embed_fails(tmp_path):
    kb = KnowledgeBase("k", tmp_path / "k.sqlite", embedder=FailingEmbed())
    await kb.initialize()
    await kb.write_entry("apple is red")
    await kb.write_entry("banana is yellow")
    hits = await kb.search("apple", top_k=5)
    contents = [h["content"] for h in hits]
    assert any("apple" in c for c in contents)
    await kb.close()


async def test_kb_fts_search_direct(tmp_path):
    kb = KnowledgeBase("k", tmp_path / "k.sqlite", embedder=FixedEmbed())
    await kb.initialize()
    await kb.write_entry("apple is red", tags=["fruit"])
    await kb.write_entry("car is fast")

    hits = await kb.fts_search("apple")
    assert len(hits) == 1
    assert "apple" in hits[0]["content"]
    await kb.close()


# ---------------- KBRegistry ----------------

async def test_registry_create_kb_writes_to_gm_registry_and_disk(tmp_path):
    gm = await _gm(tmp_path)
    kb_dir = tmp_path / "knowledge_bases"
    reg = KBRegistry(gm, kb_dir=kb_dir, embedder=FixedEmbed())

    kb = await reg.create_kb("astronomy", name="Astronomy",
                              description="planets and stars")
    assert (kb_dir / "astronomy.sqlite").exists()
    metas = await reg.list_kbs()
    assert any(m["kb_id"] == "astronomy" for m in metas)
    assert kb.kb_id == "astronomy"

    await reg.close_all()
    await gm.close()


async def test_registry_open_kb_returns_same_instance(tmp_path):
    gm = await _gm(tmp_path)
    reg = KBRegistry(gm, kb_dir=tmp_path / "kbs", embedder=FixedEmbed())

    await reg.create_kb("k", name="K")
    a = await reg.open_kb("k")
    b = await reg.open_kb("k")
    assert a is b

    await reg.close_all()
    await gm.close()


async def test_registry_open_unknown_kb_raises(tmp_path):
    gm = await _gm(tmp_path)
    reg = KBRegistry(gm, kb_dir=tmp_path / "kbs", embedder=FixedEmbed())
    with pytest.raises(KeyError):
        await reg.open_kb("nope")
    await reg.close_all()
    await gm.close()


async def test_registry_create_duplicate_kb_raises(tmp_path):
    gm = await _gm(tmp_path)
    reg = KBRegistry(gm, kb_dir=tmp_path / "kbs", embedder=FixedEmbed())
    await reg.create_kb("k", name="K")
    with pytest.raises(ValueError):
        await reg.create_kb("k", name="K2")
    await reg.close_all()
    await gm.close()


async def test_registry_list_kbs_includes_topics(tmp_path):
    gm = await _gm(tmp_path)
    reg = KBRegistry(gm, kb_dir=tmp_path / "kbs", embedder=FixedEmbed())
    await reg.create_kb("astro", name="Astronomy",
                         description="space", topics=["planet", "star"])
    metas = await reg.list_kbs()
    me = next(m for m in metas if m["kb_id"] == "astro")
    assert me["name"] == "Astronomy"
    assert me["description"] == "space"
    assert "planet" in (me.get("topics") or [])
    await reg.close_all()
    await gm.close()
