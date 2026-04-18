"""Phase 1.3d: FTS5 text search as embedding-unavailable fallback."""
import pytest

from src.memory.graph_memory import GraphMemory


class Embed:
    async def __call__(self, text): return [0.0]


async def _gm(tmp_path):
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=Embed())
    await gm.initialize()
    return gm


async def test_fts_finds_node_by_name(tmp_path):
    gm = await _gm(tmp_path)
    await gm.insert_node(name="apple", category="FACT", description="red fruit")
    await gm.insert_node(name="banana", category="FACT",
                          description="yellow fruit")
    await gm.insert_node(name="car", category="FACT", description="vehicle")

    hits = await gm.fts_search("apple")
    assert [n["name"] for n in hits] == ["apple"]
    await gm.close()


async def test_fts_finds_by_description_word(tmp_path):
    gm = await _gm(tmp_path)
    await gm.insert_node(name="apple", category="FACT", description="red fruit")
    await gm.insert_node(name="banana", category="FACT",
                          description="yellow fruit")
    hits = await gm.fts_search("fruit")
    names = {n["name"] for n in hits}
    assert names == {"apple", "banana"}
    await gm.close()


async def test_fts_multi_word_or_match(tmp_path):
    gm = await _gm(tmp_path)
    await gm.insert_node(name="apple", category="FACT", description="red fruit")
    await gm.insert_node(name="car", category="FACT", description="a vehicle")

    # Matches "apple" OR "vehicle" — both should appear
    hits = await gm.fts_search("apple vehicle")
    names = {n["name"] for n in hits}
    assert names == {"apple", "car"}
    await gm.close()


async def test_fts_empty_query_returns_empty(tmp_path):
    gm = await _gm(tmp_path)
    await gm.insert_node(name="apple", category="FACT", description="red fruit")
    assert await gm.fts_search("") == []
    assert await gm.fts_search("   ") == []
    await gm.close()


async def test_fts_special_chars_do_not_crash(tmp_path):
    gm = await _gm(tmp_path)
    await gm.insert_node(name="apple", category="FACT", description="red fruit")
    # Should not raise — operators are sanitized away
    await gm.fts_search('"AND" OR NOT *(apple)*')
    await gm.close()


async def test_fts_respects_top_k(tmp_path):
    gm = await _gm(tmp_path)
    for i in range(5):
        await gm.insert_node(name=f"banana{i}", category="FACT",
                              description="yellow fruit")
    hits = await gm.fts_search("fruit", top_k=3)
    assert len(hits) == 3
    await gm.close()


async def test_fts_unicode_tokens(tmp_path):
    gm = await _gm(tmp_path)
    await gm.insert_node(name="苹果", category="FACT", description="红色水果")
    hits = await gm.fts_search("苹果")
    assert len(hits) == 1 and hits[0]["name"] == "苹果"
    await gm.close()
