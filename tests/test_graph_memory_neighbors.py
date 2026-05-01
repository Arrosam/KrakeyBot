"""Phase 1.3e helpers: neighbor keywords + edges among a node set."""
import pytest

from krakey.memory.graph_memory import GraphMemory


class Embed:
    async def __call__(self, text): return [0.0]


async def _gm(tmp_path):
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=Embed())
    await gm.initialize()
    return gm


async def test_neighbor_keywords_direct_connections(tmp_path):
    gm = await _gm(tmp_path)
    a = await gm.insert_node(name="alpha", category="FACT", description="")
    b = await gm.insert_node(name="beta",  category="FACT", description="")
    c = await gm.insert_node(name="gamma", category="FACT", description="")
    await gm.insert_edge_with_cycle_check(a, b, "RELATED_TO")
    await gm.insert_edge_with_cycle_check(a, c, "RELATED_TO")

    out = await gm.get_neighbor_keywords([a])
    assert set(out[a]) == {"beta", "gamma"}
    await gm.close()


async def test_neighbor_keywords_no_edges_empty(tmp_path):
    gm = await _gm(tmp_path)
    a = await gm.insert_node(name="alpha", category="FACT", description="")
    out = await gm.get_neighbor_keywords([a])
    assert out == {a: []}
    await gm.close()


async def test_neighbor_keywords_unknown_id_absent(tmp_path):
    gm = await _gm(tmp_path)
    out = await gm.get_neighbor_keywords([999])
    assert out == {999: []}
    await gm.close()


async def test_get_edges_among_returns_only_internal_edges(tmp_path):
    gm = await _gm(tmp_path)
    a = await gm.insert_node(name="a", category="FACT", description="")
    b = await gm.insert_node(name="b", category="FACT", description="")
    c = await gm.insert_node(name="c", category="FACT", description="")
    await gm.insert_edge_with_cycle_check(a, b, "RELATED_TO")
    await gm.insert_edge_with_cycle_check(a, c, "CAUSES")

    edges = await gm.get_edges_among([a, b])
    assert len(edges) == 1
    e = edges[0]
    assert {e["source"], e["target"]} == {"a", "b"}
    assert e["predicate"] == "RELATED_TO"
    await gm.close()


async def test_get_edges_among_empty_set(tmp_path):
    gm = await _gm(tmp_path)
    assert await gm.get_edges_among([]) == []
    await gm.close()
