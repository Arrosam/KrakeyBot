"""Phase 1.2b: upsert_node + cycle detection + edge insertion."""
import pytest

from krakey.memory.graph_memory import GraphMemory


class FakeEmbedder:
    async def __call__(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


async def _gm(tmp_path):
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=FakeEmbedder())
    await gm.initialize()
    return gm


# ---------------- upsert_node ----------------

async def test_upsert_new_name_creates_node(tmp_path):
    gm = await _gm(tmp_path)
    nid = await gm.upsert_node({"name": "apple", "category": "FACT",
                                  "description": "red fruit"})
    assert (await gm.get_node(nid))["name"] == "apple"
    assert await gm.count_nodes() == 1
    await gm.close()


async def test_upsert_same_name_and_category_updates_existing(tmp_path):
    gm = await _gm(tmp_path)
    nid1 = await gm.upsert_node({"name": "apple", "category": "FACT",
                                   "description": "fruit"})
    nid2 = await gm.upsert_node({"name": "apple", "category": "FACT",
                                   "description": "red apple"})
    assert nid1 == nid2
    assert await gm.count_nodes() == 1
    node = await gm.get_node(nid1)
    assert node["description"] == "red apple"
    assert node["importance"] > 1.0  # incremented on upsert


async def test_upsert_same_name_different_category_creates_new(tmp_path):
    gm = await _gm(tmp_path)
    await gm.upsert_node({"name": "goal1", "category": "FACT",
                           "description": "done"})
    await gm.upsert_node({"name": "goal1", "category": "TARGET",
                           "description": "pending"})
    assert await gm.count_nodes() == 2


# ---------------- cycle detection ----------------

async def test_would_create_cycle_unrelated_nodes_false(tmp_path):
    gm = await _gm(tmp_path)
    a = await gm.insert_node(name="a", category="FACT", description="")
    b = await gm.insert_node(name="b", category="FACT", description="")
    assert await gm.would_create_cycle(a, b) is False


async def test_would_create_cycle_direct_edge_true(tmp_path):
    gm = await _gm(tmp_path)
    a = await gm.insert_node(name="a", category="FACT", description="")
    b = await gm.insert_node(name="b", category="FACT", description="")
    await gm.insert_edge_with_cycle_check(a, b, predicate="RELATED_TO")
    assert await gm.would_create_cycle(a, b) is True


async def test_would_create_cycle_transitive_path_true(tmp_path):
    gm = await _gm(tmp_path)
    a = await gm.insert_node(name="a", category="FACT", description="")
    b = await gm.insert_node(name="b", category="FACT", description="")
    c = await gm.insert_node(name="c", category="FACT", description="")
    await gm.insert_edge_with_cycle_check(a, b, predicate="RELATED_TO")
    await gm.insert_edge_with_cycle_check(b, c, predicate="RELATED_TO")
    # adding a—c would close the a—b—c—a triangle
    assert await gm.would_create_cycle(a, c) is True


# ---------------- insert_edge_with_cycle_check ----------------

async def test_normal_edge_inserted_directly(tmp_path):
    gm = await _gm(tmp_path)
    a = await gm.insert_node(name="a", category="FACT", description="")
    b = await gm.insert_node(name="b", category="FACT", description="")
    edge_info = await gm.insert_edge_with_cycle_check(a, b, "RELATED_TO")
    assert edge_info["skipped"] is False
    assert await gm.count_edges() == 1


async def test_cycle_skips_edge_no_bridge_created(tmp_path):
    """Phase 1 fix: procedural bridges grow uncontrollably (each new bridge
    becomes part of fresh cycles → recursive bridge spam). Skip the edge
    instead. The two nodes remain connected via the existing path.
    """
    gm = await _gm(tmp_path)
    a = await gm.insert_node(name="a", category="FACT", description="")
    b = await gm.insert_node(name="b", category="FACT", description="")
    await gm.insert_edge_with_cycle_check(a, b, "RELATED_TO")

    edge_info = await gm.insert_edge_with_cycle_check(a, b, "CAUSES")
    assert edge_info["skipped"] is True
    # Still just the original edge, no bridge nodes created
    assert await gm.count_edges() == 1
    assert await gm.count_nodes() == 2  # no new bridge node


async def test_chained_cycles_do_not_produce_bridge_explosion(tmp_path):
    """Regression for the observed 25-bridge chain. Many edges that would
    each create cycles must not balloon node count.
    """
    gm = await _gm(tmp_path)
    a = await gm.insert_node(name="a", category="FACT", description="")
    b = await gm.insert_node(name="b", category="FACT", description="")
    c = await gm.insert_node(name="c", category="FACT", description="")
    await gm.insert_edge_with_cycle_check(a, b, "RELATED_TO")
    await gm.insert_edge_with_cycle_check(b, c, "RELATED_TO")
    # Now try 20 more edges that all close cycles
    for pred in ("CAUSES", "FOLLOWS", "SUPPORTS") * 7:
        await gm.insert_edge_with_cycle_check(a, c, pred)
    # Node count must still be 3 — no bridge growth
    assert await gm.count_nodes() == 3


async def test_edge_swaps_when_src_greater_than_tgt(tmp_path):
    gm = await _gm(tmp_path)
    a = await gm.insert_node(name="a", category="FACT", description="")
    b = await gm.insert_node(name="b", category="FACT", description="")
    # pass src > tgt; method must normalize so node_a < node_b CHECK passes
    await gm.insert_edge_with_cycle_check(b, a, "RELATED_TO")
    assert await gm.count_edges() == 1


async def test_self_loop_rejected(tmp_path):
    gm = await _gm(tmp_path)
    a = await gm.insert_node(name="a", category="FACT", description="")
    with pytest.raises(ValueError):
        await gm.insert_edge_with_cycle_check(a, a, "RELATED_TO")
