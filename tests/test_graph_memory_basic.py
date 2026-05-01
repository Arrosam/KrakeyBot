"""Phase 1.2a: GM init + basic node CRUD.

Embeddings mocked — tests focus on DB behavior.
"""
import pytest

from krakey.memory.graph_memory import GraphMemory


class FakeEmbedder:
    def __init__(self, fixed=None):
        self._fixed = fixed or [0.1, 0.2, 0.3]
        self.calls = []

    async def __call__(self, text: str) -> list[float]:
        self.calls.append(text)
        return list(self._fixed)


async def _gm(tmp_path):
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=FakeEmbedder())
    await gm.initialize()
    return gm


async def test_initialize_creates_empty_schema(tmp_path):
    gm = await _gm(tmp_path)
    assert await gm.count_nodes() == 0
    assert await gm.count_edges() == 0
    await gm.close()


async def test_initialize_is_idempotent(tmp_path):
    path = tmp_path / "gm.sqlite"
    gm1 = GraphMemory(path, embedder=FakeEmbedder())
    await gm1.initialize()
    await gm1.insert_node(name="a", category="FACT", description="x")
    await gm1.close()

    gm2 = GraphMemory(path, embedder=FakeEmbedder())
    await gm2.initialize()
    assert await gm2.count_nodes() == 1
    await gm2.close()


async def test_insert_and_fetch_node(tmp_path):
    gm = await _gm(tmp_path)
    node_id = await gm.insert_node(
        name="apple", category="FACT", description="red fruit",
        embedding=[1.0, 2.0, 3.0], source_type="auto", importance=2.5,
    )
    node = await gm.get_node(node_id)
    assert node["name"] == "apple"
    assert node["category"] == "FACT"
    assert node["description"] == "red fruit"
    assert node["importance"] == 2.5
    assert node["source_type"] == "auto"
    assert node["embedding"] == [1.0, 2.0, 3.0]
    await gm.close()


async def test_insert_node_default_importance_and_source(tmp_path):
    gm = await _gm(tmp_path)
    node_id = await gm.insert_node(name="x", category="FOCUS", description="")
    node = await gm.get_node(node_id)
    assert node["importance"] == 1.0
    assert node["source_type"] == "auto"
    assert node["embedding"] is None
    await gm.close()


async def test_list_nodes_filters_by_category(tmp_path):
    gm = await _gm(tmp_path)
    await gm.insert_node(name="f1", category="FACT", description="")
    await gm.insert_node(name="f2", category="FACT", description="")
    await gm.insert_node(name="t1", category="TARGET", description="")

    facts = await gm.list_nodes(category="FACT")
    targets = await gm.list_nodes(category="TARGET")
    assert len(facts) == 2 and {n["name"] for n in facts} == {"f1", "f2"}
    assert len(targets) == 1 and targets[0]["name"] == "t1"
    await gm.close()


async def test_update_node_metadata_merges(tmp_path):
    gm = await _gm(tmp_path)
    node_id = await gm.insert_node(name="n", category="FACT", description="")
    await gm.set_metadata(node_id, {"classified": True})
    node = await gm.get_node(node_id)
    assert node["metadata"] == {"classified": True}

    await gm.set_metadata(node_id, {"extra": "value"})
    node = await gm.get_node(node_id)
    assert node["metadata"] == {"classified": True, "extra": "value"}
    await gm.close()


async def test_get_node_missing_returns_none(tmp_path):
    gm = await _gm(tmp_path)
    assert await gm.get_node(999) is None
    await gm.close()


async def test_delete_node_removes_fts_entry(tmp_path):
    gm = await _gm(tmp_path)
    nid = await gm.insert_node(name="banana", category="FACT", description="yellow")
    await gm.delete_node(nid)
    assert await gm.get_node(nid) is None
    assert await gm.count_nodes() == 0
    await gm.close()


async def test_sqlite_vec_extension_loaded(tmp_path):
    gm = await _gm(tmp_path)
    # vec_version() only works if sqlite-vec is loaded
    row = await gm.raw_fetchone("SELECT vec_version()")
    assert row[0].startswith("v0.")
    await gm.close()
