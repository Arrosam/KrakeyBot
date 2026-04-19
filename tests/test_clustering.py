"""Phase 2.3a: Leiden clustering + community summarization."""
import pytest

from src.memory.graph_memory import GraphMemory
from src.sleep.clustering import run_leiden_clustering


class FixedEmbed:
    def __init__(self, mapping=None):
        self._m = mapping or {}

    async def __call__(self, text):
        return list(self._m.get(text, [0.5, 0.5]))


class CountingLLM:
    def __init__(self):
        self.calls = []
        self._counter = 0

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        self._counter += 1
        return f"summary #{self._counter}"


async def _gm(tmp_path):
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=FixedEmbed())
    await gm.initialize()
    return gm


# ---------------- empty + degenerate cases ----------------

async def test_empty_gm_returns_no_communities(tmp_path):
    gm = await _gm(tmp_path)
    out = await run_leiden_clustering(gm, llm=CountingLLM(),
                                         embedder=FixedEmbed())
    assert out == []
    await gm.close()


async def test_isolated_nodes_each_form_their_own_community(tmp_path):
    gm = await _gm(tmp_path)
    for n in ("a", "b", "c"):
        await gm.insert_node(name=n, category="FACT", description=n)

    out = await run_leiden_clustering(gm, llm=CountingLLM(),
                                         embedder=FixedEmbed())
    # 3 isolated nodes → 3 communities (each of size 1)
    assert len(out) == 3
    assert {c["size"] for c in out} == {1}
    await gm.close()


# ---------------- connected components ----------------

async def test_connected_component_forms_single_community(tmp_path):
    gm = await _gm(tmp_path)
    a = await gm.insert_node(name="a", category="FACT", description="")
    b = await gm.insert_node(name="b", category="FACT", description="")
    c = await gm.insert_node(name="c", category="FACT", description="")
    await gm.insert_edge_with_cycle_check(a, b, "RELATED_TO")
    await gm.insert_edge_with_cycle_check(b, c, "RELATED_TO")

    out = await run_leiden_clustering(gm, llm=CountingLLM(),
                                         embedder=FixedEmbed())
    # All three in one connected component
    assert len(out) == 1
    assert out[0]["size"] == 3
    assert set(out[0]["member_ids"]) == {a, b, c}
    await gm.close()


async def test_two_components_become_two_communities(tmp_path):
    gm = await _gm(tmp_path)
    a = await gm.insert_node(name="a", category="FACT", description="")
    b = await gm.insert_node(name="b", category="FACT", description="")
    c = await gm.insert_node(name="c", category="FACT", description="")
    d = await gm.insert_node(name="d", category="FACT", description="")
    await gm.insert_edge_with_cycle_check(a, b, "RELATED_TO")
    await gm.insert_edge_with_cycle_check(c, d, "RELATED_TO")

    out = await run_leiden_clustering(gm, llm=CountingLLM(),
                                         embedder=FixedEmbed())
    sizes = sorted(c["size"] for c in out)
    assert sizes == [2, 2]
    await gm.close()


# ---------------- summary + embedding ----------------

async def test_each_community_gets_llm_summary_and_embedding(tmp_path):
    gm = await _gm(tmp_path)
    a = await gm.insert_node(name="a", category="FACT", description="")
    b = await gm.insert_node(name="b", category="FACT", description="")
    await gm.insert_edge_with_cycle_check(a, b, "RELATED_TO")

    llm = CountingLLM()
    embed = FixedEmbed({"summary #1": [0.7, 0.3]})
    out = await run_leiden_clustering(gm, llm=llm, embedder=embed)

    assert len(out) == 1
    assert out[0]["summary"] == "summary #1"
    assert out[0]["embedding"] == [0.7, 0.3]
    assert len(llm.calls) == 1


async def test_communities_persisted_to_db(tmp_path):
    gm = await _gm(tmp_path)
    a = await gm.insert_node(name="a", category="FACT", description="")
    b = await gm.insert_node(name="b", category="FACT", description="")
    await gm.insert_edge_with_cycle_check(a, b, "RELATED_TO")

    await run_leiden_clustering(gm, llm=CountingLLM(),
                                  embedder=FixedEmbed())

    db = gm._require()
    async with db.execute("SELECT COUNT(*) FROM gm_communities") as cur:
        row = await cur.fetchone()
        assert row[0] == 1
    async with db.execute(
        "SELECT COUNT(*) FROM gm_node_communities"
    ) as cur:
        row = await cur.fetchone()
        assert row[0] == 2  # both nodes linked
    await gm.close()


async def test_min_size_filter_drops_small_communities(tmp_path):
    """When min_size=2, isolated nodes should be skipped (no LLM call)."""
    gm = await _gm(tmp_path)
    await gm.insert_node(name="lonely", category="FACT", description="")
    a = await gm.insert_node(name="a", category="FACT", description="")
    b = await gm.insert_node(name="b", category="FACT", description="")
    await gm.insert_edge_with_cycle_check(a, b, "RELATED_TO")

    llm = CountingLLM()
    out = await run_leiden_clustering(gm, llm=llm, embedder=FixedEmbed(),
                                         min_size=2)
    assert len(out) == 1
    assert out[0]["size"] == 2
    assert len(llm.calls) == 1
    await gm.close()
