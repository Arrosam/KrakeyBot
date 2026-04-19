"""Phase 2.3b: migrate FACT/RELATION/KNOWLEDGE GM nodes into KBs by community."""
import pytest

from src.memory.graph_memory import GraphMemory
from src.memory.knowledge_base import KBRegistry
from src.sleep.clustering import run_leiden_clustering
from src.sleep.migration import migrate_gm_to_kb


class FixedEmbed:
    def __init__(self, mapping=None):
        self._m = mapping or {}

    async def __call__(self, text):
        return list(self._m.get(text, [0.5, 0.5]))


class CountingLLM:
    def __init__(self):
        self._n = 0

    async def chat(self, *a, **kw):
        self._n += 1
        return f"summary #{self._n}"


async def _setup(tmp_path):
    embed = FixedEmbed()
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=embed)
    await gm.initialize()
    reg = KBRegistry(gm, kb_dir=tmp_path / "kbs", embedder=embed)
    return gm, reg


# ---------------- core migration ----------------

async def test_fact_node_migrated_to_kb_and_removed_from_gm(tmp_path):
    gm, reg = await _setup(tmp_path)
    a = await gm.insert_node(name="apple", category="FACT",
                                description="red fruit",
                                embedding=[1.0, 0.0])
    b = await gm.insert_node(name="banana", category="FACT",
                                description="yellow fruit",
                                embedding=[0.95, 0.31])
    await gm.insert_edge_with_cycle_check(a, b, "RELATED_TO")

    await run_leiden_clustering(gm, llm=CountingLLM(), embedder=FixedEmbed())
    result = await migrate_gm_to_kb(gm, reg)

    assert result["migrated_nodes"] == 2
    # GM should no longer have those FACT nodes
    remaining = await gm.list_nodes(category="FACT")
    assert remaining == []
    # KB should have 2 entries
    kbs = await reg.list_kbs()
    assert len(kbs) == 1
    kb = await reg.open_kb(kbs[0]["kb_id"])
    assert await kb.count_entries() == 2
    await reg.close_all()
    await gm.close()


async def test_intra_community_edges_migrated_to_kb(tmp_path):
    gm, reg = await _setup(tmp_path)
    a = await gm.insert_node(name="a", category="FACT", description="",
                                embedding=[1.0, 0.0])
    b = await gm.insert_node(name="b", category="FACT", description="",
                                embedding=[0.99, 0.14])
    await gm.insert_edge_with_cycle_check(a, b, "CAUSES")

    await run_leiden_clustering(gm, llm=CountingLLM(), embedder=FixedEmbed())
    await migrate_gm_to_kb(gm, reg)

    kbs = await reg.list_kbs()
    kb = await reg.open_kb(kbs[0]["kb_id"])
    db = kb._require()
    async with db.execute("SELECT COUNT(*) FROM kb_edges") as cur:
        row = await cur.fetchone()
        assert row[0] == 1
    await reg.close_all()
    await gm.close()


async def test_two_communities_yield_two_kbs(tmp_path):
    gm, reg = await _setup(tmp_path)
    a = await gm.insert_node(name="a", category="FACT", description="",
                                embedding=[1.0, 0.0])
    b = await gm.insert_node(name="b", category="FACT", description="",
                                embedding=[1.0, 0.0])
    c = await gm.insert_node(name="c", category="FACT", description="",
                                embedding=[0.0, 1.0])
    d = await gm.insert_node(name="d", category="FACT", description="",
                                embedding=[0.0, 1.0])
    await gm.insert_edge_with_cycle_check(a, b, "RELATED_TO")
    await gm.insert_edge_with_cycle_check(c, d, "RELATED_TO")

    await run_leiden_clustering(gm, llm=CountingLLM(), embedder=FixedEmbed())
    await migrate_gm_to_kb(gm, reg)

    kbs = await reg.list_kbs()
    assert len(kbs) == 2
    await reg.close_all()
    await gm.close()


async def test_target_and_focus_not_migrated(tmp_path):
    """Sleep §11.3: only FACT/RELATION/KNOWLEDGE go to KB; TARGET/FOCUS
    are handled by Phase 4/5."""
    gm, reg = await _setup(tmp_path)
    fact = await gm.insert_node(name="x", category="FACT", description="",
                                   embedding=[1.0, 0.0])
    target = await gm.insert_node(name="t", category="TARGET", description="",
                                     embedding=[1.0, 0.0])
    focus = await gm.insert_node(name="f", category="FOCUS", description="",
                                    embedding=[1.0, 0.0])

    await run_leiden_clustering(gm, llm=CountingLLM(), embedder=FixedEmbed())
    await migrate_gm_to_kb(gm, reg)

    targets = await gm.list_nodes(category="TARGET")
    focuses = await gm.list_nodes(category="FOCUS")
    facts = await gm.list_nodes(category="FACT")
    assert len(targets) == 1
    assert len(focuses) == 1
    assert facts == []  # migrated
    await reg.close_all()
    await gm.close()


async def test_node_without_community_skipped(tmp_path):
    """If a node ended up with no community membership (shouldn't happen
    after Leiden), migration must skip it gracefully."""
    gm, reg = await _setup(tmp_path)
    # Insert a FACT but skip clustering — no community membership.
    await gm.insert_node(name="orphan", category="FACT", description="",
                            embedding=[1.0, 0.0])
    result = await migrate_gm_to_kb(gm, reg)
    assert result["migrated_nodes"] == 0
    assert result["skipped_no_community"] == 1
    # Still in GM
    remaining = await gm.list_nodes(category="FACT")
    assert len(remaining) == 1
    await reg.close_all()
    await gm.close()
