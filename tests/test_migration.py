"""Phase 2.3b: migrate FACT/RELATION/KNOWLEDGE GM nodes into KBs by community."""
import pytest

from src.memory.graph_memory import GraphMemory
from src.memory.knowledge_base import KBRegistry
from src.memory.sleep.clustering import run_leiden_clustering
from src.memory.sleep.migration import migrate_gm_to_kb


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


# ---------------- min_community_size ----------------

async def test_singleton_skipped_when_min_size_2(tmp_path):
    gm, reg = await _setup(tmp_path)
    # Single isolated node → community of size 1
    n = await gm.insert_node(name="lone", category="FACT",
                                description="alone", embedding=[1.0, 0.0])
    await run_leiden_clustering(gm, llm=CountingLLM(), embedder=FixedEmbed())

    stats = await migrate_gm_to_kb(gm, reg, min_community_size=2)
    assert stats["migrated_nodes"] == 0
    assert stats["skipped_small_community"] == 1
    # Node still in GM (NOT migrated)
    assert (await gm.list_nodes(category="FACT"))[0]["id"] == n


async def test_singleton_migrated_when_min_size_1(tmp_path):
    gm, reg = await _setup(tmp_path)
    await gm.insert_node(name="lone", category="FACT",
                            description="alone", embedding=[1.0, 0.0])
    await run_leiden_clustering(gm, llm=CountingLLM(), embedder=FixedEmbed())
    stats = await migrate_gm_to_kb(gm, reg, min_community_size=1)
    assert stats["migrated_nodes"] == 1
    assert stats["skipped_small_community"] == 0


# ---------------- KB revive on archived match ----------------

async def test_migration_revives_archived_kb_on_summary_cosine_match(tmp_path):
    """When a new community's summary embedding matches an archived KB's
    stored index vector, migration should reactivate the archive instead
    of creating a new community_X KB.
    """
    gm, reg = await _setup(tmp_path)
    # 1) Pre-existing archived KB with known index vector
    old = await reg.create_kb("astro_old", name="Old Astro",
                                 description="ancient")
    await reg.set_index_embedding("astro_old", [1.0, 0.0])
    await reg.set_archived("astro_old", True)

    # 2) Stage a community whose summary_embedding is highly similar.
    #    Easiest path: insert a community row directly + map a node to it.
    db = gm._require()  # noqa: SLF001
    from src.memory._db import encode_embedding
    cur = await db.execute(
        "INSERT INTO gm_communities(name, summary, summary_embedding, "
        "member_count) VALUES(?, ?, ?, 1)",
        ("new astro", "new astro related stuff",
         encode_embedding([1.0, 0.001])),
    )
    cid = cur.lastrowid
    n = await gm.insert_node(name="star", category="FACT",
                                description="luminous body",
                                embedding=[1.0, 0.0])
    await db.execute(
        "INSERT INTO gm_node_communities(node_id, community_id) VALUES(?, ?)",
        (n, cid),
    )
    await db.commit()

    stats = await migrate_gm_to_kb(gm, reg, min_community_size=1,
                                       revive_threshold=0.95)
    assert stats["kbs_revived"] == 1
    assert stats["kbs_created"] == 0

    # The revived KB ('astro_old') should now hold the new entry
    active = await reg.list_kbs(include_archived=False)
    assert {k["kb_id"] for k in active} == {"astro_old"}
    kb = await reg.open_kb("astro_old")
    assert await kb.count_entries() == 1


async def test_migration_creates_new_kb_when_no_archive_matches(tmp_path):
    gm, reg = await _setup(tmp_path)
    await reg.create_kb("astro_old", name="Old")
    await reg.set_index_embedding("astro_old", [1.0, 0.0])
    await reg.set_archived("astro_old", True)

    # New community embedding is orthogonal → no match
    db = gm._require()  # noqa: SLF001
    from src.memory._db import encode_embedding
    cur = await db.execute(
        "INSERT INTO gm_communities(name, summary, summary_embedding, "
        "member_count) VALUES(?, ?, ?, 1)",
        ("biology", "cells and DNA", encode_embedding([0.0, 1.0])),
    )
    cid = cur.lastrowid
    n = await gm.insert_node(name="cell", category="FACT", description="x",
                                embedding=[0.0, 1.0])
    await db.execute(
        "INSERT INTO gm_node_communities(node_id, community_id) VALUES(?, ?)",
        (n, cid),
    )
    await db.commit()

    stats = await migrate_gm_to_kb(gm, reg, min_community_size=1,
                                       revive_threshold=0.95)
    assert stats["kbs_revived"] == 0
    assert stats["kbs_created"] == 1
    active_ids = {k["kb_id"] for k in await reg.list_kbs()}
    assert f"community_{cid}" in active_ids
