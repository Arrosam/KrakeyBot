"""Phase 2.3b: GM → KB migration (DevSpec §11.3 Phase 3).

For each FACT/RELATION/KNOWLEDGE node in GM:
  - locate its community (run after clustering)
  - find or create a KB for that community
  - migrate the node into the KB as an entry
  - migrate intra-community edges (both endpoints in same community)
  - delete the node from GM (cascade removes other GM edges)

Cross-community edges are intentionally dropped here; the KB index graph
(Phase 2.3c) re-expresses inter-KB relations on a higher level.
"""
from __future__ import annotations

import json
from typing import Any

from src.memory.graph_memory import GraphMemory
from src.memory.knowledge_base import KBRegistry, KnowledgeBase


_MIGRATABLE = ("FACT", "RELATION", "KNOWLEDGE")


async def migrate_gm_to_kb(gm: GraphMemory, reg: KBRegistry
                              ) -> dict[str, int]:
    """Run the migration pass. Returns counters."""
    db = gm._require()  # noqa: SLF001
    counters = {"migrated_nodes": 0, "migrated_edges": 0,
                 "skipped_no_community": 0, "kbs_created": 0}

    nodes = await _fetch_migratable(db)
    if not nodes:
        return counters

    node_to_community = await _node_to_community_map(db)
    community_kbs: dict[int, KnowledgeBase] = {}
    gm_to_kb_entry: dict[int, tuple[int, int]] = {}  # gm_node_id → (kb_id, entry_id)

    # First pass: write entries
    for node in nodes:
        community_id = node_to_community.get(node["id"])
        if community_id is None:
            counters["skipped_no_community"] += 1
            continue
        kb = community_kbs.get(community_id)
        if kb is None:
            kb, created = await _find_or_create_kb_for_community(
                db, reg, community_id,
            )
            community_kbs[community_id] = kb
            if created:
                counters["kbs_created"] += 1

        entry_id = await kb.write_entry(
            node.get("description") or node["name"],
            tags=[node["category"], node["name"]],
            embedding=node.get("embedding"),
            source=f"gm_node:{node['id']}",
            importance=node.get("importance", 1.0),
        )
        gm_to_kb_entry[node["id"]] = (community_id, entry_id)
        counters["migrated_nodes"] += 1

    # Second pass: migrate intra-community edges
    counters["migrated_edges"] = await _migrate_edges(
        db, gm_to_kb_entry, community_kbs,
    )

    # Third pass: delete migrated GM nodes (cascade-deletes their edges)
    for gm_node_id in gm_to_kb_entry:
        await db.execute("DELETE FROM gm_nodes WHERE id=?", (gm_node_id,))
    await db.commit()

    return counters


async def _fetch_migratable(db) -> list[dict[str, Any]]:
    placeholders = ",".join("?" * len(_MIGRATABLE))
    async with db.execute(
        f"SELECT * FROM gm_nodes WHERE category IN ({placeholders})",
        list(_MIGRATABLE),
    ) as cur:
        rows = await cur.fetchall()
    # Avoid circular import at module load
    from src.memory.graph_memory import _row_to_node
    return [_row_to_node(r) for r in rows]


async def _node_to_community_map(db) -> dict[int, int]:
    """Return node_id → community_id (first community wins for nodes
    that ended up in multiple — Leiden usually gives one)."""
    out: dict[int, int] = {}
    async with db.execute(
        "SELECT node_id, community_id FROM gm_node_communities "
        "ORDER BY community_id ASC"
    ) as cur:
        async for row in cur:
            nid = int(row[0])
            cid = int(row[1])
            out.setdefault(nid, cid)
    return out


async def _find_or_create_kb_for_community(
    db, reg: KBRegistry, community_id: int,
) -> tuple[KnowledgeBase, bool]:
    async with db.execute(
        "SELECT name, summary FROM gm_communities WHERE community_id=?",
        (community_id,),
    ) as cur:
        row = await cur.fetchone()
    name = (row["name"] if row else None) or f"community {community_id}"
    summary = (row["summary"] if row else "") or name

    kb_id = f"community_{community_id}"
    try:
        kb = await reg.create_kb(kb_id, name=name, description=summary)
        return kb, True
    except ValueError:
        # Already exists from a prior sleep; reopen
        return await reg.open_kb(kb_id), False


async def _migrate_edges(db, gm_to_kb_entry: dict[int, tuple[int, int]],
                            community_kbs: dict[int, "KnowledgeBase"]) -> int:
    if not gm_to_kb_entry:
        return 0
    gm_ids = list(gm_to_kb_entry.keys())
    placeholders = ",".join("?" * len(gm_ids))
    async with db.execute(
        f"""
        SELECT node_a, node_b, predicate FROM gm_edges
        WHERE node_a IN ({placeholders}) AND node_b IN ({placeholders})
        """,
        gm_ids + gm_ids,
    ) as cur:
        rows = await cur.fetchall()

    migrated = 0
    for row in rows:
        a, b, pred = int(row[0]), int(row[1]), row[2]
        ca, ea = gm_to_kb_entry[a]
        cb, eb = gm_to_kb_entry[b]
        if ca != cb:
            continue  # cross-community → leave for index_rebuild
        kb = community_kbs[ca]
        info = await kb.write_edge(ea, eb, pred)
        if info["written"]:
            migrated += 1
    return migrated
