"""Generic graph algorithms over an edges table (parameterized).

GraphMemory has used these from day one; KnowledgeBase has a
``kb_edges`` table but didn't have access to the same primitives.
Pulling them out as table-name-parameterized helpers means KB can
adopt cycle-safe edge insertion + neighbor walking when the next
KB feature needs them, without copy-pasting SQL.

The edges table schema both targets share:
  ``CREATE TABLE <edges>(node_a INTEGER, node_b INTEGER, predicate TEXT, ...)``
with the convention that ``node_a < node_b`` (undirected, normalized).
"""
from __future__ import annotations

from typing import Any

import aiosqlite


async def would_create_cycle(
    db: aiosqlite.Connection,
    *,
    edges_table: str,
    a: int,
    b: int,
) -> bool:
    """Undirected connectivity check: would inserting an edge between
    ``a`` and ``b`` close a cycle (i.e. is ``b`` already reachable
    from ``a`` in the existing graph)?

    Self-loop (a == b) returns ``True`` so callers can short-circuit.
    """
    if a == b:
        return True
    async with db.execute(
        f"""
        WITH RECURSIVE walk(nid, visited) AS (
            SELECT ?, CAST(? AS TEXT)
            UNION ALL
            SELECT CASE WHEN e.node_a = w.nid THEN e.node_b ELSE e.node_a END,
                   w.visited || ',' ||
                   CAST(CASE WHEN e.node_a = w.nid THEN e.node_b
                             ELSE e.node_a END AS TEXT)
            FROM walk w
            JOIN {edges_table} e ON (e.node_a = w.nid OR e.node_b = w.nid)
            WHERE INSTR(
                ',' || w.visited || ',',
                ',' || CAST(CASE WHEN e.node_a = w.nid THEN e.node_b
                                 ELSE e.node_a END AS TEXT) || ','
            ) = 0
        )
        SELECT 1 FROM walk WHERE nid = ? LIMIT 1
        """,
        (a, str(a), b),
    ) as cur:
        row = await cur.fetchone()
        return row is not None


async def insert_edge_with_cycle_check(
    db: aiosqlite.Connection,
    *,
    edges_table: str,
    src: int,
    tgt: int,
    predicate: str,
) -> dict[str, Any]:
    """Normalize ``(a < b)`` and insert; skip with reason when the edge
    would close a cycle or duplicate an existing (a,b,predicate) row.

    Phase 1 dropped the procedural bridge-node mechanism: each generated
    bridge could itself become part of a fresh cycle, leading to runaway
    chains. Skipping is information-lossy but cheaply preserves the
    acyclic invariant. A future phase may reintroduce LLM-extracted
    *semantic* intermediate nodes.

    Returns ``{"skipped": bool, "reason": str | None}``.
    """
    if src == tgt:
        raise ValueError("self-loop edges not allowed")
    a, b = (src, tgt) if src < tgt else (tgt, src)
    if await would_create_cycle(db, edges_table=edges_table, a=a, b=b):
        return {"skipped": True, "reason": "would create cycle"}
    try:
        await db.execute(
            f"INSERT INTO {edges_table}(node_a, node_b, predicate) "
            f"VALUES(?, ?, ?)",
            (a, b, predicate),
        )
    except Exception:  # noqa: BLE001
        return {"skipped": True, "reason": "duplicate edge"}
    await db.commit()
    return {"skipped": False, "reason": None}


async def get_neighbor_keywords(
    db: aiosqlite.Connection,
    *,
    nodes_table: str,
    edges_table: str,
    node_ids: list[int],
    depth: int = 1,
) -> dict[int, list[str]]:
    """For each id in ``node_ids``, return a de-duplicated list of
    neighbor *names* (names only, as keyword hints). Phase 1 supports
    ``depth=1``.

    Caller's nodes table must expose ``id`` and ``name`` columns.
    """
    if not node_ids:
        return {}
    placeholders = ",".join("?" * len(node_ids))
    async with db.execute(
        f"""
        SELECT center.id AS center_id, neighbor.name AS neighbor_name
        FROM {nodes_table} AS center
        JOIN {edges_table} AS e
          ON (e.node_a = center.id OR e.node_b = center.id)
        JOIN {nodes_table} AS neighbor
          ON neighbor.id = CASE WHEN e.node_a = center.id
                                THEN e.node_b ELSE e.node_a END
        WHERE center.id IN ({placeholders})
        """,
        list(node_ids),
    ) as cur:
        rows = await cur.fetchall()
    out: dict[int, list[str]] = {nid: [] for nid in node_ids}
    seen: dict[int, set[str]] = {nid: set() for nid in node_ids}
    for row in rows:
        cid = row["center_id"]
        name = row["neighbor_name"]
        if name not in seen[cid]:
            seen[cid].add(name)
            out[cid].append(name)
    return out


async def get_edges_among(
    db: aiosqlite.Connection,
    *,
    nodes_table: str,
    edges_table: str,
    node_ids: list[int],
) -> list[dict[str, Any]]:
    """Return edges whose both endpoints are within ``node_ids``, with
    source/target node names included for prompt rendering. Keys:
    ``predicate``, ``source_id``, ``source``, ``target_id``, ``target``.
    """
    if not node_ids:
        return []
    placeholders = ",".join("?" * len(node_ids))
    async with db.execute(
        f"""
        SELECT e.predicate AS predicate,
               na.id AS source_id, na.name AS source,
               nb.id AS target_id, nb.name AS target
        FROM {edges_table} AS e
        JOIN {nodes_table} AS na ON na.id = e.node_a
        JOIN {nodes_table} AS nb ON nb.id = e.node_b
        WHERE e.node_a IN ({placeholders})
          AND e.node_b IN ({placeholders})
        """,
        list(node_ids) + list(node_ids),
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]
