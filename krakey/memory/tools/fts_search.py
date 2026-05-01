"""Generic FTS5 fallback search over a SQLite table.

Same pattern in GraphMemory and KnowledgeBase: build an FTS-safe
query, JOIN the FTS shadow table on rowid, ORDER BY rank. Only
table names + an optional extra filter differ.
"""
from __future__ import annotations

from typing import Any, Callable

import aiosqlite

from krakey.memory._db import build_fts_query


RowDecoder = Callable[[aiosqlite.Row], dict[str, Any]]


async def fts_scan(
    db: aiosqlite.Connection,
    *,
    table: str,
    fts_table: str,
    query: str,
    row_decoder: RowDecoder,
    top_k: int = 5,
    extra_where: str = "",
) -> list[dict[str, Any]]:
    """Run an FTS5 MATCH query against ``fts_table`` joined to ``table``
    on ``rowid``. ``extra_where`` is appended after ``AND`` (e.g.
    ``"is_active = 1"`` for KB). Empty string = no extra filter.

    Returns decoded rows in MATCH-rank order. An empty/sanitized-out
    query returns ``[]``.
    """
    fts_q = build_fts_query(query)
    if fts_q is None:
        return []
    sql = f"""
        SELECT {table}.*
        FROM {table}
        JOIN {fts_table} ON {fts_table}.rowid = {table}.id
        WHERE {fts_table} MATCH ?
    """
    if extra_where:
        sql += f" AND {extra_where}\n"
    sql += " ORDER BY rank LIMIT ?"
    async with db.execute(sql, (fts_q, top_k)) as cur:
        rows = await cur.fetchall()
    return [row_decoder(r) for r in rows]
