"""Generic brute-force cosine vector search over a SQLite table.

Both GraphMemory and KnowledgeBase do the same scan:
  ``SELECT * FROM <table> WHERE embedding IS NOT NULL [extra-filter]``
  → cosine each row → top_k by similarity.

Adequate for the Phase 1 scale targets (≤ a few thousand rows per
table). The Phase ? upgrade to ANN/HNSW would replace this module's
internals; the call sites stay the same.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

import aiosqlite

from krakey.memory._db import cosine_similarity


RowDecoder = Callable[[aiosqlite.Row], dict[str, Any]]


async def vec_scan(
    db: aiosqlite.Connection,
    *,
    table: str,
    query_vec: list[float],
    row_decoder: RowDecoder,
    top_k: int = 5,
    min_similarity: float = 0.0,
    extra_where: str = "",
) -> list[tuple[dict[str, Any], float]]:
    """Brute-force cosine over rows where ``embedding IS NOT NULL``.

    ``extra_where`` is appended verbatim after ``AND`` — pass e.g.
    ``"is_active = 1"`` for KB. Empty string = no extra filter.
    The decoded row dict MUST contain an ``"embedding"`` key with a
    decoded ``list[float]``; ``row_decoder`` is responsible.
    Returns ``(decoded_row, similarity)`` pairs, descending.
    """
    sql = f"SELECT * FROM {table} WHERE embedding IS NOT NULL"
    if extra_where:
        sql += f" AND {extra_where}"
    async with db.execute(sql) as cur:
        rows = await cur.fetchall()
    scored: list[tuple[dict[str, Any], float]] = []
    for row in rows:
        decoded = row_decoder(row)
        sim = cosine_similarity(query_vec, decoded["embedding"])
        if sim >= min_similarity:
            scored.append((decoded, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
