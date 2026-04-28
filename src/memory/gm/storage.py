"""GraphMemory storage half — connection lifecycle + node CRUD + stats.

This is the SQLite layer: owns the ``aiosqlite`` connection, applies
the schema, knows the ``gm_nodes`` / ``gm_edges`` table layout, and
exposes CRUD + counting + small write helpers (upsert / find_by_name /
update_node_category).

What's NOT here:
  * Search / graph-walk facades — see ``gm/query.py`` (mixin
    delegating to ``src/memory/tools/``).
  * LLM-driven write strategies — see ``src/memory/writer.py``,
    invoked through the facade methods on ``GraphMemory`` itself.

``GMStorage`` is intentionally usable on its own (no LLM deps), so a
caller that only needs basic CRUD can construct one without ever
touching the writer-related fields. ``GraphMemory`` extends it with
those.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import aiosqlite

from src.memory._db import (
    apply_schema, decode_embedding as _decode_embedding,
    encode_embedding as _encode_embedding, open_db_with_vec,
)


class AsyncEmbedder(Protocol):
    async def __call__(self, text: str) -> list[float]: ...


def _row_to_node(row: aiosqlite.Row) -> dict[str, Any]:
    meta_raw = row["metadata"]
    metadata = json.loads(meta_raw) if meta_raw else {}
    return {
        "id": row["id"],
        "name": row["name"],
        "category": row["category"],
        "description": row["description"],
        "importance": row["importance"],
        "metadata": metadata,
        "embedding": _decode_embedding(row["embedding"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "last_accessed": row["last_accessed"],
        "access_count": row["access_count"],
        "source_heartbeat": row["source_heartbeat"],
        "source_type": row["source_type"],
    }


class GMStorage:
    def __init__(self, db_path: str | Path, embedder: AsyncEmbedder):
        self.db_path = str(db_path)
        self._embedder = embedder
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        if self._db is not None:
            return
        self._db = await open_db_with_vec(self.db_path)
        await apply_schema(self._db)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def _require(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError(
                "GraphMemory not initialized; call initialize() first"
            )
        return self._db

    # ---------- node CRUD ----------

    async def insert_node(self, *, name: str, category: str, description: str,
                           embedding: list[float] | None = None,
                           importance: float = 1.0,
                           source_type: str = "auto",
                           source_heartbeat: int | None = None,
                           metadata: dict | None = None) -> int:
        db = self._require()
        cur = await db.execute(
            "INSERT INTO gm_nodes(name, category, description, embedding, "
            "importance, source_type, source_heartbeat, metadata) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (name, category, description, _encode_embedding(embedding),
             importance, source_type, source_heartbeat,
             json.dumps(metadata) if metadata else None),
        )
        await db.commit()
        return cur.lastrowid

    async def get_node(self, node_id: int) -> dict[str, Any] | None:
        db = self._require()
        async with db.execute(
            "SELECT * FROM gm_nodes WHERE id=?", (node_id,)
        ) as cur:
            row = await cur.fetchone()
            return _row_to_node(row) if row else None

    async def list_nodes(self, *, category: str | None = None,
                          limit: int | None = None) -> list[dict[str, Any]]:
        db = self._require()
        sql = "SELECT * FROM gm_nodes"
        params: list[Any] = []
        if category is not None:
            sql += " WHERE category = ?"
            params.append(category)
        sql += " ORDER BY id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
            return [_row_to_node(r) for r in rows]

    async def delete_node(self, node_id: int) -> None:
        db = self._require()
        await db.execute("DELETE FROM gm_nodes WHERE id=?", (node_id,))
        await db.commit()

    async def count_nodes(self) -> int:
        db = self._require()
        async with db.execute("SELECT COUNT(*) FROM gm_nodes") as cur:
            row = await cur.fetchone()
            return int(row[0])

    async def count_by_category(self, category: str) -> int:
        db = self._require()
        async with db.execute(
            "SELECT COUNT(*) FROM gm_nodes WHERE category = ?", (category,),
        ) as cur:
            row = await cur.fetchone()
            return int(row[0])

    async def delete_by_category(self, category: str) -> int:
        """Delete all nodes of a single category. Returns count removed.

        Used by Sleep phase 5 (clear FOCUS) — kept inside GraphMemory so
        callers don't need to touch ``_require()`` or know the table
        layout.
        """
        n = await self.count_by_category(category)
        db = self._require()
        await db.execute(
            "DELETE FROM gm_nodes WHERE category = ?", (category,),
        )
        await db.commit()
        return n

    async def counts_by_category(self) -> dict[str, int]:
        """``{category: count}`` across all nodes — for stats dashboards."""
        db = self._require()
        async with db.execute(
            "SELECT category, COUNT(*) FROM gm_nodes GROUP BY category"
        ) as cur:
            rows = await cur.fetchall()
        return {r[0]: r[1] for r in rows}

    async def counts_by_source(self) -> dict[str, int]:
        """``{source_type: count}`` across all nodes — for stats dashboards."""
        db = self._require()
        async with db.execute(
            "SELECT source_type, COUNT(*) FROM gm_nodes GROUP BY source_type"
        ) as cur:
            rows = await cur.fetchall()
        return {r[0]: r[1] for r in rows}

    async def list_edges_named(
        self, *, limit: int,
    ) -> list[dict[str, str]]:
        """Edges resolved to ``{source, predicate, target}`` name triples
        (vs ``list_edges`` / ``get_edges_among`` which return numeric IDs).
        Used by the dashboard's GM-edges browser."""
        db = self._require()
        async with db.execute(
            "SELECT na.name AS source, e.predicate AS predicate, "
            "nb.name AS target FROM gm_edges e "
            "JOIN gm_nodes na ON na.id=e.node_a "
            "JOIN gm_nodes nb ON nb.id=e.node_b "
            "ORDER BY e.id ASC LIMIT ?", (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [{"source": r["source"], "target": r["target"],
                  "predicate": r["predicate"]} for r in rows]

    async def count_edges(self) -> int:
        db = self._require()
        async with db.execute("SELECT COUNT(*) FROM gm_edges") as cur:
            row = await cur.fetchone()
            return int(row[0])

    async def set_metadata(self, node_id: int, delta: dict) -> None:
        """Merge `delta` into the existing metadata JSON."""
        db = self._require()
        current = await self.get_node(node_id)
        if current is None:
            raise KeyError(f"no node id={node_id}")
        merged = {**current["metadata"], **delta}
        await db.execute(
            "UPDATE gm_nodes SET metadata=?, updated_at=CURRENT_TIMESTAMP "
            "WHERE id=?",
            (json.dumps(merged), node_id),
        )
        await db.commit()

    # ---------- upsert ----------

    async def upsert_node(self, node: dict[str, Any]) -> int:
        """Same (name, category) → update description/embedding + bump
        importance. Otherwise → new insert. Returns the node id.
        """
        db = self._require()
        name = node["name"]
        category = node["category"]
        async with db.execute(
            "SELECT id, importance FROM gm_nodes WHERE name=? AND category=?",
            (name, category),
        ) as cur:
            row = await cur.fetchone()

        if row is None:
            return await self.insert_node(
                name=name,
                category=category,
                description=node.get("description", ""),
                embedding=node.get("embedding"),
                importance=node.get("importance", 1.0),
                source_type=node.get("source_type", "auto"),
                source_heartbeat=node.get("source_heartbeat"),
                metadata=node.get("metadata"),
            )

        node_id = row["id"]
        new_importance = float(row["importance"]) + 0.5
        desc = node.get("description")
        emb = node.get("embedding")
        sets = ["importance = ?", "updated_at = CURRENT_TIMESTAMP"]
        params: list[Any] = [new_importance]
        if desc is not None:
            sets.append("description = ?")
            params.append(desc)
        if emb is not None:
            sets.append("embedding = ?")
            params.append(_encode_embedding(emb))
        params.append(node_id)
        await db.execute(
            f"UPDATE gm_nodes SET {', '.join(sets)} WHERE id = ?", params,
        )
        await db.commit()
        return node_id

    # ---------- name lookup + category update (used by writer.py) ----------

    async def find_by_name(self, name: str) -> int | None:
        db = self._require()
        async with db.execute(
            "SELECT id FROM gm_nodes WHERE name = ? LIMIT 1", (name,)
        ) as cur:
            row = await cur.fetchone()
            return int(row["id"]) if row else None

    async def update_node_category(self, node_name: str,
                                      new_category: str) -> bool:
        """Hypothalamus path: change category by name (e.g. TARGET → FACT).
        Returns True if a row was updated, False if name not found.
        """
        db = self._require()
        cur = await db.execute(
            "UPDATE gm_nodes SET category=?, updated_at=CURRENT_TIMESTAMP "
            "WHERE name=?", (new_category, node_name),
        )
        await db.commit()
        return cur.rowcount > 0

    # ---------- low-level escape hatch for tests ----------

    async def raw_fetchone(self, sql: str, params: tuple = ()):
        db = self._require()
        async with db.execute(sql, params) as cur:
            return await cur.fetchone()
