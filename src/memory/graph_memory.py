"""Graph Memory — middle-tier working memory (DevSpec §7).

Phase 1.2a: init + basic node CRUD. Later sub-phases add upsert,
cycle-checked edges, auto_ingest, explicit_write, classify.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import aiosqlite
import sqlite_vec


SCHEMA_PATH = Path(__file__).parent / "schemas.sql"


class AsyncEmbedder(Protocol):
    async def __call__(self, text: str) -> list[float]: ...


def _encode_embedding(vec: list[float] | None) -> bytes | None:
    if vec is None:
        return None
    return json.dumps(list(vec)).encode("utf-8")


def _decode_embedding(blob: bytes | None) -> list[float] | None:
    if blob is None:
        return None
    return json.loads(blob.decode("utf-8"))


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


class GraphMemory:
    def __init__(self, db_path: str | Path, embedder: AsyncEmbedder):
        self.db_path = str(db_path)
        self._embedder = embedder
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        if self._db is not None:
            return
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        await db.enable_load_extension(True)
        await db.load_extension(sqlite_vec.loadable_path())
        await db.enable_load_extension(False)
        await db.execute("PRAGMA foreign_keys = ON")
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        await db.executescript(schema)
        await db.commit()
        self._db = db

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def _require(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("GraphMemory not initialized; call initialize() first")
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
            "UPDATE gm_nodes SET metadata=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (json.dumps(merged), node_id),
        )
        await db.commit()

    # ---------- low-level escape hatch for tests ----------

    async def raw_fetchone(self, sql: str, params: tuple = ()):
        db = self._require()
        async with db.execute(sql, params) as cur:
            return await cur.fetchone()
