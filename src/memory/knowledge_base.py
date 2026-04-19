"""Knowledge Base — long-term per-topic stores (DevSpec §8).

Each KB is a separate SQLite file under workspace/data/knowledge_bases/.
Schema (kb_meta, kb_entries, kb_edges, kb_entries_fts) lives in schemas.sql
and is applied at KB initialize time. The GM-side `kb_registry` table
tracks all KBs so Sleep + recall can enumerate them without scanning disk.

KBs are read-mostly: writes happen during Sleep (or test setup); recall
queries them via vec_search → FTS5 fallback (same pattern as GM).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

import aiosqlite
import sqlite_vec

from src.memory.graph_memory import (
    GraphMemory, SCHEMA_PATH, _decode_embedding, _encode_embedding,
    cosine_similarity,
)


class AsyncEmbedder(Protocol):
    async def __call__(self, text: str) -> list[float]: ...


# --- helpers shared with GraphMemory; lifted into _db.py during refactor A ---

_FTS_TOKEN = re.compile(r"\w+", re.UNICODE)


def _build_fts_query(text: str) -> str | None:
    tokens = _FTS_TOKEN.findall(text or "")
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


async def _open_db_with_vec(path: str | Path) -> aiosqlite.Connection:
    db = await aiosqlite.connect(str(path))
    db.row_factory = aiosqlite.Row
    await db.enable_load_extension(True)
    await db.load_extension(sqlite_vec.loadable_path())
    await db.enable_load_extension(False)
    await db.execute("PRAGMA foreign_keys = ON")
    return db


def _row_to_entry(row: aiosqlite.Row) -> dict[str, Any]:
    tags_raw = row["tags"]
    return {
        "id": row["id"],
        "content": row["content"],
        "source": row["source"],
        "tags": json.loads(tags_raw) if tags_raw else [],
        "embedding": _decode_embedding(row["embedding"]),
        "importance": row["importance"],
        "created_at": row["created_at"],
        "last_accessed": row["last_accessed"],
        "access_count": row["access_count"],
        "is_active": bool(row["is_active"]),
    }


# ---------------------------------------------------------------------------
# KnowledgeBase — one file, one connection, one topic cluster
# ---------------------------------------------------------------------------


class KnowledgeBase:
    def __init__(self, kb_id: str, file_path: str | Path,
                  embedder: AsyncEmbedder):
        self.kb_id = kb_id
        self.path = Path(file_path)
        self._embedder = embedder
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        if self._db is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await _open_db_with_vec(self.path)
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        await self._db.executescript(schema)
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def _require(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError(f"KB '{self.kb_id}' not initialized")
        return self._db

    # ---------- entries ----------

    async def write_entry(self, content: str, *,
                            tags: list[str] | None = None,
                            embedding: list[float] | None = None,
                            source: str | None = None,
                            importance: float = 1.0) -> int:
        db = self._require()
        cur = await db.execute(
            "INSERT INTO kb_entries(content, source, tags, embedding, "
            "importance) VALUES(?, ?, ?, ?, ?)",
            (content, source,
             json.dumps(tags) if tags else None,
             _encode_embedding(embedding),
             importance),
        )
        await db.commit()
        return cur.lastrowid

    async def get_entry(self, entry_id: int) -> dict[str, Any] | None:
        db = self._require()
        async with db.execute(
            "SELECT * FROM kb_entries WHERE id=?", (entry_id,)
        ) as cur:
            row = await cur.fetchone()
            return _row_to_entry(row) if row else None

    async def count_entries(self) -> int:
        db = self._require()
        async with db.execute("SELECT COUNT(*) FROM kb_entries") as cur:
            row = await cur.fetchone()
            return int(row[0])

    # ---------- edges ----------

    async def write_edge(self, entry_a: int, entry_b: int,
                           predicate: str) -> dict[str, Any]:
        if entry_a == entry_b:
            raise ValueError("self-loop edges not allowed")
        a, b = (entry_a, entry_b) if entry_a < entry_b else (entry_b, entry_a)
        db = self._require()
        try:
            await db.execute(
                "INSERT INTO kb_edges(entry_a, entry_b, predicate) "
                "VALUES(?, ?, ?)", (a, b, predicate),
            )
        except Exception:  # noqa: BLE001
            return {"written": False, "reason": "duplicate or constraint"}
        await db.commit()
        return {"written": True}

    # ---------- search ----------

    async def search(self, query: str, *, top_k: int = 5,
                       min_similarity: float = 0.3) -> list[dict[str, Any]]:
        """Embed → vec_search; fall back to FTS when embedder fails or
        returns nothing matching threshold."""
        try:
            vec = await self._embedder(query)
            results = await self._vec_search(
                vec, top_k=top_k, min_similarity=min_similarity,
            )
        except Exception:  # noqa: BLE001
            results = []
        if results:
            return results
        return await self.fts_search(query, top_k=top_k)

    async def _vec_search(self, query_vec: list[float], *,
                            top_k: int,
                            min_similarity: float) -> list[dict[str, Any]]:
        db = self._require()
        async with db.execute(
            "SELECT * FROM kb_entries WHERE embedding IS NOT NULL "
            "AND is_active = 1"
        ) as cur:
            rows = await cur.fetchall()
        scored: list[tuple[dict[str, Any], float]] = []
        for row in rows:
            entry = _row_to_entry(row)
            sim = cosine_similarity(query_vec, entry["embedding"])
            if sim >= min_similarity:
                scored.append((entry, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for (e, _s) in scored[:top_k]]

    async def fts_search(self, query: str, *,
                           top_k: int = 5) -> list[dict[str, Any]]:
        fts_q = _build_fts_query(query)
        if fts_q is None:
            return []
        db = self._require()
        async with db.execute(
            f"""
            SELECT kb_entries.*
            FROM kb_entries
            JOIN kb_entries_fts ON kb_entries_fts.rowid = kb_entries.id
            WHERE kb_entries_fts MATCH ?
              AND kb_entries.is_active = 1
            ORDER BY rank
            LIMIT ?
            """,
            (fts_q, top_k),
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_entry(r) for r in rows]


# ---------------------------------------------------------------------------
# KBRegistry — manages set of KBs via gm.kb_registry table
# ---------------------------------------------------------------------------


class KBRegistry:
    def __init__(self, gm: GraphMemory, kb_dir: str | Path,
                  embedder: AsyncEmbedder):
        self.gm = gm
        self.kb_dir = Path(kb_dir)
        self.embedder = embedder
        self._open: dict[str, KnowledgeBase] = {}

    async def create_kb(self, kb_id: str, *, name: str,
                          description: str = "",
                          topics: list[str] | None = None
                          ) -> KnowledgeBase:
        if kb_id in self._open:
            raise ValueError(f"KB '{kb_id}' already exists")
        # Check disk + registry
        existing = await self._fetch_meta(kb_id)
        if existing is not None:
            raise ValueError(f"KB '{kb_id}' already registered")

        path = self.kb_dir / f"{kb_id}.sqlite"
        kb = KnowledgeBase(kb_id, path, embedder=self.embedder)
        await kb.initialize()
        self._open[kb_id] = kb

        gm_db = self.gm._require()  # noqa: SLF001 — controlled internal access
        await gm_db.execute(
            "INSERT INTO kb_registry(kb_id, name, path, description, topics) "
            "VALUES(?, ?, ?, ?, ?)",
            (kb_id, name, str(path), description,
             json.dumps(topics) if topics else None),
        )
        await gm_db.commit()
        return kb

    async def open_kb(self, kb_id: str) -> KnowledgeBase:
        if kb_id in self._open:
            return self._open[kb_id]
        meta = await self._fetch_meta(kb_id)
        if meta is None:
            raise KeyError(f"no KB registered with id '{kb_id}'")
        kb = KnowledgeBase(kb_id, Path(meta["path"]), embedder=self.embedder)
        await kb.initialize()
        self._open[kb_id] = kb
        return kb

    async def list_kbs(self) -> list[dict[str, Any]]:
        gm_db = self.gm._require()  # noqa: SLF001
        async with gm_db.execute(
            "SELECT kb_id, name, path, description, topics, entry_count, "
            "created_at, updated_at FROM kb_registry"
        ) as cur:
            rows = await cur.fetchall()
        out = []
        for r in rows:
            out.append({
                "kb_id": r["kb_id"],
                "name": r["name"],
                "path": r["path"],
                "description": r["description"],
                "topics": json.loads(r["topics"]) if r["topics"] else [],
                "entry_count": r["entry_count"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            })
        return out

    async def _fetch_meta(self, kb_id: str) -> dict[str, Any] | None:
        gm_db = self.gm._require()  # noqa: SLF001
        async with gm_db.execute(
            "SELECT kb_id, name, path, description, topics FROM kb_registry "
            "WHERE kb_id = ?", (kb_id,),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def close_all(self) -> None:
        for kb in list(self._open.values()):
            await kb.close()
        self._open.clear()
