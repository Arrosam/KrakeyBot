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
from pathlib import Path
from typing import Any, Protocol

import aiosqlite

from src.memory._db import (
    apply_schema, build_fts_query, cosine_similarity, decode_embedding,
    encode_embedding, open_db_with_vec,
)
from src.memory.graph_memory import GraphMemory


class AsyncEmbedder(Protocol):
    async def __call__(self, text: str) -> list[float]: ...


def _row_to_entry(row: aiosqlite.Row) -> dict[str, Any]:
    tags_raw = row["tags"]
    return {
        "id": row["id"],
        "content": row["content"],
        "source": row["source"],
        "tags": json.loads(tags_raw) if tags_raw else [],
        "embedding": decode_embedding(row["embedding"]),
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
        self._db = await open_db_with_vec(self.path)
        await apply_schema(self._db)

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
             encode_embedding(embedding),
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

    async def list_active_entries(
        self, *, limit: int,
    ) -> list[dict[str, Any]]:
        """Active (is_active=1) entries, newest first, with tags decoded
        from JSON. Used by the dashboard's KB-entries browser — kept
        inside KnowledgeBase so callers don't need to touch
        ``_require()`` or know the table layout."""
        import json as _json
        db = self._require()
        async with db.execute(
            "SELECT id, content, source, tags, importance, created_at "
            "FROM kb_entries WHERE is_active = 1 ORDER BY id DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        out = []
        for r in rows:
            tags = _json.loads(r["tags"]) if r["tags"] else []
            out.append({"id": r["id"], "content": r["content"],
                          "source": r["source"], "tags": tags,
                          "importance": r["importance"],
                          "created_at": r["created_at"]})
        return out

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
        from src.memory.tools.vec_search import vec_scan
        scored = await vec_scan(
            self._require(), table="kb_entries",
            query_vec=query_vec, row_decoder=_row_to_entry,
            top_k=top_k, min_similarity=min_similarity,
            extra_where="is_active = 1",
        )
        return [e for (e, _s) in scored]

    async def fts_search(self, query: str, *,
                           top_k: int = 5) -> list[dict[str, Any]]:
        from src.memory.tools.fts_search import fts_scan
        return await fts_scan(
            self._require(), table="kb_entries", fts_table="kb_entries_fts",
            query=query, row_decoder=_row_to_entry, top_k=top_k,
            extra_where="kb_entries.is_active = 1",
        )


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

    async def list_kbs(self, *, include_archived: bool = False
                          ) -> list[dict[str, Any]]:
        gm_db = self.gm._require()  # noqa: SLF001
        sql = (
            "SELECT kb_id, name, path, description, topics, entry_count, "
            "is_archived, index_embedding, created_at, updated_at "
            "FROM kb_registry"
        )
        if not include_archived:
            sql += " WHERE COALESCE(is_archived, 0) = 0"
        async with gm_db.execute(sql) as cur:
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
                "is_archived": bool(r["is_archived"]),
                "index_embedding": decode_embedding(r["index_embedding"]),
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            })
        return out

    async def set_archived(self, kb_id: str, archived: bool) -> None:
        gm_db = self.gm._require()  # noqa: SLF001
        await gm_db.execute(
            "UPDATE kb_registry SET is_archived = ?, "
            "updated_at = CURRENT_TIMESTAMP WHERE kb_id = ?",
            (1 if archived else 0, kb_id),
        )
        await gm_db.commit()

    async def set_index_embedding(self, kb_id: str,
                                     embedding: list[float] | None) -> None:
        gm_db = self.gm._require()  # noqa: SLF001
        await gm_db.execute(
            "UPDATE kb_registry SET index_embedding = ? WHERE kb_id = ?",
            (encode_embedding(embedding), kb_id),
        )
        await gm_db.commit()

    async def delete_kb(self, kb_id: str) -> None:
        """Drop a KB completely: registry row, on-disk file, and any open
        connection. Caller is responsible for moving entries out first if
        they want to preserve them (consolidation does this)."""
        kb = self._open.pop(kb_id, None)
        if kb is not None:
            await kb.close()
        gm_db = self.gm._require()  # noqa: SLF001
        async with gm_db.execute(
            "SELECT path FROM kb_registry WHERE kb_id = ?", (kb_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return
        await gm_db.execute("DELETE FROM kb_registry WHERE kb_id = ?", (kb_id,))
        await gm_db.commit()
        try:
            Path(row["path"]).unlink()
        except (OSError, FileNotFoundError):
            pass

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
