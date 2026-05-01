"""Single Knowledge Base file — long-term per-topic store (DevSpec §8).

One ``KnowledgeBase`` instance owns one SQLite file under
``workspace/data/knowledge_bases/<kb_id>.sqlite``. Schema (kb_meta /
kb_entries / kb_edges / kb_entries_fts) lives in ``schemas.sql`` and
is applied at ``initialize()`` time.

Read-mostly: writes happen during Sleep migration; recall queries
go vec_search → FTS5 fallback (same pattern as GM).

A KB has no knowledge of the registry that owns it. The registry
(``KBRegistry`` in ``registry.py``) handles fleet-level concerns:
which KBs exist, archive flags, on-disk paths, index vectors.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import aiosqlite

from krakey.memory._db import (
    apply_schema, build_fts_query, decode_embedding, encode_embedding,
    open_db_with_vec,
)


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

    async def merge_entry(self, entry_id: int, *,
                            new_content: str,
                            new_embedding: list[float] | None,
                            new_importance: float,
                            new_tags: list[str] | None) -> None:
        """Replace ``entry_id``'s content/embedding/tags; **sum** the
        importance with the existing row's. Used by the sleep-migration
        dedup pass when an LLM judge confirms a GM node describes the
        same thing as an existing KB entry. Tags are union-deduped so
        each merge preserves provenance (category + source-name tag from
        every contributing GM node).
        """
        db = self._require()
        async with db.execute(
            "SELECT importance, tags FROM kb_entries WHERE id=?", (entry_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise KeyError(f"entry {entry_id} not found in KB '{self.kb_id}'")
        existing_imp = float(row["importance"])
        existing_tags = json.loads(row["tags"]) if row["tags"] else []
        merged_tags = sorted(set(existing_tags) | set(new_tags or []))
        summed_imp = existing_imp + new_importance
        await db.execute(
            "UPDATE kb_entries SET content=?, embedding=?, tags=?, "
            "importance=? WHERE id=?",
            (new_content, encode_embedding(new_embedding),
             json.dumps(merged_tags) if merged_tags else None,
             summed_imp, entry_id),
        )
        await db.commit()

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
        db = self._require()
        async with db.execute(
            "SELECT id, content, source, tags, importance, created_at "
            "FROM kb_entries WHERE is_active = 1 ORDER BY id DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        out = []
        for r in rows:
            tags = json.loads(r["tags"]) if r["tags"] else []
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
            scored = await self.vec_search(
                vec, top_k=top_k, min_similarity=min_similarity,
            )
            results = [e for (e, _s) in scored]
        except Exception:  # noqa: BLE001
            results = []
        if results:
            return results
        return await self.fts_search(query, top_k=top_k)

    async def vec_search(self, query_vec: list[float], *,
                           top_k: int = 5,
                           min_similarity: float = 0.5
                           ) -> list[tuple[dict[str, Any], float]]:
        """Brute-force python-side cosine over active entries. Same
        contract as ``gm.vec_search``: returns ``(entry, similarity)``
        pairs sorted descending. Used by ``KnowledgeBase.search`` (the
        string-query path) and by the sleep-migration dedup pass
        (which already has a pre-computed embedding).
        """
        from krakey.memory.tools.vec_search import vec_scan
        return await vec_scan(
            self._require(), table="kb_entries",
            query_vec=query_vec, row_decoder=_row_to_entry,
            top_k=top_k, min_similarity=min_similarity,
            extra_where="is_active = 1",
        )

    async def fts_search(self, query: str, *,
                           top_k: int = 5) -> list[dict[str, Any]]:
        from krakey.memory.tools.fts_search import fts_scan
        return await fts_scan(
            self._require(), table="kb_entries", fts_table="kb_entries_fts",
            query=query, row_decoder=_row_to_entry, top_k=top_k,
            extra_where="kb_entries.is_active = 1",
        )
