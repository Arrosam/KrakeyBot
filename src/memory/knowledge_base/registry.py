"""KB fleet manager — talks to GraphMemory's ``kb_registry`` table.

Owns:
  * Which KBs exist (by id) and where they live on disk.
  * Archive flags + index vectors (cosine search over KB summaries).
  * Lazy ``KnowledgeBase`` instance cache (``open_kb`` / ``close_all``).

Does NOT own KB internals — that's ``entry_store.KnowledgeBase``.
The registry pokes at the ``kb_registry`` table (which lives inside
the GM SQLite, not in any KB file) and treats each KB as opaque.
Sleep migrates entries between KBs by going through ``KnowledgeBase``
APIs; the registry never reads kb_entries.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.memory._db import decode_embedding, encode_embedding
from src.memory.graph_memory import GraphMemory
from src.memory.knowledge_base.entry_store import (
    AsyncEmbedder, KnowledgeBase,
)


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
