"""Graph Memory — middle-tier working memory (DevSpec §7).

This module is the **public facade**. The class ``GraphMemory``
combines three pieces, kept in their own files:

  * ``gm/storage.py``  — ``GMStorage``: SQLite connection lifecycle +
                          node CRUD + stats + upsert + name lookup +
                          category update + raw test escape hatch.
  * ``gm/query.py``    — ``GMQueryMixin``: vec_search / fts_search /
                          would_create_cycle / insert_edge_with_cycle_check /
                          get_neighbor_keywords / get_edges_among, all
                          delegating to ``krakey/memory/tools/`` (parameterized
                          primitives shared with KnowledgeBase).
  * ``writer.py``      — LLM-driven write strategies (auto_ingest /
                          explicit_write / classify_and_link_pending),
                          exposed here as facade methods so call sites
                          stay ``gm.method(...)``.

Callers continue to use ``from krakey.engines.memory._internal.graph_memory import GraphMemory``
and the usual instance methods. The split is purely internal.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from krakey.engines.memory._internal._db import cosine_similarity  # noqa: F401  re-export for tests
from krakey.engines.memory._internal.gm.query import GMQueryMixin
from krakey.engines.memory._internal.gm.storage import (  # noqa: F401  re-exports for callers
    GMStorage,
    _row_to_node,
)
from krakey.interfaces.duck import AsyncEmbedder, ChatLike


__all__ = [
    "GraphMemory", "AsyncEmbedder", "cosine_similarity",
    "_row_to_node",
]


class GraphMemory(GMStorage, GMQueryMixin):
    """Public facade for the GraphMemory subsystem.

    Inherits storage (CRUD + connection) from ``GMStorage`` and search /
    graph-walk methods from ``GMQueryMixin``. Adds the LLM-driven write
    facades directly so callers can keep using ``gm.auto_ingest(...)``
    etc. without importing ``writer`` themselves.
    """

    def __init__(self, db_path: str | Path, embedder: AsyncEmbedder,
                  *, auto_ingest_threshold: float = 0.92,
                  extractor_llm: ChatLike | None = None,
                  classifier_llm: ChatLike | None = None,
                  classify_batch_size: int = 10,
                  classify_existing_context: int = 30):
        super().__init__(db_path, embedder)
        # Writer-related config / dependencies live here, not on
        # GMStorage, since storage doesn't reach into LLM-driven
        # writes itself — only the facades below do.
        self._auto_ingest_threshold = auto_ingest_threshold
        self._extractor_llm = extractor_llm
        self._classifier_llm = classifier_llm
        self._classify_batch_size = classify_batch_size
        self._classify_existing_context = classify_existing_context

    # ---------- LLM-driven writes (impl in krakey/memory/writer.py) ----------

    async def auto_ingest(self, content: str,
                            *, source_heartbeat: int | None = None
                            ) -> dict[str, Any]:
        """Zero-LLM write — see ``src.memory.writer.auto_ingest``."""
        from krakey.engines.memory._internal import writer
        return await writer.auto_ingest(
            self, content, source_heartbeat=source_heartbeat,
        )

    async def explicit_write(self, content: str, *,
                               importance: str = "normal",
                               recall_context: list[dict[str, Any]] | None = None,
                               source_heartbeat: int | None = None
                               ) -> dict[str, Any]:
        """LLM-assisted write — see ``src.memory.writer.explicit_write``."""
        if self._extractor_llm is None:
            raise RuntimeError("explicit_write requires an extractor_llm")
        from krakey.engines.memory._internal import writer
        return await writer.explicit_write(
            self, content,
            extractor_llm=self._extractor_llm,
            importance=importance,
            recall_context=recall_context,
            source_heartbeat=source_heartbeat,
        )

    async def classify_and_link_pending(self) -> dict[str, int]:
        """Background classifier — see
        ``src.memory.writer.classify_and_link_pending``."""
        if self._classifier_llm is None:
            return {"classified": 0, "edges": 0}
        from krakey.engines.memory._internal import writer
        return await writer.classify_and_link_pending(
            self,
            classifier_llm=self._classifier_llm,
            batch_size=self._classify_batch_size,
            existing_context=self._classify_existing_context,
        )

    # ---------- Read façade (search + recall_context) ----------

    async def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        min_similarity: float = 0.3,
    ) -> list[tuple[dict[str, Any], float]]:
        """Scored free-text recall: embed -> vec_search, FTS fallback on
        embed failure / empty result / no embedder. ``top_k <= 0`` -> []."""
        if top_k <= 0:
            return []
        candidates: list[tuple[dict[str, Any], float]] = []
        if self._embedder is not None:
            try:
                vec = await self._embedder(query)
                candidates = await self.vec_search(
                    vec, top_k=top_k, min_similarity=min_similarity,
                )
            except Exception:  # noqa: BLE001
                candidates = []
        if not candidates:
            fts_hits = await self.fts_search(query, top_k=top_k)
            candidates = [(n, 0.0) for n in fts_hits]
        return candidates

    async def recall_context(self, node_ids: list[int]) -> dict[str, Any]:
        """Recall-time enrichment: neighbor keywords + edges among the set.
        Empty ``node_ids`` -> empty enrichment."""
        if not node_ids:
            return {"neighbor_keywords": {}, "edges": []}
        return {
            "neighbor_keywords": await self.get_neighbor_keywords(node_ids),
            "edges": await self.get_edges_among(node_ids),
        }

    async def remember_extraction(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Bulk store of already-distilled structure (nodes + edges).

        Each node dict ``{name, category, description, source_type?}`` is
        upserted (building a name→id map); malformed nodes (missing name or
        category) are skipped. Each edge dict
        ``{source_name, target_name, predicate}`` is resolved through the map
        (fallback ``find_by_name``) and inserted with a cycle check; edges
        with an unresolved endpoint or a self-loop are skipped. Integrity
        (id resolution, cycle checks) is enforced here so callers never touch
        raw node ids or edge primitives.

        Returns ``{"nodes_written": int, "edges_written": int}``.
        """
        name_to_id: dict[str, int] = {}
        nodes_written = 0
        for node in nodes:
            try:
                name = node.get("name")
                category = node.get("category")
                if not name or not category:
                    continue
                nid = await self.upsert_node({
                    "name": name,
                    "category": category,
                    "description": node.get("description", ""),
                    "source_type": node.get("source_type", "compact"),
                })
                name_to_id[name] = nid
                nodes_written += 1
            except Exception:
                continue

        edges_written = 0
        for edge in edges:
            try:
                src_name = edge.get("source_name")
                tgt_name = edge.get("target_name")
                if not src_name or not tgt_name:
                    continue
                src = name_to_id.get(src_name)
                if src is None:
                    src = await self.find_by_name(src_name)
                tgt = name_to_id.get(tgt_name)
                if tgt is None:
                    tgt = await self.find_by_name(tgt_name)
                if src is None or tgt is None or src == tgt:
                    continue
                await self.insert_edge_with_cycle_check(
                    src, tgt, edge.get("predicate", ""),
                )
                edges_written += 1
            except Exception:
                continue

        return {"nodes_written": nodes_written, "edges_written": edges_written}
