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

Callers continue to use ``from krakey.memory.graph_memory import GraphMemory``
and the usual instance methods. The split is purely internal.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from krakey.memory._db import cosine_similarity  # noqa: F401  re-export for tests
from krakey.memory.gm.query import GMQueryMixin
from krakey.memory.gm.storage import (  # noqa: F401  re-exports for callers
    AsyncEmbedder,
    GMStorage,
    _row_to_node,
)


__all__ = [
    "GraphMemory", "AsyncChatLLM", "AsyncEmbedder", "cosine_similarity",
    "_row_to_node",
]


class AsyncChatLLM(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...


class GraphMemory(GMStorage, GMQueryMixin):
    """Public facade for the GraphMemory subsystem.

    Inherits storage (CRUD + connection) from ``GMStorage`` and search /
    graph-walk methods from ``GMQueryMixin``. Adds the LLM-driven write
    facades directly so callers can keep using ``gm.auto_ingest(...)``
    etc. without importing ``writer`` themselves.
    """

    def __init__(self, db_path: str | Path, embedder: AsyncEmbedder,
                  *, auto_ingest_threshold: float = 0.92,
                  extractor_llm: AsyncChatLLM | None = None,
                  classifier_llm: AsyncChatLLM | None = None,
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
        from krakey.memory import writer
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
        from krakey.memory import writer
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
        from krakey.memory import writer
        return await writer.classify_and_link_pending(
            self,
            classifier_llm=self._classifier_llm,
            batch_size=self._classify_batch_size,
            existing_context=self._classify_existing_context,
        )
