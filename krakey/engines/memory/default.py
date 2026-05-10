"""``GraphMemoryEngine`` — default impl of ``MemoryEngine``.

Subclass of ``GraphMemory`` (so the 20+ GM CRUD methods stay
directly reachable) plus three responsibilities layered on top:

  * **KB management** — ``create_kb`` / ``open_kb`` / ``list_kbs`` /
    ``set_archived`` / ``set_index_embedding`` / ``delete_kb`` /
    ``close_all_kbs`` — delegating to an internal ``KBRegistry`` built
    lazily during ``initialize()``.
  * **Sleep cycle** — ``sleep_cycle`` runs the full
    ``enter_sleep_mode`` pipeline (clustering → migration → KB
    consolidation/archival → index rebuild) without callers having to
    know that subsystem exists.

Initialize ordering: ``initialize()`` calls ``GraphMemory.initialize()``
first (opens the SQLite connection + applies schema), then constructs
the ``KBRegistry`` (which needs the now-open connection to read its
``kb_registry`` table). Calling KB methods before ``initialize()``
raises a clear error rather than NoneType-attribute errors.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from krakey.engines.memory._internal.graph_memory import GraphMemory
from krakey.interfaces.duck import ChatLike
from krakey.engines.memory._internal.knowledge_base import KBRegistry

if TYPE_CHECKING:
    from krakey.interfaces.engines.memory import KnowledgeBaseLike


class GraphMemoryEngine(GraphMemory):
    """Built-in ``MemoryEngine`` impl. Subclass of ``GraphMemory`` so
    every GM method (CRUD + search + edges + LLM-driven writers)
    stays available without forwarding boilerplate; KB management
    and sleep are added as new methods that delegate to an internal
    ``KBRegistry`` and the existing sleep pipeline.
    """

    def __init__(
        self,
        *,
        db_path: str,
        embedder,
        kb_dir: str,
        auto_ingest_threshold: float = 0.92,
        extractor_llm: ChatLike | None = None,
        classifier_llm: ChatLike | None = None,
        classify_batch_size: int = 10,
        classify_existing_context: int = 30,
    ):
        super().__init__(
            db_path,
            embedder=embedder,
            auto_ingest_threshold=auto_ingest_threshold,
            extractor_llm=extractor_llm,
            classifier_llm=classifier_llm,
            classify_batch_size=classify_batch_size,
            classify_existing_context=classify_existing_context,
        )
        self._kb_dir = kb_dir
        self._kb_registry: KBRegistry | None = None

    # ---- lifecycle -----------------------------------------------------

    async def initialize(self) -> None:
        """Open the SQLite connection (via GraphMemory) AND build the
        internal KBRegistry. The registry needs the now-initialized
        GM to back its ``kb_registry`` table reads."""
        await super().initialize()
        if self._kb_registry is None:
            self._kb_registry = KBRegistry(
                self, kb_dir=self._kb_dir, embedder=self._embedder,
            )

    async def close(self) -> None:
        """Close every open KB first, then the GM connection."""
        if self._kb_registry is not None:
            await self._kb_registry.close_all()
        await super().close()

    def _require_kb_registry(self) -> KBRegistry:
        if self._kb_registry is None:
            raise RuntimeError(
                "GraphMemoryEngine.initialize() must be called before "
                "KB management methods"
            )
        return self._kb_registry

    # ---- KB management (delegates to internal KBRegistry) -------------

    async def create_kb(
        self,
        kb_id: str,
        *,
        name: str,
        description: str = "",
        topics: list[str] | None = None,
    ) -> "KnowledgeBaseLike":
        return await self._require_kb_registry().create_kb(
            kb_id, name=name, description=description, topics=topics,
        )

    async def open_kb(self, kb_id: str) -> "KnowledgeBaseLike":
        return await self._require_kb_registry().open_kb(kb_id)

    async def list_kbs(
        self, *, include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        if self._kb_registry is None:
            return []
        return await self._kb_registry.list_kbs(
            include_archived=include_archived,
        )

    async def set_archived(self, kb_id: str, archived: bool) -> None:
        await self._require_kb_registry().set_archived(kb_id, archived)

    async def set_index_embedding(
        self, kb_id: str, embedding: list[float] | None,
    ) -> None:
        await self._require_kb_registry().set_index_embedding(
            kb_id, embedding,
        )

    async def delete_kb(self, kb_id: str) -> None:
        await self._require_kb_registry().delete_kb(kb_id)

    async def close_all_kbs(self) -> None:
        if self._kb_registry is not None:
            await self._kb_registry.close_all()

    # ---- sleep cycle ---------------------------------------------------

    async def sleep_cycle(
        self,
        *,
        channels: Any,
        log_dir: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a full sleep cycle. ``config`` carries the user's sleep
        tuning + the LLM/reranker the pipeline needs (since sleep
        clustering + migration use those Engines).

        Expected ``config`` keys (all optional, with sensible
        defaults from ``cfg.sleep``):

          * llm                          — chat client used by
                                            clustering summaries +
                                            sleep-time KB dedup judge
          * reranker                     — RerankerEngine used during
                                            sleep migration dedup
          * min_community_size           — drop tiny clusters
          * kb_consolidation_threshold   — pairwise KB merge threshold
          * kb_index_max                 — soft cap on active KB count
          * kb_archive_pct               — % of low-importance KBs to
                                            archive when over the cap
          * kb_revive_threshold          — revive an archived KB when
                                            new community is this close
        """
        from krakey.engines.memory._internal.sleep.sleep_manager import (
            enter_sleep_mode,
        )

        registry = self._require_kb_registry()
        return await enter_sleep_mode(
            self,
            registry,
            channels,
            llm=config.get("llm"),
            embedder=self._embedder,
            reranker=config.get("reranker"),
            log_dir=log_dir,
            min_community_size=config.get("min_community_size", 1),
            kb_consolidation_threshold=config.get(
                "kb_consolidation_threshold", 0.85,
            ),
            kb_index_max=config.get("kb_index_max", 30),
            kb_archive_pct=config.get("kb_archive_pct", 10),
            kb_revive_threshold=config.get("kb_revive_threshold", 0.80),
        )
