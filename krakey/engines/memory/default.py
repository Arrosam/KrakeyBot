"""``GraphMemoryEngine`` — default impl of ``MemoryEngine``.

Subclass of ``GraphMemory`` (so the 20+ GM CRUD methods stay
directly reachable) plus three responsibilities layered on top:

  * **KB management** — ``create_kb`` / ``open_kb`` / ``list_kbs`` /
    ``set_archived`` / ``set_index_embedding`` / ``delete_kb`` /
    ``close_all_kbs`` — delegating to an internal ``KBRegistry`` built
    lazily during ``initialize()``.
  * **Sleep cycle** — ``request_sleep`` runs the full
    ``enter_sleep_mode`` pipeline (clustering → migration → KB
    consolidation/archival → index rebuild) using deps injected at
    construction time. The engine self-triggers sleep when the GM node
    count reaches ``sleep_config["auto_sleep_node_threshold"]``
    (0 = disabled). Channels are never paused.

Initialize ordering: ``initialize()`` calls ``GraphMemory.initialize()``
first (opens the SQLite connection + applies schema), then constructs
the ``KBRegistry`` (which needs the now-open connection to read its
``kb_registry`` table). Calling KB methods before ``initialize()``
raises a clear error rather than NoneType-attribute errors.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

from krakey.engines.memory._internal.graph_memory import GraphMemory
from krakey.interfaces.duck import ChatLike
from krakey.engines.memory._internal.knowledge_base import KBRegistry

# ``KnowledgeBaseLike`` is no longer a public Protocol (KB browsing/editing
# is served by the memory engine's own web service, not handed to callers).
# The KB instance type is an engine-internal concern; annotate as Any.
KnowledgeBaseLike = Any

logger = logging.getLogger(__name__)


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
        # Sleep deps — injected at construction, never per-call
        sleep_llm: ChatLike | None = None,
        reranker=None,
        sleep_config=None,
        sleep_log_dir: str = "workspace/logs",
        # Web service config — optional; enabled=False by default
        web_config=None,
        # Engine-own settings — passed by the registry at construction
        config=None,
        config_path=None,
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

        # Sleep deps
        self._sleep_llm = sleep_llm
        self._reranker = reranker
        # Normalise sleep_config into a plain dict
        if sleep_config is None:
            self._sleep_cfg: dict[str, Any] = {}
        elif dataclasses.is_dataclass(sleep_config) and not isinstance(sleep_config, type):
            self._sleep_cfg = dataclasses.asdict(sleep_config)
        else:
            self._sleep_cfg = dict(sleep_config)
        self._sleep_log_dir = sleep_log_dir

        # Public counter + in-flight guard
        self.sleep_cycles_run: int = 0
        self._sleeping: bool = False

        # Web service config — normalise to a plain dict (or None)
        if web_config is None:
            self._web_config: dict[str, Any] | None = None
        elif dataclasses.is_dataclass(web_config) and not isinstance(web_config, type):
            self._web_config = dataclasses.asdict(web_config)
        else:
            self._web_config = dict(web_config)
        self._web_server = None

        # Engine-own settings (from the engine's own settings file)
        self._config: dict[str, Any] = dict(config) if config else {}
        self._config_path: str | None = config_path

        # gm_node_soft_limit — memory-engine-owned threshold (replaces
        # the old config.fatigue.gm_node_soft_limit). Readable by the
        # runtime via engine.gm_node_soft_limit; default 1000.
        self.gm_node_soft_limit: int = int(
            (self._config or {}).get("gm_node_soft_limit", 1000)
        )

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

        # Start the self-hosted web service if enabled in web_config.
        # A startup failure MUST NOT crash initialize() — log + continue.
        cfg = self._web_config or {}
        _enabled = cfg.get("enabled", False)
        if _enabled:
            try:
                from krakey.engines.memory.web import create_memory_app
                from krakey.engines.memory.web_server import ThreadedMemoryWebServer
                _host = cfg.get("host", "127.0.0.1")
                _port = int(cfg.get("port", 8766))
                _app = create_memory_app(self)
                self._web_server = ThreadedMemoryWebServer(_app, host=_host, port=_port)
                self._web_server.start()
            except Exception as e:  # noqa: BLE001
                logger.warning("memory web service failed to start: %s", e)
                self._web_server = None

    async def close(self) -> None:
        """Stop the web server first, then close every open KB, then
        the GM connection."""
        if self._web_server is not None:
            try:
                self._web_server.stop()
            except Exception:  # noqa: BLE001
                pass
            self._web_server = None
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

    # ---- MemoryEngine Protocol methods (new minimal-surface API) ----------

    async def ingest(
        self,
        content: str,
        *,
        source_heartbeat: int | None = None,
    ) -> dict:
        """Passive, low-cost store. Delegates to ``auto_ingest``."""
        result = await self.auto_ingest(content, source_heartbeat=source_heartbeat)
        await self._maybe_auto_sleep()
        return result

    async def remember(
        self,
        content: str,
        *,
        importance: str = "normal",
        recall_context: list[dict] | None = None,
        source_heartbeat: int | None = None,
    ) -> dict:
        """Deliberate store with optional LLM extraction. Delegates to
        ``explicit_write``."""
        result = await self.explicit_write(
            content,
            importance=importance,
            recall_context=recall_context,
            source_heartbeat=source_heartbeat,
        )
        await self._maybe_auto_sleep()
        return result

    async def remember_extraction(
        self,
        nodes: list[dict],
        edges: list[dict],
    ) -> dict:
        """Bulk store of already-distilled structure (nodes + edges)."""
        result = await super().remember_extraction(nodes, edges)
        await self._maybe_auto_sleep()
        return result

    async def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        min_similarity: float = 0.3,
    ) -> list[tuple[dict, float]]:
        """Embed → vec_search with FTS fallback on embed failure or empty
        result. If no embedder is configured, goes straight to FTS.
        FTS hits receive score ``0.0``. ``top_k <= 0`` returns ``[]``."""
        return await super().search(query, top_k=top_k, min_similarity=min_similarity)

    async def recall_context(
        self,
        node_ids: list[int],
    ) -> dict:
        """Return recall-time enrichment for a set of node ids.

        Returns ``{"neighbor_keywords": {…}, "edges": […]}``. Empty inputs
        return empty enrichment without touching the DB."""
        return await super().recall_context(node_ids)

    async def recall_kb(
        self,
        kb_id: str,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[dict]:
        """Recall entries from a named knowledge base. Raises ``KeyError``
        if the KB does not exist."""
        kb = await self.open_kb(kb_id)
        return await kb.search(query, top_k=top_k)

    # ---- sleep (engine-owned) ------------------------------------------

    async def request_sleep(self, reason: str = "") -> dict[str, Any]:
        """The ONLY sleep entry point. Runs one consolidation cycle using
        the construction-injected sleep_llm / reranker / embedder /
        sleep_config. Does NOT pause channels.

        Return / raise contract (so callers can tell a real cycle from a
        no-op from a failure):
          * Returns a NON-EMPTY stats dict when a cycle actually ran.
          * Returns ``{}`` for a genuine NO-OP — no ``sleep_llm`` configured,
            or a cycle is already in flight (coalesced). Nothing happened.
          * RAISES on a real pipeline failure (clustering/migration/IO).
            The caller (``runtime.trigger_memory_sleep``) surfaces that as
            a SleepFailed event + a corrective stimulus to Self. We do NOT
            swallow it into ``{}`` — that would masquerade a crash as a
            successful (or no-op) cycle and, under force-sleep, loop forever.
        """
        if self._sleep_llm is None:
            return {}
        if self._sleeping:
            # Coalesce: a cycle is already in flight, do not stack
            return {}
        self._sleeping = True
        try:
            from krakey.engines.memory._internal.sleep.sleep_manager import (
                enter_sleep_mode,
            )
            cfg = self._sleep_cfg
            stats = await enter_sleep_mode(
                self,
                self._require_kb_registry(),
                channels=None,  # no channel pausing
                llm=self._sleep_llm,
                embedder=self._embedder,
                reranker=self._reranker,
                log_dir=self._sleep_log_dir,
                min_community_size=cfg.get("min_community_size", 1),
                kb_consolidation_threshold=cfg.get(
                    "kb_consolidation_threshold", 0.85,
                ),
                kb_index_max=cfg.get("kb_index_max", 30),
                kb_archive_pct=cfg.get("kb_archive_pct", 10),
                kb_revive_threshold=cfg.get("kb_revive_threshold", 0.80),
            )
            self.sleep_cycles_run += 1
            if reason:
                logger.debug(
                    "sleep cycle completed (reason=%r, cycles_run=%d)",
                    reason, self.sleep_cycles_run,
                )
            # enter_sleep_mode always returns a stats dict; guarantee a
            # truthy result so the caller sees "a cycle ran" even if the
            # pipeline had nothing to migrate.
            return stats or {"facts_migrated": 0, "completed": True}
        except Exception as exc:
            logger.error(
                "sleep cycle failed (reason=%r): %s", reason, exc, exc_info=True,
            )
            raise
        finally:
            self._sleeping = False

    async def _maybe_auto_sleep(self) -> None:
        """Trigger a sleep cycle when the GM node count reaches the
        configured threshold. threshold=0 (default) disables auto-sleep.

        Fired from the storage path (ingest/remember/remember_extraction),
        so a sleep failure must NEVER propagate up and break the write that
        triggered it — ``request_sleep`` now raises on real failures, so we
        catch + log here and let the storage op succeed regardless.
        """
        threshold = int(self._sleep_cfg.get("auto_sleep_node_threshold", 0) or 0)
        if threshold <= 0:
            return
        if self._sleeping or self._sleep_llm is None:
            return
        try:
            if await self.count_nodes() >= threshold:
                await self.request_sleep(reason="auto: node threshold")
        except Exception as exc:  # noqa: BLE001
            logger.warning("auto-sleep failed (non-fatal): %s", exc)
