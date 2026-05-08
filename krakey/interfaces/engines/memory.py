"""``MemoryEngine`` — the unified memory Engine surface.

Replaces three previous Protocols (``MemoryService``, ``KBRegistryService``,
plus the ``enter_sleep_mode`` free function) with a single flat Protocol
that any compliant impl must satisfy. Rationale: from the heartbeat's
point of view "memory" is one thing — the runtime asks the Engine to
ingest, recall, and consolidate, and doesn't care that the default
impl partitions storage into a graph + KB fleet under the hood.

A user replacing ``memory`` with their own backend (Postgres, Redis,
remote service, etc.) provides one class implementing every method
below. The default impl ``GraphMemoryEngine`` keeps its internal
modules (gm/, kb/, sleep/) but exposes the flat surface.

``KnowledgeBaseLike`` is the per-KB instance shape returned by
``open_kb``. Kept as a separate Protocol because it's a value type
the Engine's methods produce, not the Engine itself.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class KnowledgeBaseLike(Protocol):
    """A single knowledge base instance, returned by
    ``MemoryEngine.open_kb``. Append-mostly long-form store; the default
    impl is a per-topic SQLite file under
    ``workspace/data/knowledge_bases/<kb_id>.sqlite``.
    """

    async def initialize(self) -> None: ...

    async def close(self) -> None: ...

    async def write_entry(
        self,
        content: str,
        *,
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
        source: str | None = None,
        importance: float = 1.0,
    ) -> int: ...

    async def merge_entry(
        self,
        entry_id: int,
        *,
        new_content: str,
        new_embedding: list[float] | None,
        new_importance: float,
        new_tags: list[str] | None,
    ) -> None: ...

    async def write_edge(
        self, entry_a: int, entry_b: int, predicate: str,
    ) -> dict[str, Any]: ...

    async def count_entries(self) -> int: ...

    async def list_active_entries(
        self, *, limit: int,
    ) -> list[dict[str, Any]]: ...

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_similarity: float = 0.3,
    ) -> list[dict[str, Any]]: ...

    async def vec_search(
        self,
        query_vec: list[float],
        *,
        top_k: int = 5,
        min_similarity: float = 0.5,
    ) -> list[tuple[dict[str, Any], float]]: ...


@runtime_checkable
class MemoryEngine(Protocol):
    """The memory slot's flat surface — three responsibilities in one
    Protocol:

      1. Working-memory graph: node + edge CRUD, vector + FTS search,
         LLM-driven write facades (``auto_ingest`` / ``explicit_write`` /
         ``classify_and_link_pending``).
      2. Long-term knowledge base fleet: ``create_kb`` / ``open_kb`` /
         ``list_kbs`` / archive + revive lifecycle.
      3. Sleep cycle: ``sleep_cycle()`` runs the full clustering →
         migration → KB consolidation/archival → index-rebuild
         pipeline. The default impl delegates to the
         ``engines/memory/sleep/`` subpackage; users replacing the
         Engine can implement their own consolidation strategy.

    A custom Engine must satisfy ALL methods. To stub one (e.g. an
    Engine that never sleeps), return empty data structures from the
    relevant calls — never raise NotImplementedError, since the
    heartbeat invokes them unconditionally.
    """

    # ---- lifecycle ----

    async def initialize(self) -> None:
        """One-shot setup: schema migrations, connection pools, etc.
        Called once before any other method."""
        ...

    async def close(self) -> None:
        """Shutdown: flush + close any open resources."""
        ...

    # ---- graph node CRUD ----

    async def upsert_node(self, node: dict[str, Any]) -> int: ...

    async def find_by_name(self, name: str) -> int | None: ...

    async def update_node_category(
        self, node_name: str, new_category: str,
    ) -> bool: ...

    async def list_nodes(
        self,
        *,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...

    async def count_nodes(self) -> int: ...

    async def count_by_category(self, category: str) -> int: ...

    async def counts_by_category(self) -> dict[str, int]: ...

    async def counts_by_source(self) -> dict[str, int]: ...

    async def delete_by_category(self, category: str) -> int: ...

    # ---- graph edge CRUD ----

    async def insert_edge_with_cycle_check(
        self, src: int, tgt: int, predicate: str,
    ) -> dict[str, Any]: ...

    async def list_edges_named(
        self, *, limit: int,
    ) -> list[dict[str, str]]: ...

    async def count_edges(self) -> int: ...

    async def get_neighbor_keywords(
        self, node_ids: list[int], *, depth: int = 1,
    ) -> dict[int, list[str]]: ...

    async def get_edges_among(
        self, node_ids: list[int],
    ) -> list[dict[str, Any]]: ...

    # ---- search ----

    async def vec_search(
        self,
        query_vec: list[float],
        *,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[tuple[dict[str, Any], float]]: ...

    async def fts_search(
        self, query: str, *, top_k: int = 5,
    ) -> list[dict[str, Any]]: ...

    # ---- LLM-driven writers ----

    async def auto_ingest(
        self, content: str, *, source_heartbeat: int | None = None,
    ) -> dict[str, Any]: ...

    async def explicit_write(
        self,
        content: str,
        *,
        importance: str = "normal",
        recall_context: list[dict[str, Any]] | None = None,
        source_heartbeat: int | None = None,
    ) -> dict[str, Any]: ...

    async def classify_and_link_pending(self) -> dict[str, int]: ...

    # ---- KB fleet management ----

    async def create_kb(
        self,
        kb_id: str,
        *,
        name: str,
        description: str = "",
        topics: list[str] | None = None,
    ) -> KnowledgeBaseLike: ...

    async def open_kb(self, kb_id: str) -> KnowledgeBaseLike: ...

    async def list_kbs(
        self, *, include_archived: bool = False,
    ) -> list[dict[str, Any]]: ...

    async def set_archived(self, kb_id: str, archived: bool) -> None: ...

    async def set_index_embedding(
        self, kb_id: str, embedding: list[float] | None,
    ) -> None: ...

    async def delete_kb(self, kb_id: str) -> None: ...

    async def close_all_kbs(self) -> None: ...

    # ---- sleep cycle ----

    async def sleep_cycle(
        self,
        *,
        channels: Any,
        log_dir: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a full sleep cycle. ``config`` carries the user's sleep
        tuning (min_community_size, kb_consolidation_threshold,
        kb_index_max, kb_archive_pct, kb_revive_threshold, etc.).
        ``channels`` is the StimulusBuffer (impl pauses non-urgent
        channels for the duration and resumes after). Returns a stats
        dict the heartbeat logs + publishes as SleepDoneEvent.

        A custom Engine that doesn't implement consolidation should
        return ``{}`` rather than raise — the runtime treats absent
        keys as zero counts.
        """
        ...
