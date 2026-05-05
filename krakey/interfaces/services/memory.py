"""Memory service Protocols for the GraphMemory + KBRegistry slots.

Three Protocols, layered:

  * ``MemoryService`` — what ``krakey.memory.graph_memory.GraphMemory``
    satisfies. Surfaces every method called from outside ``graph_memory``
    (heartbeat hot path, sleep pipeline, dashboard, recall plugin).
  * ``KnowledgeBaseLike`` — what a KB instance (returned by
    ``KBRegistryService.open_kb``) satisfies.
  * ``KBRegistryService`` — what
    ``krakey.memory.knowledge_base.registry.KBRegistry`` satisfies.
    Methods that return KB instances are typed as
    ``KnowledgeBaseLike``.

Phase 3 of the slot rollout — these Protocols ship in their own commit
(no wiring) so users can start writing alternative backends against
them while subsequent commits decouple internal callers + add the
resolver-backed instantiation in Runtime's composition root.

## Scope notes

The Protocols cover **externally-called** surface only:
  * Methods used purely inside ``graph_memory.py`` are NOT on
    ``MemoryService`` — they're implementation details GraphMemory's
    own internals use, not contracts the runtime depends on.
  * ``GraphMemory._require()`` (returns the aiosqlite connection) is
    consumed by ``entry_store.py`` and ``registry.py`` — those
    components are tightly coupled to GraphMemory's SQLite backing
    by design and use the concrete class directly. They don't pass
    through this Protocol; that's a separate refactor (tracked in
    docs/plans/melodic-beaming-llama.md as a later phase).

## LLM-driven facades

``auto_ingest``, ``explicit_write``, and ``classify_and_link_pending``
internally combine the embedder + an extractor LLM + GM writes. A user
replacing the memory slot must implement these too (or stub them as
no-ops if their use case doesn't include LLM-driven memory growth).
The runtime's heartbeat will call them every beat by default.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MemoryService(Protocol):
    """The GraphMemory slot's contract — what every external caller
    expects from the memory backing store.

    Implementations may use SQLite (the built-in default), Postgres,
    Redis, in-memory dicts, or anything else. KrakeyBot doesn't care
    about persistence shape as long as these methods behave per their
    docstrings.
    """

    # ---- lifecycle ----

    async def initialize(self) -> None:
        """Called once at runtime startup before any other method.

        Schema migrations, connection pool setup, index warmup —
        anything one-shot that must precede the heartbeat.
        """
        ...

    async def close(self) -> None:
        """Called at runtime shutdown. Flush + close connections."""
        ...

    # ---- node CRUD ----

    async def upsert_node(self, node: dict[str, Any]) -> int:
        """Insert or update a node by name; return its node_id.

        ``node`` must contain at least ``name`` (str), ``category``
        (one of FACT|RELATION|KNOWLEDGE|TARGET|FOCUS), ``description``
        (str), ``source_type`` (str). Embedding is computed if absent.
        """
        ...

    async def find_by_name(self, name: str) -> int | None:
        """Return the node_id of the node with the given name, or
        None if no such node exists. Exact match (case-sensitive)."""
        ...

    async def update_node_category(
        self, node_name: str, new_category: str,
    ) -> bool:
        """Reassign a node to a different category. Returns True on
        success, False if the node doesn't exist."""
        ...

    async def list_nodes(
        self, *, category: str | None = None, limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List nodes, optionally filtered by category and capped."""
        ...

    async def count_nodes(self) -> int: ...

    async def counts_by_category(self) -> dict[str, int]: ...

    async def counts_by_source(self) -> dict[str, int]: ...

    async def delete_by_category(self, category: str) -> int:
        """Bulk delete by category; return number of nodes deleted."""
        ...

    # ---- edge CRUD ----

    async def insert_edge_with_cycle_check(
        self, src: int, tgt: int, predicate: str,
    ) -> dict[str, Any]:
        """Insert an edge ``src --predicate--> tgt`` if it would not
        create a cycle. Returns a dict describing the result (the
        edge if inserted; an explanation otherwise — runtime callers
        rely on the dict shape, see graph_memory.py for the contract)."""
        ...

    async def list_edges_named(
        self, *, limit: int,
    ) -> list[dict[str, str]]: ...

    async def count_edges(self) -> int: ...

    async def get_neighbor_keywords(
        self, node_ids: list[int], *, depth: int = 1,
    ) -> dict[int, list[str]]:
        """For each node_id, return a list of names of its 1..depth
        graph neighbors. Used by the recall pipeline to surface
        related concepts."""
        ...

    async def get_edges_among(
        self, node_ids: list[int],
    ) -> list[dict[str, Any]]:
        """Return edges where BOTH endpoints are in ``node_ids``."""
        ...

    # ---- search ----

    async def vec_search(
        self, query_vec: list[float], *,
        top_k: int = 5, min_similarity: float = 0.0,
    ) -> list[tuple[dict[str, Any], float]]:
        """Vector-similarity search. Returns ``(node, similarity)``
        pairs sorted descending; pairs below ``min_similarity`` are
        filtered out."""
        ...

    async def fts_search(
        self, query: str, *, top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Full-text search fallback, used when embedding fails or
        the query is short."""
        ...

    # ---- LLM-driven facades ----

    async def auto_ingest(
        self, content: str, *,
        source_heartbeat: int | None = None,
    ) -> dict[str, Any]:
        """Embed + dedupe + insert content as a node. Triggers async
        classification. Called every beat from tool_feedback signals.
        Returns ``{"node_id": int, "merged": bool, ...}``."""
        ...

    async def explicit_write(
        self, content: str, *,
        importance: str = "normal",
        recall_context: list[dict[str, Any]] | None = None,
        source_heartbeat: int | None = None,
    ) -> dict[str, Any]:
        """Self-initiated write ('I want to remember X'). Same shape
        as ``auto_ingest`` but with importance + provenance hints."""
        ...

    async def classify_and_link_pending(self) -> dict[str, int]:
        """Run the async classifier on any nodes that haven't been
        categorized yet, and create edges the classifier suggests.
        Called once per heartbeat. Returns ``{"classified": N,
        "linked": M}``."""
        ...


@runtime_checkable
class KnowledgeBaseLike(Protocol):
    """A single knowledge base, returned by ``KBRegistryService.open_kb``.

    KBs are append-mostly stores of long-form knowledge, separate from
    the working-memory graph. Sleep migrates clusters from GM into KBs;
    recall searches both.
    """

    async def initialize(self) -> None: ...

    async def close(self) -> None: ...

    async def write_entry(
        self, content: str, *,
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
        source: str | None = None,
        importance: float = 1.0,
    ) -> int: ...

    async def merge_entry(
        self, entry_id: int, *,
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
        self, query: str, *,
        top_k: int = 5, min_similarity: float = 0.3,
    ) -> list[dict[str, Any]]: ...

    async def vec_search(
        self, query_vec: list[float], *,
        top_k: int = 5, min_similarity: float = 0.5,
    ) -> list[tuple[dict[str, Any], float]]: ...


@runtime_checkable
class KBRegistryService(Protocol):
    """Manages multiple KnowledgeBase instances. The runtime owns one
    registry; sleep + plugins go through it for any KB access."""

    async def create_kb(
        self, kb_id: str, *,
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

    async def close_all(self) -> None: ...
