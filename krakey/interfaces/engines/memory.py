"""``MemoryEngine`` — the minimal memory Engine surface.

A 12-method Protocol covering the three responsibilities the runtime asks
of memory, expressed at the highest useful abstraction:

  * **Lifecycle**: ``initialize`` / ``close`` — one-shot setup and shutdown.
  * **Storage** (write): ``ingest`` / ``remember`` / ``remember_extraction`` —
    passive incidental capture, deliberate agent-driven storage, and bulk
    distilled-structure ingestion respectively.
  * **Recall** (read): ``search`` / ``recall_context`` / ``recall_kb`` —
    free-text ranked retrieval, graph-enrichment for a result set, and
    named-KB lookup.
  * **Stats**: ``count_nodes`` / ``count_edges`` / ``counts_by_category`` /
    ``counts_by_source`` — lightweight counters for observability.

Graph node/edge CRUD, raw vector/FTS search primitives, KB fleet management
(create/open/list/archive/delete), and the sleep-cycle pipeline are all
engine-internal concerns. The concrete implementation keeps them; the
Protocol does not declare them. KB browsing and editing are served by the
memory engine's own web service, not by handing KB objects to callers.

A user replacing ``memory`` with their own backend provides one class
implementing every method below. The default impl ``GraphMemoryEngine``
partitions storage across internal modules (gm/, kb/, sleep/) but exposes
this flat surface.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MemoryEngine(Protocol):
    """Minimal 12-method surface for the memory engine slot.

    Custom Engine implementors must satisfy all 12 methods. To stub a
    method (e.g. an Engine that has no KB tier), return an empty data
    structure rather than raising ``NotImplementedError`` — the runtime
    invokes these unconditionally.
    """

    # ---- lifecycle ----

    async def initialize(self) -> None:
        """One-shot setup: schema migrations, connection pools, etc.
        Called once before any other method."""
        ...

    async def close(self) -> None:
        """Shutdown: flush + close any open resources."""
        ...

    # ---- storage ----

    async def ingest(
        self,
        content: str,
        *,
        source_heartbeat: int | None = None,
    ) -> dict[str, Any]:
        """Passive, low-cost store of incidental content (e.g. tool
        feedback). The engine decides internally whether/how to dedup,
        classify, and link — the caller has no control over that
        pipeline. Returns a stats dict (keys are engine-defined; callers
        must not branch logic on its contents beyond logging).
        """
        ...

    async def remember(
        self,
        content: str,
        *,
        importance: str = "normal",
        recall_context: list[dict[str, Any]] | None = None,
        source_heartbeat: int | None = None,
    ) -> dict[str, Any]:
        """Deliberate store of content the agent explicitly chose to
        remember. The engine may invoke an LLM internally to extract
        structure. ``recall_context`` is optional surrounding context
        (e.g. records from a prior ``search`` call) the engine may use
        for dedup and linking — passing it is never required. Returns a
        stats dict.
        """
        ...

    async def remember_extraction(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Bulk store of already-distilled structure, used by history
        compaction. ``nodes`` is a list of dicts with keys
        ``{name, category, description, source_type?}``; ``edges`` is a
        list of dicts with keys ``{source_name, target_name, predicate}``.
        The engine resolves names to internal ids and enforces its own
        integrity rules (cycle checks, dedup) INTERNALLY — callers never
        see node ids or edge primitives. Returns a stats dict such as
        ``{"nodes_written": int, "edges_written": int}``.
        """
        ...

    # ---- recall ----

    async def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        min_similarity: float = 0.3,
    ) -> list[tuple[dict[str, Any], float]]:
        """Return scored candidate memory records for a free-text query,
        best first. The engine internally chooses its retrieval strategy
        (vector search, full-text fallback, hybrid, …). Each element is
        ``(record_dict, relevance_score)`` where the score is opaque and
        higher means more relevant; an engine with no scoring notion
        returns ``0.0`` for every record. An empty list is a valid result.
        ``top_k=0`` always returns ``[]``.
        """
        ...

    async def recall_context(
        self,
        node_ids: list[int],
    ) -> dict[str, Any]:
        """Given a list of record ids obtained from ``search``, return
        recall-time enrichment:
        ``{"neighbor_keywords": dict[int, list[str]], "edges": list[dict]}``.
        ``neighbor_keywords`` maps each id to a list of related keyword
        hints; ``edges`` lists relationships among the given set as dicts
        ``{source, predicate, target}`` with resolved names. A non-graph
        backend returns ``{"neighbor_keywords": {}, "edges": []}`` and
        callers must tolerate empty enrichment. ``recall_context([])``
        returns empty enrichment without error.
        """
        ...

    async def recall_kb(
        self,
        kb_id: str,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Recall entries from a specific named knowledge base, ranked by
        relevance to ``query``. Returns a list of record dicts (engine-
        defined shape). Raises ``KeyError`` if no KB with ``kb_id``
        exists. An engine without a KB tier may always raise ``KeyError``
        (because no KB is ever named); callers must handle the miss
        gracefully.
        """
        ...

    # ---- stats ----

    async def count_nodes(self) -> int:
        """Return the total number of nodes in the working-memory graph."""
        ...

    async def count_edges(self) -> int:
        """Return the total number of edges in the working-memory graph."""
        ...

    async def counts_by_category(self) -> dict[str, int]:
        """Return a mapping of category name → node count for all
        categories that have at least one node."""
        ...

    async def counts_by_source(self) -> dict[str, int]:
        """Return a mapping of source type → node count for all source
        types that have at least one node."""
        ...
