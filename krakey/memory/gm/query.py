"""GraphMemory query half — search + graph algorithm facades.

Mixin shape: every method needs ``self._require()`` (the storage's
SQLite connection) but holds no state of its own. Mixed into the
``GraphMemory`` facade alongside ``GMStorage``.

Each method is a 5-10 line wrapper around a generic primitive in
``krakey/memory/tools/``. The primitives take a ``(connection, table)``
pair so KnowledgeBase can use them too (KB uses vec_scan / fts_scan
directly today; the graph algorithms are not yet wired into KB but
the parameterization is ready for it).
"""
from __future__ import annotations

from typing import Any


class GMQueryMixin:
    """Adds search / graph methods to anything exposing
    ``self._require() -> aiosqlite.Connection`` (i.e. ``GMStorage``).

    Mixed into ``GraphMemory`` so call sites stay ``gm.vec_search(...)``
    / ``gm.would_create_cycle(...)`` while the implementations live in
    ``krakey/memory/tools/`` parameterized by table name.
    """

    # ---- vector search --------------------------------------------------

    async def vec_search(self, query_vec: list[float], *,
                           top_k: int = 5,
                           min_similarity: float = 0.0
                           ) -> list[tuple[dict[str, Any], float]]:
        """Brute-force python-side cosine over rows with embedding != NULL.

        Adequate for Phase 1 scale (≤ soft_limit nodes). Returns
        (node_dict, similarity) pairs sorted descending by similarity.
        """
        from krakey.memory.gm.storage import _row_to_node
        from krakey.memory.tools.vec_search import vec_scan
        return await vec_scan(
            self._require(), table="gm_nodes",
            query_vec=query_vec, row_decoder=_row_to_node,
            top_k=top_k, min_similarity=min_similarity,
        )

    # ---- FTS5 fallback --------------------------------------------------

    async def fts_search(self, query: str, *,
                           top_k: int = 5) -> list[dict[str, Any]]:
        """Full-text search fallback used when embeddings are unavailable.
        Tokens are sanitized so MATCH never sees FTS5 operators."""
        from krakey.memory.gm.storage import _row_to_node
        from krakey.memory.tools.fts_search import fts_scan
        return await fts_scan(
            self._require(), table="gm_nodes", fts_table="gm_nodes_fts",
            query=query, row_decoder=_row_to_node, top_k=top_k,
        )

    # ---- cycle-safe edges ----------------------------------------------

    async def would_create_cycle(self, a: int, b: int) -> bool:
        """Undirected connectivity check (DevSpec §7.7) — thin wrapper
        around the generic ``tools.graph`` walker."""
        from krakey.memory.tools.graph import would_create_cycle as _cycle
        return await _cycle(self._require(), edges_table="gm_edges",
                              a=a, b=b)

    async def insert_edge_with_cycle_check(self, src: int, tgt: int,
                                             predicate: str) -> dict[str, Any]:
        """Normalize (a<b) and skip the edge if it would close a cycle.
        Delegates to ``tools.graph.insert_edge_with_cycle_check``."""
        from krakey.memory.tools.graph import (
            insert_edge_with_cycle_check as _insert,
        )
        return await _insert(
            self._require(), edges_table="gm_edges",
            src=src, tgt=tgt, predicate=predicate,
        )

    # ---- neighbor expansion + edges among a set ------------------------

    async def get_neighbor_keywords(self, node_ids: list[int], *,
                                      depth: int = 1) -> dict[int, list[str]]:
        """For each node in `node_ids`, return a de-duplicated list of
        neighbor names (DevSpec §9.3 keyword hints). Phase 1 supports
        depth=1. Delegates to the generic graph walker."""
        from krakey.memory.tools.graph import (
            get_neighbor_keywords as _neighbors,
        )
        return await _neighbors(
            self._require(), nodes_table="gm_nodes",
            edges_table="gm_edges", node_ids=node_ids, depth=depth,
        )

    async def get_edges_among(self, node_ids: list[int]
                                ) -> list[dict[str, Any]]:
        """Return edges whose both endpoints are within `node_ids`,
        with source/target names. Delegates to the generic graph
        walker. Keys match DevSpec §3.6 Layer-2 renderer."""
        from krakey.memory.tools.graph import get_edges_among as _among
        return await _among(
            self._require(), nodes_table="gm_nodes",
            edges_table="gm_edges", node_ids=node_ids,
        )
