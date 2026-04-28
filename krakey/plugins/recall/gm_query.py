"""Shared GM-query primitive.

Both the auto-recall reflect (``incremental.py``) and the explicit
``memory_recall`` tentacle (``tentacle.py``) need the same low-level
operation: turn a query string into a list of GM nodes via vector
search, falling back to FTS when the embedder is down or vec_search
returns nothing. They each layer different orchestration on top —
the reflect accumulates across stimuli with weight-merge, the
tentacle takes the first batch and dedups for top-K — but the bottom
turn-text-into-candidates step is identical. Living here, used by
both, single source of truth.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from krakey.memory.graph_memory import GraphMemory
    from krakey.memory.recall import AsyncEmbedder


async def query_gm_with_fts_fallback(
    gm: "GraphMemory",
    embedder: "AsyncEmbedder",
    text: str,
    *,
    top_k: int,
    min_similarity: float = 0.3,
) -> list[tuple[dict[str, Any], float]]:
    """Embed → vec_search; FTS fallback on embed failure or empty result.

    Returns ``(node, similarity)`` tuples in the order the underlying
    search returned them. FTS hits get similarity ``0.0`` (FTS doesn't
    produce vector cosines); callers that don't need the score can
    drop it.
    """
    candidates: list[tuple[dict[str, Any], float]] = []
    try:
        vec = await embedder(text)
        candidates = await gm.vec_search(
            vec, top_k=top_k, min_similarity=min_similarity,
        )
    except Exception:  # noqa: BLE001
        candidates = []
    if not candidates:
        fts_hits = await gm.fts_search(text, top_k=top_k)
        candidates = [(n, 0.0) for n in fts_hits]
    return candidates
