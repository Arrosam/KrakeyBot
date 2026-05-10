"""Shared GM-query primitive for the recall Engine + memory_recall tool.

Both ``IncrementalRecall`` (the per-beat recall driver in this Engine
package) and the ``memory_recall`` tool plugin (Self-driven explicit
recall) need the same low-level operation: turn a query string into a
list of GM nodes via vector search, falling back to FTS when the
embedder is down or vec_search returns nothing. They each layer
different orchestration on top — the engine accumulates across stimuli
with weight-merge, the tool takes the first batch and dedups for top-K
— but the bottom turn-text-into-candidates step is identical.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from krakey.interfaces.engines.memory import MemoryEngine
    from krakey.llm.resolve import AsyncEmbedder


async def query_gm_with_fts_fallback(
    memory: "MemoryEngine",
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
        candidates = await memory.vec_search(
            vec, top_k=top_k, min_similarity=min_similarity,
        )
    except Exception:  # noqa: BLE001
        candidates = []
    if not candidates:
        fts_hits = await memory.fts_search(text, top_k=top_k)
        candidates = [(n, 0.0) for n in fts_hits]
    return candidates
