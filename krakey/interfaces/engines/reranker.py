"""``RerankerEngine`` — score-based reordering for recall + KB dedup.

Replaces the previous ``Reranker`` Protocol that lived in
``memory/recall/scoring.py``. Now an Engine rather than an optional
collaborator: every Engine slot must always have an impl, so the
default ``DefaultRerankerEngine`` embeds the scripted-scoring
fallback inside a single class:

  * If a reranker LLM tag is bound → call its ``rerank`` endpoint.
  * If unbound, or the call fails → fall back to scripted scoring
    (formula in ``engines/recall/scoring.py``).

This eliminates the previous tri-state (None / scripted-fallback /
real-LLM) at the call site — recall + sleep migration always get a
working reranker, the impl handles the strategy choice internally.

Users replacing the Engine can implement their own reranker (cohere,
local cross-encoder, learned-to-rank model, etc.). The Protocol just
asks for "score N docs against this query".
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RerankerEngine(Protocol):
    """Async reranker — returns one float score per doc, in input order.
    Higher = better. Implementations must always return ``len(docs)``
    floats; callers pair them with the candidates and re-sort
    descending.
    """

    async def rerank(self, query: str, docs: list[str]) -> list[float]: ...
