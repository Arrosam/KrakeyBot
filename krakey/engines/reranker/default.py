"""``DefaultRerankerEngine`` — Reranker Engine with internal fallback.

User requirement: every Engine slot must always be populated; no
None / Optional escape hatch. So this default impl handles the three
"reranker not actually available" cases internally, returning a
deterministic score list every time:

  1. No reranker tag bound (``cfg.llm.reranker`` empty).
  2. The upstream client raises during ``rerank``.
  3. The upstream client returns the wrong number of scores.

In any of those cases the Engine returns *preserve-order scores* —
strictly decreasing floats so that callers performing
``paired.sort(key=lambda x: x[1], reverse=True)`` end up with the
input order intact. This is the lightest fallback that satisfies the
``rerank(query, docs) -> list[float]`` contract without LLM /
heuristic guesswork.

The richer "scripted multi-axis scoring" (vec_sim + time-decay +
access-count + importance + category weight) is recall-time scoring,
NOT reranker-time. It already runs upstream of the reranker call;
when no reranker is configured the recall pipeline keeps that
scripted ranking and the reranker effectively becomes a stable noop.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from krakey.interfaces.engines.llm_factory import LLMClientFactoryEngine


class DefaultRerankerEngine:
    """Reranker Engine that always returns scores — falls back to
    preserve-order scoring when the configured client is unavailable
    or misbehaves."""

    def __init__(self, *, factory: "LLMClientFactoryEngine"):
        self._factory = factory

    async def rerank(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        client = self._factory.rerank_client()
        if client is None:
            return self._fallback_scores(len(docs))
        try:
            scores = await client.rerank(query, docs)
        except Exception:  # noqa: BLE001 — fallback covers all failures
            return self._fallback_scores(len(docs))
        if not isinstance(scores, list) or len(scores) != len(docs):
            return self._fallback_scores(len(docs))
        return [float(s) for s in scores]

    @staticmethod
    def _fallback_scores(n: int) -> list[float]:
        """Strictly decreasing scores [n, n-1, ..., 1]. Stable-sort by
        these preserves input order, which is the right behaviour
        when we have no signal to reorder by."""
        return [float(n - i) for i in range(n)]
