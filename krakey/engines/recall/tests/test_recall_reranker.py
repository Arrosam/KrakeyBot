"""Reranker fail-soft contract — exercised through IncrementalRecall.

After the recall-engine cleanup that pulled scoring helpers private
and dropped the standalone ``rerank()`` wrapper, the
"reranker present → reorder; reranker missing/fails → scripted
fallback" contract is owned by the caller. ``IncrementalRecall``
inlines the fallback inside ``_rerank_or_fallback``; these tests
exercise that helper directly with stub rerankers + a stub
MemoryEngine to keep the surface narrow.
"""
from datetime import datetime

import pytest

from krakey.engines.recall._internal.scoring import ScoringWeights
from krakey.engines.recall._internal.incremental import IncrementalRecall


def _node(id_, name, *, description="", category="FACT", access_count=0,
           importance=1.0, created_at="2026-01-01 00:00:00"):
    return {
        "id": id_, "name": name, "description": description,
        "category": category, "access_count": access_count,
        "importance": importance, "created_at": created_at,
    }


class _StubMemory:
    """Minimal MemoryEngine stand-in — exercises the rerank-fallback
    helper without touching SQLite. The helper doesn't call any of
    these methods (it operates on the candidate list it's given), so
    they're stubs."""
    async def vec_search(self, *_, **__): return []
    async def fts_search(self, *_, **__): return []
    async def get_neighbor_keywords(self, *_, **__): return {}
    async def get_edges_among(self, *_, **__): return []


def _make_recall(reranker=None) -> IncrementalRecall:
    return IncrementalRecall(
        _StubMemory(),
        embedder=lambda _t: [],  # unused by _rerank_or_fallback
        per_stimulus_k=10,
        recall_token_budget=10000,
        weights=ScoringWeights(),
        reranker=reranker,
        now=lambda: datetime(2026, 1, 1, 0, 0, 0),
    )


class _StubReranker:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    async def rerank(self, query, docs):
        self.calls.append((query, list(docs)))
        return list(self.scores)


class _FailingReranker:
    async def rerank(self, query, docs):
        raise ConnectionError("reranker unavailable")


async def test_reranker_available_reorders_by_its_scores():
    r = _make_recall(reranker=_StubReranker(scores=[0.1, 0.5, 0.95]))
    candidates = [
        (_node(1, "A"), 0.9),
        (_node(2, "B"), 0.8),
        (_node(3, "C"), 0.7),
    ]
    ranked = await r._rerank_or_fallback("anything", candidates)
    names = [n["name"] for (n, _score) in ranked]
    assert names == ["C", "B", "A"]


async def test_reranker_raises_falls_back_to_scripted_score():
    """Reranker exception → scripted multi-axis score (vec_sim
    dominates with the default weights, so highest sim wins)."""
    r = _make_recall(reranker=_FailingReranker())
    candidates = [
        (_node(1, "low",  category="FACT"), 0.1),
        (_node(2, "high", category="FACT"), 0.9),
    ]
    ranked = await r._rerank_or_fallback("q", candidates)
    names = [n["name"] for (n, _score) in ranked]
    assert names == ["high", "low"]


async def test_no_reranker_falls_back_to_scripted_score():
    """No reranker bound → scripted fallback, same shape."""
    r = _make_recall(reranker=None)
    candidates = [
        (_node(1, "low",  category="FACT"), 0.2),
        (_node(2, "high", category="FACT"), 0.95),
    ]
    ranked = await r._rerank_or_fallback("q", candidates)
    names = [n["name"] for (n, _score) in ranked]
    assert names == ["high", "low"]


async def test_empty_candidates_returns_empty_list():
    r = _make_recall(reranker=None)
    assert await r._rerank_or_fallback("q", []) == []
    r2 = _make_recall(reranker=_StubReranker(scores=[]))
    assert await r2._rerank_or_fallback("q", []) == []


async def test_reranker_score_count_mismatch_falls_back():
    """1 score returned for 2 candidates → fail-soft to scripted."""
    r = _make_recall(reranker=_StubReranker(scores=[0.5]))
    candidates = [
        (_node(1, "A", category="FACT"), 0.4),
        (_node(2, "B", category="FACT"), 0.8),
    ]
    ranked = await r._rerank_or_fallback("q", candidates)
    names = [n["name"] for (n, _score) in ranked]
    # B's higher vec_sim wins under the scripted formula.
    assert names == ["B", "A"]
