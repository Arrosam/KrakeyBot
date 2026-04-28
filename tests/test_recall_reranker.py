"""Phase 1.3c: reranker integration + fallback to scripted sort."""
from datetime import datetime

import pytest

from krakey.memory.recall import ScoringWeights, rank_candidates


def _node(id_, name, *, category="FACT", importance=1.0,
            access_count=0, created_at="2026-04-19 00:00:00",
            description=""):
    return {
        "id": id_, "name": name, "category": category,
        "description": description, "importance": importance,
        "access_count": access_count, "created_at": created_at,
    }


class StubReranker:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    async def rerank(self, query, docs):
        self.calls.append((query, list(docs)))
        return list(self.scores)


class FailingReranker:
    async def rerank(self, query, docs):
        raise ConnectionError("reranker unavailable")


async def test_reranker_available_reorders_by_its_scores():
    candidates = [
        (_node(1, "A"), 0.9),
        (_node(2, "B"), 0.8),
        (_node(3, "C"), 0.7),
    ]
    # Reranker says C best, A worst
    reranker = StubReranker(scores=[0.1, 0.5, 0.95])
    now = datetime(2026, 4, 19)

    ranked = await rank_candidates(
        candidates, query="anything", reranker=reranker,
        weights=ScoringWeights(), now=now,
    )
    names = [n["name"] for (n, _score) in ranked]
    assert names == ["C", "B", "A"]
    assert reranker.calls[0][0] == "anything"
    assert len(reranker.calls[0][1]) == 3


async def test_reranker_failure_falls_back_to_scripted_sort():
    """Vec sim ordering should survive when reranker raises."""
    candidates = [
        (_node(1, "low", importance=1.0), 0.1),
        (_node(2, "high", importance=1.0), 0.9),
        (_node(3, "mid", importance=1.0), 0.5),
    ]
    reranker = FailingReranker()
    now = datetime(2026, 4, 19)

    ranked = await rank_candidates(
        candidates, query="q", reranker=reranker,
        weights=ScoringWeights(), now=now,
    )
    names = [n["name"] for (n, _score) in ranked]
    # Scripted score is dominated by vec_sim with equal other factors
    assert names == ["high", "mid", "low"]


async def test_no_reranker_uses_scripted_sort():
    candidates = [
        (_node(1, "a", category="FACT"), 0.5),
        (_node(2, "t", category="TARGET"), 0.5),
    ]
    now = datetime(2026, 4, 19)

    ranked = await rank_candidates(
        candidates, query="q", reranker=None,
        weights=ScoringWeights(), now=now,
    )
    # TARGET has higher category weight → comes first
    assert ranked[0][0]["name"] == "t"


async def test_empty_candidates_returns_empty():
    now = datetime(2026, 4, 19)
    out = await rank_candidates(
        [], query="q", reranker=None, weights=ScoringWeights(), now=now,
    )
    assert out == []


async def test_reranker_score_count_mismatch_falls_back():
    candidates = [(_node(1, "A"), 0.9), (_node(2, "B"), 0.8)]
    # Returns wrong number of scores → fall back
    reranker = StubReranker(scores=[0.5])  # only 1, expected 2
    now = datetime(2026, 4, 19)

    ranked = await rank_candidates(
        candidates, query="q", reranker=reranker,
        weights=ScoringWeights(), now=now,
    )
    # Falls back → still 2 items, ordered by vec_sim
    names = [n["name"] for (n, _s) in ranked]
    assert names == ["A", "B"]
