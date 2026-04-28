"""Phase 1.3c: reranker contract.

`rerank()` is the pure reranker wrapper — it returns reranker-scored
pairs on success and ``None`` on any fail-soft condition (no
reranker, exception, score-count mismatch). Callers apply their own
fallback policy. The recall plugin's scripted_score fallback is
covered by the plugin-level tests in ``test_incremental_recall``.
"""
import pytest

from krakey.memory.recall import rerank


def _node(id_, name, *, description=""):
    return {"id": id_, "name": name, "description": description}


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

    ranked = await rerank(
        candidates, query="anything", reranker=reranker,
    )
    assert ranked is not None
    names = [n["name"] for (n, _score) in ranked]
    assert names == ["C", "B", "A"]
    assert reranker.calls[0][0] == "anything"
    assert len(reranker.calls[0][1]) == 3


async def test_reranker_raises_returns_none():
    candidates = [
        (_node(1, "low"), 0.1),
        (_node(2, "high"), 0.9),
    ]
    reranker = FailingReranker()
    out = await rerank(candidates, query="q", reranker=reranker)
    assert out is None


async def test_no_reranker_returns_none():
    candidates = [(_node(1, "a"), 0.5), (_node(2, "t"), 0.5)]
    out = await rerank(candidates, query="q", reranker=None)
    assert out is None


async def test_empty_candidates_returns_empty_list_not_none():
    """Empty in → empty out. No reranker call needed; not a failure."""
    out = await rerank([], query="q", reranker=None)
    assert out == []
    out = await rerank([], query="q", reranker=StubReranker(scores=[]))
    assert out == []


async def test_reranker_score_count_mismatch_returns_none():
    candidates = [(_node(1, "A"), 0.9), (_node(2, "B"), 0.8)]
    reranker = StubReranker(scores=[0.5])  # 1 score, expected 2
    out = await rerank(candidates, query="q", reranker=reranker)
    assert out is None
