"""DefaultRerankerEngine — forwards to LLM client when available,
falls back to preserve-order scores otherwise.

Three fallback paths: no client bound, client raises, client returns
the wrong shape. All three produce identical preserve-order scores so
callers' stable sort by descending score keeps input ordering."""
from __future__ import annotations

import pytest

from krakey.engines.reranker.default import DefaultRerankerEngine
from krakey.interfaces.engines import RerankerEngine


class _FakeFactory:
    def __init__(self, client):
        self._client = client

    def client_for_tag(self, tag_name): return None
    def client_for_core_purpose(self, purpose): return None
    def embed_client(self): return None
    def rerank_client(self): return self._client


class _GoodClient:
    async def rerank(self, query: str, docs: list[str]) -> list[float]:
        # Return reverse-of-position scores so we can detect the
        # forwarding worked.
        return [float(len(docs) - i) * 0.1 for i in range(len(docs))]


class _RaisingClient:
    async def rerank(self, query, docs):
        raise RuntimeError("boom")


class _WrongShapeClient:
    async def rerank(self, query, docs):
        return [0.5]  # wrong length, regardless of docs


def test_satisfies_reranker_engine_protocol():
    eng = DefaultRerankerEngine(factory=_FakeFactory(None))
    assert isinstance(eng, RerankerEngine)


@pytest.mark.asyncio
async def test_forwards_to_client_when_available():
    eng = DefaultRerankerEngine(factory=_FakeFactory(_GoodClient()))
    out = await eng.rerank("q", ["a", "b", "c"])
    assert out == pytest.approx([0.3, 0.2, 0.1])


@pytest.mark.asyncio
async def test_fallback_when_no_client():
    eng = DefaultRerankerEngine(factory=_FakeFactory(None))
    out = await eng.rerank("q", ["a", "b", "c"])
    # Strictly decreasing — preserve-order under stable sort
    assert out == [3.0, 2.0, 1.0]


@pytest.mark.asyncio
async def test_fallback_when_client_raises():
    eng = DefaultRerankerEngine(factory=_FakeFactory(_RaisingClient()))
    out = await eng.rerank("q", ["a", "b"])
    assert out == [2.0, 1.0]


@pytest.mark.asyncio
async def test_fallback_when_client_wrong_length():
    eng = DefaultRerankerEngine(factory=_FakeFactory(_WrongShapeClient()))
    out = await eng.rerank("q", ["a", "b", "c"])
    assert out == [3.0, 2.0, 1.0]


@pytest.mark.asyncio
async def test_empty_docs_returns_empty_list():
    eng = DefaultRerankerEngine(factory=_FakeFactory(_GoodClient()))
    assert await eng.rerank("q", []) == []
