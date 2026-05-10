"""TagBoundEmbedderEngine — pulls embed_client from the factory and
forwards. Lazy-fails when no embedding tag bound."""
from __future__ import annotations

import pytest

from krakey.engines.embedder.default import TagBoundEmbedderEngine
from krakey.interfaces.engines import EmbedderEngine


class _FakeFactory:
    """Stand-in for LLMClientFactoryEngine — we only test
    ``embed_client()`` here."""

    def __init__(self, client):
        self._client = client

    def client_for_tag(self, tag_name): return None
    def client_for_core_purpose(self, purpose): return None
    def embed_client(self): return self._client
    def rerank_client(self): return None


class _FakeClient:
    def __init__(self):
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [1.0, 2.0, 3.0]


def test_satisfies_embedder_engine_protocol():
    eng = TagBoundEmbedderEngine(factory=_FakeFactory(_FakeClient()))
    assert isinstance(eng, EmbedderEngine)


@pytest.mark.asyncio
async def test_forwards_to_factory_embed_client():
    client = _FakeClient()
    eng = TagBoundEmbedderEngine(factory=_FakeFactory(client))
    out = await eng("hello world")
    assert out == [1.0, 2.0, 3.0]
    assert client.calls == ["hello world"]


@pytest.mark.asyncio
async def test_raises_when_no_embedding_tag_bound():
    eng = TagBoundEmbedderEngine(factory=_FakeFactory(client=None))
    with pytest.raises(RuntimeError, match="no embedding tag bound"):
        await eng("anything")
