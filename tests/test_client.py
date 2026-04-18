import pytest

from src.llm.client import LLMClient
from src.models.config import Provider


class FakeTransport:
    """Replaces HTTP calls. Records requests, returns canned JSON."""

    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    async def post_json(self, url, headers, json_body):
        self.calls.append({"url": url, "headers": headers, "body": json_body})
        return self.response


def _openai_provider():
    return Provider(type="openai_compatible", base_url="http://server:8080",
                    api_key="k1", models=[])


def _anthropic_provider():
    return Provider(type="anthropic", base_url="https://api.anthropic.com",
                    api_key="ak-x", models=[])


async def test_chat_openai_compatible():
    t = FakeTransport({"choices": [{"message": {"content": "hello!"}}]})
    c = LLMClient(_openai_provider(), model="qwen", transport=t)
    out = await c.chat([{"role": "user", "content": "hi"}])
    assert out == "hello!"
    assert t.calls[0]["url"].endswith("/v1/chat/completions")
    assert t.calls[0]["headers"]["Authorization"] == "Bearer k1"
    assert t.calls[0]["body"]["model"] == "qwen"
    assert t.calls[0]["body"]["messages"][0]["content"] == "hi"


async def test_chat_string_input_wrapped_as_user_message():
    t = FakeTransport({"choices": [{"message": {"content": "ok"}}]})
    c = LLMClient(_openai_provider(), model="qwen", transport=t)
    await c.chat("just a prompt string")
    assert t.calls[0]["body"]["messages"] == [{"role": "user", "content": "just a prompt string"}]


async def test_chat_anthropic():
    t = FakeTransport({"content": [{"type": "text", "text": "hi there"}]})
    c = LLMClient(_anthropic_provider(), model="claude", transport=t)
    out = await c.chat([{"role": "user", "content": "hello"}])
    assert out == "hi there"
    assert t.calls[0]["url"].endswith("/v1/messages")
    assert t.calls[0]["headers"]["x-api-key"] == "ak-x"
    assert "anthropic-version" in t.calls[0]["headers"]


async def test_embed_returns_float_list():
    t = FakeTransport({"data": [{"embedding": [0.1, 0.2, 0.3]}]})
    c = LLMClient(_openai_provider(), model="bge-m3", transport=t)
    v = await c.embed("text")
    assert v == [0.1, 0.2, 0.3]
    assert t.calls[0]["url"].endswith("/v1/embeddings")
    assert t.calls[0]["body"]["input"] == "text"


async def test_rerank_returns_scores():
    t = FakeTransport({"results": [
        {"index": 0, "relevance_score": 0.9},
        {"index": 1, "relevance_score": 0.1},
    ]})
    c = LLMClient(_openai_provider(), model="bge-rerank", transport=t)
    scores = await c.rerank("query", ["doc a", "doc b"])
    assert scores == [0.9, 0.1]
    assert t.calls[0]["url"].endswith("/v1/rerank")


async def test_chat_no_api_key_omits_authorization():
    p = Provider(type="openai_compatible", base_url="http://x", api_key=None, models=[])
    t = FakeTransport({"choices": [{"message": {"content": "y"}}]})
    c = LLMClient(p, model="m", transport=t)
    await c.chat("hi")
    assert "Authorization" not in t.calls[0]["headers"]
