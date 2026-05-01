import pytest

from krakey.llm.client import LLMClient, TransportError
from krakey.models.config import LLMParams, Provider


class FakeTransport:
    """Replaces HTTP calls. Records requests, returns canned JSON."""

    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    async def post_json(self, url, headers, json_body):
        self.calls.append({"url": url, "headers": headers, "body": json_body})
        return self.response


class SequenceTransport:
    """Returns a scripted sequence of results (dicts) and exceptions
    (TransportError). Used to test retry behavior."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    async def post_json(self, url, headers, json_body):
        self.calls.append({"url": url, "headers": headers, "body": json_body})
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


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


# ---------------- Params translation ----------------


async def test_openai_body_applies_params():
    """Universal params (max_output_tokens, temperature,
    stop_sequences, response_format, seed) must land in the request
    body for the OpenAI-compatible adapter.

    Note: `max_output_tokens` is our internal, direction-explicit name;
    it maps to the wire field `max_tokens` on OpenAI-classic and
    `max_completion_tokens` on OpenAI reasoning endpoints.
    """
    t = FakeTransport({"choices": [{"message": {"content": "x"}}]})
    params = LLMParams(
        max_output_tokens=2048, temperature=0.3, top_p=0.9,
        stop_sequences=["END"], response_format="json_object", seed=42,
        reasoning_mode="off", max_retries=0,
    )
    c = LLMClient(_openai_provider(), model="m", transport=t, params=params)
    await c.chat("hi")
    body = t.calls[0]["body"]
    assert body["max_tokens"] == 2048
    assert body["temperature"] == 0.3
    assert body["top_p"] == 0.9
    assert body["stop"] == ["END"]
    assert body["response_format"] == {"type": "json_object"}
    assert body["seed"] == 42
    # reasoning off → no reasoning_effort / max_completion_tokens.
    assert "reasoning_effort" not in body
    assert "max_completion_tokens" not in body


async def test_max_input_tokens_never_sent_on_wire():
    """max_input_tokens is a local declaration (context window) — no
    provider has a wire field for it, so the client must never send it
    in the request body regardless of provider."""
    params = LLMParams(max_output_tokens=2048, max_input_tokens=128000,
                         max_retries=0)
    # OpenAI-compatible
    t1 = FakeTransport({"choices": [{"message": {"content": "x"}}]})
    c1 = LLMClient(_openai_provider(), model="m", transport=t1, params=params)
    await c1.chat("hi")
    assert "max_input_tokens" not in t1.calls[0]["body"]
    # Anthropic
    t2 = FakeTransport({"content": [{"type": "text", "text": "x"}]})
    c2 = LLMClient(_anthropic_provider(), model="claude", transport=t2,
                    params=params)
    await c2.chat("hi")
    assert "max_input_tokens" not in t2.calls[0]["body"]


async def test_openai_reasoning_translates_to_reasoning_effort():
    """reasoning_mode=medium → reasoning_effort=medium, and the token
    cap moves to max_completion_tokens (OpenAI o-series / GPT-5 contract)."""
    t = FakeTransport({"choices": [{"message": {"content": "x"}}]})
    params = LLMParams(
        max_output_tokens=8000, reasoning_mode="medium", temperature=0.7,
        max_retries=0,
    )
    c = LLMClient(_openai_provider(), model="m", transport=t, params=params)
    await c.chat("hi")
    body = t.calls[0]["body"]
    assert body["reasoning_effort"] == "medium"
    assert body["max_completion_tokens"] == 8000
    # Reasoning models drop temperature/top_p — providers reject or
    # ignore them with warnings.
    assert "max_tokens" not in body
    assert "temperature" not in body
    assert "top_p" not in body


async def test_anthropic_reasoning_translates_to_thinking_block():
    """reasoning_mode != off on Anthropic → thinking: {type, budget_tokens},
    temperature/top_p dropped (thinking requires temperature=1)."""
    t = FakeTransport({"content": [{"type": "text", "text": "x"}]})
    params = LLMParams(
        max_output_tokens=8000, reasoning_mode="medium",
        reasoning_budget_tokens=3000, temperature=0.5, max_retries=0,
    )
    c = LLMClient(_anthropic_provider(), model="claude", transport=t,
                   params=params)
    await c.chat("hi")
    body = t.calls[0]["body"]
    assert body["max_tokens"] == 8000
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 3000}
    assert "temperature" not in body
    assert "top_p" not in body


async def test_anthropic_reasoning_auto_budget_from_mode():
    """When reasoning_budget_tokens is None, budget is derived from the
    mode (low=0.25, medium=0.5, high=0.75 of max_output_tokens),
    clamped to [1024, max_output_tokens-1]."""
    t = FakeTransport({"content": [{"type": "text", "text": "x"}]})
    params = LLMParams(max_output_tokens=8000, reasoning_mode="high",
                         reasoning_budget_tokens=None, max_retries=0)
    c = LLMClient(_anthropic_provider(), model="claude", transport=t,
                   params=params)
    await c.chat("hi")
    thinking = t.calls[0]["body"]["thinking"]
    assert thinking["type"] == "enabled"
    assert thinking["budget_tokens"] == 6000  # 8000 * 0.75


async def test_anthropic_reasoning_off_passes_temperature():
    """With reasoning off on Anthropic, normal sampling params are sent."""
    t = FakeTransport({"content": [{"type": "text", "text": "x"}]})
    params = LLMParams(max_output_tokens=4096, reasoning_mode="off",
                         temperature=0.2, top_p=0.85,
                         stop_sequences=["STOP"], max_retries=0)
    c = LLMClient(_anthropic_provider(), model="claude", transport=t,
                   params=params)
    await c.chat("hi")
    body = t.calls[0]["body"]
    assert "thinking" not in body
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.85
    assert body["stop_sequences"] == ["STOP"]
    # Anthropic has no native JSON mode / seed — these are always dropped.
    assert "response_format" not in body
    assert "seed" not in body


async def test_anthropic_max_tokens_falls_back_to_4096_when_none():
    """Anthropic's wire `max_tokens` is required. If the user cleared
    our `max_output_tokens` to None, we must still send 4096 rather
    than 400 out."""
    t = FakeTransport({"content": [{"type": "text", "text": "x"}]})
    params = LLMParams(max_output_tokens=None, reasoning_mode="off",
                         max_retries=0)
    c = LLMClient(_anthropic_provider(), model="claude", transport=t,
                   params=params)
    await c.chat("hi")
    assert t.calls[0]["body"]["max_tokens"] == 4096


async def test_default_params_when_none_supplied():
    """LLMClient with no params kwarg behaves like LLMParams() — sane
    defaults, reasoning off."""
    t = FakeTransport({"choices": [{"message": {"content": "y"}}]})
    c = LLMClient(_openai_provider(), model="m", transport=t)
    await c.chat("hi")
    body = t.calls[0]["body"]
    assert body["max_tokens"] == 4096
    assert body["temperature"] == 0.7
    assert "reasoning_effort" not in body


# ---------------- Retry behavior ----------------


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    """asyncio.sleep in the retry loop would make these tests slow.
    Patch it to a no-op for the client module only."""
    import krakey.llm.client as client_mod

    async def _noop(_):
        return None

    monkeypatch.setattr(client_mod.asyncio, "sleep", _noop)
    yield


async def test_retries_on_5xx_then_succeeds():
    """5xx is in retry_on_status by default — client should retry and
    eventually return the success payload."""
    success = {"choices": [{"message": {"content": "ok"}}]}
    t = SequenceTransport([
        TransportError(502, "bad gateway"),
        TransportError(503, "unavailable"),
        success,
    ])
    params = LLMParams(max_retries=3)
    c = LLMClient(_openai_provider(), model="m", transport=t, params=params)
    out = await c.chat("hi")
    assert out == "ok"
    assert len(t.calls) == 3


async def test_no_retry_on_4xx_non_429():
    """400 / 401 / 403 must fail fast — retrying them is always wrong."""
    t = SequenceTransport([TransportError(400, "bad request")])
    params = LLMParams(max_retries=5)
    c = LLMClient(_openai_provider(), model="m", transport=t, params=params)
    with pytest.raises(TransportError) as exc:
        await c.chat("hi")
    assert exc.value.status == 400
    assert len(t.calls) == 1


async def test_retries_on_429():
    """429 (rate limit) is the one 4xx that's in the default retry set."""
    success = {"choices": [{"message": {"content": "ok"}}]}
    t = SequenceTransport([TransportError(429, "slow down"), success])
    params = LLMParams(max_retries=3)
    c = LLMClient(_openai_provider(), model="m", transport=t, params=params)
    await c.chat("hi")
    assert len(t.calls) == 2


async def test_gives_up_after_max_retries():
    """Exhausting max_retries re-raises the last error."""
    t = SequenceTransport([
        TransportError(503),
        TransportError(503),
        TransportError(503),
    ])
    params = LLMParams(max_retries=2)  # total attempts = 3
    c = LLMClient(_openai_provider(), model="m", transport=t, params=params)
    with pytest.raises(TransportError):
        await c.chat("hi")
    assert len(t.calls) == 3


async def test_custom_retry_on_status_overrides_default():
    """User can widen / narrow which statuses retry."""
    success = {"choices": [{"message": {"content": "ok"}}]}
    t = SequenceTransport([TransportError(418, "teapot"), success])
    params = LLMParams(max_retries=3, retry_on_status=[418])
    c = LLMClient(_openai_provider(), model="m", transport=t, params=params)
    await c.chat("hi")
    assert len(t.calls) == 2
