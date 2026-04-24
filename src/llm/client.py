"""Unified LLM client (DevSpec §14).

Supports openai_compatible and anthropic providers. Transport is
injectable to keep tests offline; the default transport uses aiohttp.

Per-request parameters come from `LLMParams` on the role binding. The
client owns the provider-specific translation so the rest of the code
(self_agent, hypothalamus, compact, etc.) can stay provider-agnostic:

* ``reasoning_mode`` maps to Anthropic ``thinking.budget_tokens`` or
  OpenAI ``reasoning_effort`` depending on the provider type.
* ``max_tokens`` becomes ``max_completion_tokens`` when we detect an
  OpenAI reasoning model (heuristic on ``reasoning_mode != "off"``).
* ``response_format="json_object"`` becomes OpenAI's nested JSON-mode
  object; Anthropic has no native JSON mode so it is dropped silently
  (prompts are expected to enforce the format there).
* Fields unsupported by the target provider are silently omitted
  rather than sent — e.g. ``temperature`` is dropped for DeepSeek
  Reasoner / OpenAI o-series because they reject it or ignore it with
  confusing warnings.

Retries: transient failures (5xx + 429 by default) are retried with
exponential backoff (1s * 2^attempt) plus jitter, capped at
``params.max_retries``. 4xx other than 429 fails fast.
"""
from __future__ import annotations

import asyncio
import random
from typing import Any, Protocol

import aiohttp

from src.models.config import LLMParams, Provider


class Transport(Protocol):
    async def post_json(self, url: str, headers: dict[str, str],
                        json_body: dict[str, Any]) -> dict[str, Any]: ...


class TransportError(Exception):
    """Raised by transports for HTTP-level failures so retry logic can
    inspect the status code. Wraps the underlying status + message."""

    def __init__(self, status: int, message: str = ""):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


class AiohttpTransport:
    def __init__(self, timeout_seconds: float = 120.0):
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def post_json(self, url, headers, json_body):
        async with aiohttp.ClientSession(timeout=self._timeout) as s:
            async with s.post(url, headers=headers, json=json_body) as r:
                if r.status >= 400:
                    body = ""
                    try:
                        body = await r.text()
                    except Exception:  # noqa: BLE001
                        pass
                    raise TransportError(r.status, body[:500])
                return await r.json()


class LLMClient:
    """Thin wrapper around a Transport that knows how to speak to
    OpenAI-compatible + Anthropic APIs. Parameters supplied via
    ``params`` (an ``LLMParams`` instance) are merged into every
    chat request and translated to each provider's field names.

    ``transport`` is overridable so tests can inject a fake. When a
    transport is constructed by this client, its timeout is taken from
    ``params.timeout_seconds`` so per-role timeouts actually hit the
    wire (a 20s hypothalamus vs a 180s self call).
    """

    def __init__(
        self,
        provider: Provider,
        model: str,
        transport: Transport | None = None,
        *,
        params: LLMParams | None = None,
    ):
        self.provider = provider
        self.model = model
        self.params = params or LLMParams()
        self.transport: Transport = (
            transport if transport is not None
            else AiohttpTransport(timeout_seconds=self.params.timeout_seconds)
        )

    async def chat(self, messages, **kwargs) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if self.provider.type == "anthropic":
            body = self._build_anthropic_body(messages, kwargs)
            url = self._url("/v1/messages")
            headers = self._anthropic_headers()
            data = await self._post_with_retry(url, headers, body)
            parts = [p["text"] for p in data.get("content", [])
                     if p.get("type") == "text"]
            return "".join(parts)
        # openai_compatible
        body = self._build_openai_body(messages, kwargs)
        url = self._url("/v1/chat/completions")
        headers = self._openai_headers()
        data = await self._post_with_retry(url, headers, body)
        return data["choices"][0]["message"]["content"]

    async def embed(self, text: str) -> list[float]:
        url = self._url("/v1/embeddings")
        body = {"model": self.model, "input": text}
        data = await self._post_with_retry(url, self._openai_headers(), body)
        return list(data["data"][0]["embedding"])

    async def rerank(self, query: str, docs: list[str]) -> list[float]:
        url = self._url("/v1/rerank")
        body = {"model": self.model, "query": query, "documents": docs}
        data = await self._post_with_retry(url, self._openai_headers(), body)
        results = sorted(data["results"], key=lambda r: r["index"])
        return [float(r["relevance_score"]) for r in results]

    # ---------------- body builders ----------------

    def _build_openai_body(
        self, messages: list[dict[str, Any]], overrides: dict[str, Any],
    ) -> dict[str, Any]:
        p = self.params
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        reasoning_on = p.reasoning_mode and p.reasoning_mode != "off"
        # Newer OpenAI reasoning models (o-series, GPT-5) require
        # `max_completion_tokens` instead of `max_tokens`. Heuristic:
        # if reasoning is enabled we're on a reasoning-capable endpoint,
        # so use the new field. Otherwise stick with the classic one
        # which DeepSeek / Qwen / vLLM / llama.cpp all still accept.
        # max_input_tokens is intentionally never sent — providers
        # have no wire field for it; it's a local declaration only.
        if p.max_output_tokens is not None:
            if reasoning_on:
                body["max_completion_tokens"] = p.max_output_tokens
            else:
                body["max_tokens"] = p.max_output_tokens
        # Temperature: o-series + DeepSeek-Reasoner reject or ignore it;
        # dropping it silently is safer than letting the server 400.
        if p.temperature is not None and not reasoning_on:
            body["temperature"] = p.temperature
        if p.top_p is not None and not reasoning_on:
            body["top_p"] = p.top_p
        if p.stop_sequences:
            body["stop"] = list(p.stop_sequences)
        if p.response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        if p.seed is not None:
            body["seed"] = p.seed
        if reasoning_on:
            body["reasoning_effort"] = p.reasoning_mode
        body.update(overrides)
        return body

    def _build_anthropic_body(
        self, messages: list[dict[str, Any]], overrides: dict[str, Any],
    ) -> dict[str, Any]:
        p = self.params
        # Anthropic's wire field is called `max_tokens`; ours is
        # `max_output_tokens` (direction made explicit). It is required
        # by the Messages API — if the user cleared ours to None, fall
        # back to the dataclass default so we still send a valid body.
        max_output = (p.max_output_tokens
                      if p.max_output_tokens is not None else 4096)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_output,
        }
        reasoning_on = p.reasoning_mode and p.reasoning_mode != "off"
        if reasoning_on:
            # Derive a token budget: explicit `reasoning_budget_tokens`
            # wins; otherwise scale by mode. Must be ≥ 1024 and strictly
            # less than max_output_tokens (Anthropic constraint).
            budget = p.reasoning_budget_tokens
            if budget is None:
                scale = {"low": 0.25, "medium": 0.5, "high": 0.75}.get(
                    p.reasoning_mode, 0.5
                )
                budget = int(max_output * scale)
            budget = max(1024, min(budget, max_output - 1))
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
            # Anthropic thinking mode requires temperature == 1 (or
            # unset). Only pass temperature when thinking is off.
        else:
            if p.temperature is not None:
                body["temperature"] = p.temperature
            if p.top_p is not None:
                body["top_p"] = p.top_p
        if p.stop_sequences:
            body["stop_sequences"] = list(p.stop_sequences)
        # Anthropic has no native JSON mode and no `seed` field —
        # response_format + seed are intentionally dropped here.
        body.update(overrides)
        return body

    # ---------------- transport + retry ----------------

    async def _post_with_retry(
        self, url: str, headers: dict[str, str], body: dict[str, Any],
    ) -> dict[str, Any]:
        p = self.params
        attempts = max(1, p.max_retries + 1)
        last_err: Exception | None = None
        for i in range(attempts):
            try:
                return await self.transport.post_json(url, headers, body)
            except TransportError as e:
                last_err = e
                if e.status not in p.retry_on_status or i == attempts - 1:
                    raise
            except Exception as e:  # noqa: BLE001 — network/timeouts etc.
                last_err = e
                if i == attempts - 1:
                    raise
            # Exponential backoff with jitter. Base 1s, doubled per
            # attempt, capped at 30s to stay under the request timeout.
            delay = min(30.0, 1.0 * (2 ** i)) * (0.5 + random.random())
            await asyncio.sleep(delay)
        if last_err is not None:  # pragma: no cover — loop always raises
            raise last_err
        raise RuntimeError("unreachable")

    # ---------------- url + header helpers ----------------

    def _url(self, path: str) -> str:
        return f"{self.provider.base_url.rstrip('/')}{path}"

    def _openai_headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.provider.api_key:
            h["Authorization"] = f"Bearer {self.provider.api_key}"
        return h

    def _anthropic_headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json",
             "anthropic-version": "2023-06-01"}
        if self.provider.api_key:
            h["x-api-key"] = self.provider.api_key
        return h
