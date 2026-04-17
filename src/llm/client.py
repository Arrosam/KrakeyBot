"""Unified LLM client (DevSpec §14).

Supports openai_compatible and anthropic providers. Transport is injectable
to keep tests offline; the default transport uses aiohttp.
"""
from __future__ import annotations

from typing import Any, Protocol

import aiohttp

from src.models.config import Provider


class Transport(Protocol):
    async def post_json(self, url: str, headers: dict[str, str],
                        json_body: dict[str, Any]) -> dict[str, Any]: ...


class AiohttpTransport:
    def __init__(self, timeout_seconds: float = 120.0):
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def post_json(self, url, headers, json_body):
        async with aiohttp.ClientSession(timeout=self._timeout) as s:
            async with s.post(url, headers=headers, json=json_body) as r:
                r.raise_for_status()
                return await r.json()


class LLMClient:
    def __init__(self, provider: Provider, model: str, transport: Transport | None = None):
        self.provider = provider
        self.model = model
        self.transport: Transport = transport or AiohttpTransport()

    async def chat(self, messages, **kwargs) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if self.provider.type == "anthropic":
            return await self._chat_anthropic(messages, **kwargs)
        return await self._chat_openai(messages, **kwargs)

    async def embed(self, text: str) -> list[float]:
        url = self._url("/v1/embeddings")
        body = {"model": self.model, "input": text}
        data = await self.transport.post_json(url, self._openai_headers(), body)
        return list(data["data"][0]["embedding"])

    async def rerank(self, query: str, docs: list[str]) -> list[float]:
        url = self._url("/v1/rerank")
        body = {"model": self.model, "query": query, "documents": docs}
        data = await self.transport.post_json(url, self._openai_headers(), body)
        results = sorted(data["results"], key=lambda r: r["index"])
        return [float(r["relevance_score"]) for r in results]

    async def _chat_openai(self, messages, **kwargs) -> str:
        body = {"model": self.model, "messages": messages, **kwargs}
        data = await self.transport.post_json(
            self._url("/v1/chat/completions"), self._openai_headers(), body)
        return data["choices"][0]["message"]["content"]

    async def _chat_anthropic(self, messages, **kwargs) -> str:
        max_tokens = kwargs.pop("max_tokens", 4096)
        body = {"model": self.model, "messages": messages,
                "max_tokens": max_tokens, **kwargs}
        data = await self.transport.post_json(
            self._url("/v1/messages"), self._anthropic_headers(), body)
        parts = [p["text"] for p in data.get("content", []) if p.get("type") == "text"]
        return "".join(parts)

    def _url(self, path: str) -> str:
        return f"{self.provider.base_url.rstrip('/')}{path}"

    def _openai_headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.provider.api_key:
            h["Authorization"] = f"Bearer {self.provider.api_key}"
        return h

    def _anthropic_headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        if self.provider.api_key:
            h["x-api-key"] = self.provider.api_key
        return h
