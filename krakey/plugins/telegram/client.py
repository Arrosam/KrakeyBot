"""Telegram HTTP client — shared by the sensory + reply tool.

One `HttpTelegramClient` instance is built by the plugin factory and
handed to both components so a single bot connection serves both
inbound (getUpdates) and outbound (sendMessage) directions. Tests mock
the `TelegramClient` Protocol directly.
"""
from __future__ import annotations

from typing import Any, Protocol

import aiohttp


class TelegramClient(Protocol):
    async def get_updates(self, offset: int,
                            timeout: int = 10) -> list[dict[str, Any]]: ...

    async def send_message(self, chat_id: int, text: str) -> None: ...


class HttpTelegramClient:
    """Production client — raw aiohttp against api.telegram.org."""

    def __init__(self, token: str,
                  base_url: str = "https://api.telegram.org",
                  poll_timeout: int = 25):
        self._token = token
        self._base = f"{base_url.rstrip('/')}/bot{token}"
        self._poll_timeout = poll_timeout
        self._timeout = aiohttp.ClientTimeout(total=poll_timeout + 10)

    async def get_updates(self, offset: int, timeout: int | None = None
                            ) -> list[dict[str, Any]]:
        params = {"offset": offset,
                   "timeout": timeout if timeout is not None else self._poll_timeout}
        async with aiohttp.ClientSession(timeout=self._timeout) as s:
            async with s.get(f"{self._base}/getUpdates", params=params) as r:
                r.raise_for_status()
                data = await r.json()
        if not data.get("ok"):
            raise RuntimeError(f"telegram getUpdates not ok: {data}")
        return list(data.get("result", []))

    async def send_message(self, chat_id: int, text: str) -> None:
        body = {"chat_id": chat_id, "text": text}
        async with aiohttp.ClientSession(timeout=self._timeout) as s:
            async with s.post(f"{self._base}/sendMessage", json=body) as r:
                r.raise_for_status()
                data = await r.json()
        if not data.get("ok"):
            raise RuntimeError(f"telegram sendMessage not ok: {data}")
