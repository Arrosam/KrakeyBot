"""Phase 3 / D: Telegram Sensory + thin HTTP client.

Inbound side. Polls Telegram Bot API in a background task and pushes each
text message as a `user_message` stimulus (adrenalin=True by default,
since incoming messages from a real human warrant attention).

Outbound side lives in src/tentacles/telegram_reply.py — the same client
instance is shared with that tentacle so a single bot connection serves
both directions.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Protocol

import aiohttp

from src.interfaces.sensory import Sensory
from src.models.stimulus import Stimulus
from src.runtime.stimulus_buffer import StimulusBuffer


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


class TelegramSensory(Sensory):
    def __init__(self, client: TelegramClient, *,
                  allowed_chat_ids: set[int] | None = None,
                  error_backoff: float = 2.0):
        self._client = client
        self._allowed = allowed_chat_ids
        self._error_backoff = error_backoff
        self._buffer: StimulusBuffer | None = None
        self._task: asyncio.Task | None = None
        self._stopped = False
        self._offset = 0

    @property
    def name(self) -> str:
        return "telegram"

    @property
    def default_adrenalin(self) -> bool:
        return True

    async def start(self, buffer: StimulusBuffer) -> None:
        if self._task and not self._task.done():
            return
        self._buffer = buffer
        self._stopped = False
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _poll_loop(self) -> None:
        assert self._buffer is not None
        while not self._stopped:
            try:
                updates = await self._client.get_updates(self._offset)
            except Exception:  # noqa: BLE001
                await asyncio.sleep(self._error_backoff)
                continue
            for u in updates:
                self._offset = max(self._offset, u.get("update_id", 0) + 1)
                msg = u.get("message") or {}
                chat = msg.get("chat") or {}
                chat_id = chat.get("id")
                text = msg.get("text")
                if chat_id is None or not text:
                    continue
                if self._allowed and chat_id not in self._allowed:
                    continue
                await self._buffer.push(Stimulus(
                    type="user_message",
                    source=f"sensory:telegram:{chat_id}",
                    content=text,
                    timestamp=datetime.now(),
                    adrenalin=True,
                    metadata={"chat_id": chat_id},
                ))
            await asyncio.sleep(0)
