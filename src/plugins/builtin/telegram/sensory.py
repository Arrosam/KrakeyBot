"""Telegram inbound sensory — polls getUpdates in a background task.

Pushes each incoming text message as a `user_message` Stimulus with
`adrenalin=True` (a real human contacted Krakey; that's worth waking
for). Chat-id allowlisting is enforced here, not at the client level.
"""
from __future__ import annotations

import asyncio
from datetime import datetime

from src.interfaces.sensory import Sensory
from src.models.stimulus import Stimulus
from src.runtime.stimulus_buffer import StimulusBuffer

from .client import TelegramClient


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
