"""Central FIFO queue for stimuli + adrenalin/new-event signaling.

Sensories produce stimuli via the ``push`` callable; the heartbeat
consumes via ``drain``. Hibernate uses ``wait_for_adrenalin`` /
``wait_for_any`` to break early when fresh stimuli arrive.

This is one half of what used to be ``StimulusBuffer``. The other
half — sensory lifecycle ownership — lives in
``src.runtime.stimuli.sensory_registry``. The fusion was originally
needed to avoid an import cycle in ``interfaces/sensory.py``, but
``Sensory.start`` now takes a ``PushCallback`` (a callable) instead
of the buffer itself, so the two responsibilities can stand alone.
"""
from __future__ import annotations

import asyncio

from src.models.stimulus import Stimulus


class StimulusQueue:
    """FIFO queue with adrenalin + new-event signals."""

    def __init__(self):
        self._queue: list[Stimulus] = []
        self._recalled_up_to: int = 0
        self._adrenalin_event = asyncio.Event()
        self._new_event = asyncio.Event()

    async def push(self, s: Stimulus) -> None:
        self._queue.append(s)
        self._new_event.set()
        if s.adrenalin:
            self._adrenalin_event.set()

    def drain(self) -> list[Stimulus]:
        items = sorted(self._queue, key=lambda s: s.timestamp)
        self._queue = []
        self._recalled_up_to = 0
        self._adrenalin_event.clear()
        self._new_event.clear()
        return items

    def peek_unrecalled(self) -> list[Stimulus]:
        new = self._queue[self._recalled_up_to:]
        self._recalled_up_to = len(self._queue)
        self._new_event.clear()
        return new

    async def wait_for_adrenalin(self) -> None:
        await self._adrenalin_event.wait()

    async def wait_for_any(self) -> None:
        await self._new_event.wait()

    def has_adrenalin(self) -> bool:
        return self._adrenalin_event.is_set()
