"""Hibernate — wait + adrenalin break (DevSpec §6.3).

Adrenalin only interrupts hibernation, never LLM inference.
Preloads recall during the wait so the next heartbeat's prompt
already has fresh context.
"""
from __future__ import annotations

import asyncio
from typing import Protocol

from src.runtime.stimuli.queue import StimulusQueue


class IncrementalRecallLike(Protocol):
    async def add_stimuli(self, stimuli) -> None: ...


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


async def hibernate_with_recall(
    interval: float, queue: StimulusQueue,
    recall: IncrementalRecallLike, *,
    min_interval: float, max_interval: float,
    poll_slice: float = 0.05,
) -> None:
    """Wait up to `interval` seconds for new stimuli. Peek (not drain) each
    new batch into `recall` so the next heartbeat's Self prompt has fresh
    context. Break early on adrenalin.
    """
    duration = clamp(interval, min_interval, max_interval)
    loop = asyncio.get_event_loop()
    deadline = loop.time() + duration

    # Short-circuit: already-adrenalin queue means we never sleep.
    if queue.has_adrenalin():
        new = queue.peek_unrecalled()
        if new:
            await recall.add_stimuli(new)
        return

    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        try:
            await asyncio.wait_for(
                queue.wait_for_any(), timeout=min(remaining, poll_slice),
            )
        except asyncio.TimeoutError:
            continue

        new = queue.peek_unrecalled()
        if new:
            await recall.add_stimuli(new)

        if queue.has_adrenalin():
            break
