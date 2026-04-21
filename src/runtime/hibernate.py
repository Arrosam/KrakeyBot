"""Hibernate — wait + adrenalin break (DevSpec §6.3).

Adrenalin only interrupts hibernation, never LLM inference.
Phase 1 adds `hibernate_with_recall` to preload recall during the wait.
"""
from __future__ import annotations

import asyncio
from typing import Protocol

from src.runtime.stimulus_buffer import StimulusBuffer


class IncrementalRecallLike(Protocol):
    async def add_stimuli(self, stimuli) -> None: ...


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


async def hibernate(interval: float, buffer: StimulusBuffer, *,
                    min_interval: float, max_interval: float) -> None:
    duration = clamp(interval, min_interval, max_interval)
    try:
        await asyncio.wait_for(buffer.wait_for_adrenalin(), timeout=duration)
    except asyncio.TimeoutError:
        pass


async def hibernate_with_recall(
    interval: float, buffer: StimulusBuffer,
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

    # Short-circuit: already-adrenalin buffer means we never sleep.
    if buffer.has_adrenalin():
        new = buffer.peek_unrecalled()
        if new:
            await recall.add_stimuli(new)
        return

    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        try:
            await asyncio.wait_for(
                buffer.wait_for_any(), timeout=min(remaining, poll_slice),
            )
        except asyncio.TimeoutError:
            continue

        new = buffer.peek_unrecalled()
        if new:
            await recall.add_stimuli(new)

        if buffer.has_adrenalin():
            break
