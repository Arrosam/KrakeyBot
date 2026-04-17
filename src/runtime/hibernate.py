"""Hibernate — wait + adrenalin break (DevSpec §6.3, Phase-0 subset).

Adrenalin only interrupts hibernation, never LLM inference.
Incremental recall will be layered in during Phase 1.
"""
from __future__ import annotations

import asyncio

from src.runtime.stimulus_buffer import StimulusBuffer


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


async def hibernate(interval: float, buffer: StimulusBuffer, *,
                    min_interval: float, max_interval: float) -> None:
    duration = clamp(interval, min_interval, max_interval)
    try:
        await asyncio.wait_for(buffer.wait_for_adrenalin(), timeout=duration)
    except asyncio.TimeoutError:
        pass
