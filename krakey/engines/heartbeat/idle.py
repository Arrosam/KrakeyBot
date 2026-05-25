"""Idle — wait + adrenalin break (DevSpec §6.3).

Adrenalin only interrupts hibernation, never LLM inference.
Preloads recall during the wait so the next heartbeat's prompt
already has fresh context.

Two helpers are provided:

``idle_with_recall`` — full hibernation wait: polls the buffer,
  preloads recall with any new stimuli that arrive, and breaks early
  on adrenalin.

``wait_or_adrenalin`` — lightweight interruption signal only: does NOT
  touch recall or drain the buffer; returns True as soon as an adrenalin
  stimulus is present, False if the full duration elapses with none.
  Used by _phase_run_self to make retry-waits interruptible while
  keeping the LLM call itself non-interruptible.
"""
from __future__ import annotations

import asyncio

from krakey.interfaces.engines.recall import RecallSession
from krakey.runtime.stimuli.stimulus_buffer import StimulusBuffer


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


async def wait_or_adrenalin(
    buffer: StimulusBuffer,
    seconds: float,
    *,
    poll_slice: float = 0.05,
) -> bool:
    """Wait up to ``seconds`` for an adrenalin stimulus.

    Returns True as soon as ``buffer.has_adrenalin()`` is True —
    including immediately on entry if it is already True before any
    sleeping occurs. Returns False if the full ``seconds`` elapse with
    no adrenalin.

    This helper is the interruptible side of the retry-wait pattern:
    it signals interruption only. It does NOT drain, peek, or touch
    recall — the caller is responsible for draining and rebuilding
    context when True is returned.

    Adrenalin only interrupts hibernation, never LLM inference; callers
    must not wrap an LLM call with this helper.
    """
    # Fast path: already-adrenalin buffer — return without sleeping.
    if buffer.has_adrenalin():
        return True

    if seconds <= 0:
        return False

    loop = asyncio.get_event_loop()
    deadline = loop.time() + seconds

    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        slice_dur = min(remaining, poll_slice)
        t0 = loop.time()
        timed_out = False
        try:
            await asyncio.wait_for(
                buffer.wait_for_any(), timeout=slice_dur,
            )
        except asyncio.TimeoutError:
            timed_out = True

        if buffer.has_adrenalin():
            return True

        # Prevent busy-spin: wait_for_any() resolved but it wasn't
        # adrenalin — the internal event stays set (we must not
        # drain), so the next iteration would return instantly.
        # Sleep the remainder of the poll slice to throttle.
        if not timed_out:
            leftover = slice_dur - (loop.time() - t0)
            if leftover > 0:
                await asyncio.sleep(leftover)

    return False


async def idle_with_recall(
    interval: float, buffer: StimulusBuffer,
    recall: RecallSession, *,
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
