import asyncio
from datetime import datetime, timedelta

import pytest

from src.models.stimulus import Stimulus
from src.runtime.stimuli.stimulus_buffer import StimulusBuffer


def _s(content, *, adrenalin=False, offset_seconds=0):
    return Stimulus(
        type="user_message",
        source="test",
        content=content,
        timestamp=datetime(2026, 4, 18, 0, 0, 0) + timedelta(seconds=offset_seconds),
        adrenalin=adrenalin,
    )


async def test_push_drain_preserves_time_order():
    buf = StimulusBuffer()
    await buf.push(_s("a", offset_seconds=2))
    await buf.push(_s("b", offset_seconds=1))
    await buf.push(_s("c", adrenalin=True, offset_seconds=3))

    drained = buf.drain()
    assert [s.content for s in drained] == ["b", "a", "c"]
    assert buf.drain() == []  # reset


async def test_adrenalin_sets_event_and_cleared_by_drain():
    buf = StimulusBuffer()
    await buf.push(_s("calm"))
    assert not buf.has_adrenalin()

    await buf.push(_s("urgent!", adrenalin=True))
    assert buf.has_adrenalin()

    buf.drain()
    assert not buf.has_adrenalin()


async def test_wait_for_adrenalin_returns_when_set():
    buf = StimulusBuffer()

    async def producer():
        await asyncio.sleep(0.02)
        await buf.push(_s("urgent", adrenalin=True))

    asyncio.create_task(producer())
    await asyncio.wait_for(buf.wait_for_adrenalin(), timeout=1.0)
    assert buf.has_adrenalin()


async def test_wait_for_any_returns_on_any_push():
    buf = StimulusBuffer()

    async def producer():
        await asyncio.sleep(0.02)
        await buf.push(_s("calm"))

    asyncio.create_task(producer())
    await asyncio.wait_for(buf.wait_for_any(), timeout=1.0)


async def test_peek_unrecalled_returns_new_items_only_once():
    buf = StimulusBuffer()
    await buf.push(_s("a", offset_seconds=1))
    await buf.push(_s("b", offset_seconds=2))

    first = buf.peek_unrecalled()
    assert [s.content for s in first] == ["a", "b"]

    # second peek: nothing new
    assert buf.peek_unrecalled() == []

    await buf.push(_s("c", offset_seconds=3))
    second = buf.peek_unrecalled()
    assert [s.content for s in second] == ["c"]


async def test_drain_resets_peek_index():
    buf = StimulusBuffer()
    await buf.push(_s("a", offset_seconds=1))
    buf.peek_unrecalled()
    buf.drain()
    await buf.push(_s("b", offset_seconds=2))
    assert [s.content for s in buf.peek_unrecalled()] == ["b"]
