"""Phase 1.6: hibernate_with_recall — incremental preload during hibernate."""
import asyncio
import time
from datetime import datetime

import pytest

from src.models.stimulus import Stimulus
from src.runtime.hibernate import hibernate_with_recall
from src.runtime.stimulus_buffer import StimulusBuffer


class SpyRecall:
    def __init__(self):
        self.received: list = []
        self.calls = 0

    async def add_stimuli(self, stimuli):
        self.calls += 1
        self.received.extend(stimuli)


def _stim(content, *, adrenalin=False):
    return Stimulus(
        type="user_message", source="test", content=content,
        timestamp=datetime.now(), adrenalin=adrenalin,
    )


async def test_times_out_with_no_stimulus():
    buf = StimulusBuffer()
    recall = SpyRecall()
    t0 = time.perf_counter()
    await hibernate_with_recall(0.2, buf, recall,
                                  min_interval=0.01, max_interval=10)
    elapsed = time.perf_counter() - t0
    assert 0.18 <= elapsed <= 0.6
    assert recall.received == []


async def test_preloads_non_adrenalin_stimulus_during_hibernate():
    buf = StimulusBuffer()
    recall = SpyRecall()

    async def producer():
        await asyncio.sleep(0.05)
        await buf.push(_stim("msg1"))

    asyncio.create_task(producer())
    await hibernate_with_recall(0.3, buf, recall,
                                  min_interval=0.01, max_interval=10)
    # Recall should have picked up msg1 without consuming the buffer
    assert len(recall.received) == 1
    assert recall.received[0].content == "msg1"
    # Buffer still holds the stimulus for drain later
    assert len(buf.drain()) == 1


async def test_adrenalin_stimulus_breaks_hibernate_early():
    buf = StimulusBuffer()
    recall = SpyRecall()

    async def producer():
        await asyncio.sleep(0.05)
        await buf.push(_stim("urgent!", adrenalin=True))

    asyncio.create_task(producer())
    t0 = time.perf_counter()
    await hibernate_with_recall(5.0, buf, recall,
                                  min_interval=0.01, max_interval=10)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.8
    # Adrenalin stim should still be preloaded
    assert any(s.adrenalin for s in recall.received)


async def test_multiple_stimuli_all_fed_to_recall():
    buf = StimulusBuffer()
    recall = SpyRecall()

    async def producer():
        for i in range(3):
            await asyncio.sleep(0.03)
            await buf.push(_stim(f"msg{i}"))

    asyncio.create_task(producer())
    await hibernate_with_recall(0.5, buf, recall,
                                  min_interval=0.01, max_interval=10)
    contents = sorted(s.content for s in recall.received)
    assert contents == ["msg0", "msg1", "msg2"]


async def test_preexisting_adrenalin_returns_immediately():
    buf = StimulusBuffer()
    await buf.push(_stim("urgent!", adrenalin=True))
    recall = SpyRecall()
    t0 = time.perf_counter()
    await hibernate_with_recall(2.0, buf, recall,
                                  min_interval=0.01, max_interval=10)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.3


async def test_peek_does_not_consume_buffer():
    """After hibernate preloads stimuli, drain() must still return them
    so the next heartbeat sees the full batch."""
    buf = StimulusBuffer()
    recall = SpyRecall()

    async def producer():
        await asyncio.sleep(0.03)
        await buf.push(_stim("msg"))

    asyncio.create_task(producer())
    await hibernate_with_recall(0.15, buf, recall,
                                  min_interval=0.01, max_interval=10)
    remaining = buf.drain()
    assert [s.content for s in remaining] == ["msg"]
