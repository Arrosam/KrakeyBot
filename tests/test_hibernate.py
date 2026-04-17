import asyncio
import time
from datetime import datetime

import pytest

from src.models.stimulus import Stimulus
from src.runtime.hibernate import clamp, hibernate
from src.runtime.stimulus_buffer import StimulusBuffer


def test_clamp_bounds():
    assert clamp(1, 2, 10) == 2
    assert clamp(20, 2, 10) == 10
    assert clamp(5, 2, 10) == 5


async def test_waits_for_full_interval_when_no_adrenalin():
    buf = StimulusBuffer()
    t0 = time.perf_counter()
    await hibernate(0.15, buf, min_interval=0.01, max_interval=10)
    elapsed = time.perf_counter() - t0
    assert 0.13 <= elapsed <= 0.5


async def test_interrupts_on_adrenalin():
    buf = StimulusBuffer()

    async def shouter():
        await asyncio.sleep(0.02)
        await buf.push(Stimulus(
            type="user_message", source="test",
            content="!", timestamp=datetime.now(), adrenalin=True,
        ))

    asyncio.create_task(shouter())
    t0 = time.perf_counter()
    await hibernate(2.0, buf, min_interval=0.01, max_interval=10)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"should have broken early, took {elapsed:.3f}s"


async def test_non_adrenalin_push_does_not_interrupt():
    buf = StimulusBuffer()

    async def whisperer():
        await asyncio.sleep(0.02)
        await buf.push(Stimulus(
            type="user_message", source="test",
            content="quiet", timestamp=datetime.now(), adrenalin=False,
        ))

    asyncio.create_task(whisperer())
    t0 = time.perf_counter()
    await hibernate(0.15, buf, min_interval=0.01, max_interval=10)
    elapsed = time.perf_counter() - t0
    assert elapsed >= 0.13


async def test_clamps_interval_before_sleeping():
    buf = StimulusBuffer()
    t0 = time.perf_counter()
    # requested 100s but max is 0.05 → sleeps 0.05s
    await hibernate(100, buf, min_interval=0.01, max_interval=0.05)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5


async def test_pre_existing_adrenalin_returns_immediately():
    buf = StimulusBuffer()
    await buf.push(Stimulus(
        type="user_message", source="test", content="!",
        timestamp=datetime.now(), adrenalin=True,
    ))
    t0 = time.perf_counter()
    await hibernate(2.0, buf, min_interval=0.01, max_interval=10)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.2
