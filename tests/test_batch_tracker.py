"""Phase 1.5: BatchTrackerSensory — fires adrenalin stimulus when dispatched
batch drains."""
import pytest

from src.runtime.batch_tracker import BatchTrackerSensory
from src.runtime.stimulus_buffer import StimulusBuffer


async def _tracker(buffer=None):
    buf = buffer or StimulusBuffer()
    t = BatchTrackerSensory()
    await t.start(buf.push)
    return t, buf


async def test_complete_partial_no_trigger():
    t, buf = await _tracker()
    t.register_batch(["a", "b", "c"])
    await t.mark_completed("a")
    await t.mark_completed("b")
    assert buf.drain() == []  # still 1 pending


async def test_complete_last_triggers_adrenalin_stimulus():
    t, buf = await _tracker()
    t.register_batch(["a", "b", "c"])
    await t.mark_completed("a")
    await t.mark_completed("b")
    await t.mark_completed("c")
    stims = buf.drain()
    assert len(stims) == 1
    assert stims[0].type == "batch_complete"
    assert stims[0].source == "sensory:batch_tracker"
    assert stims[0].adrenalin is True


async def test_extend_during_flight_delays_trigger():
    """register 3 → complete 2 → extend 1 → complete remaining old →
    should NOT trigger yet (the new one is still pending) → complete new →
    triggers."""
    t, buf = await _tracker()
    t.register_batch(["a", "b", "c"])
    await t.mark_completed("a")
    await t.mark_completed("b")
    t.extend_batch(["d"])
    await t.mark_completed("c")
    assert buf.drain() == []  # "d" still pending
    await t.mark_completed("d")
    stims = buf.drain()
    assert len(stims) == 1
    assert stims[0].adrenalin is True


async def test_register_empty_never_triggers():
    t, buf = await _tracker()
    t.register_batch([])
    assert buf.drain() == []


async def test_mark_unknown_id_ignored():
    t, buf = await _tracker()
    t.register_batch(["a"])
    await t.mark_completed("does-not-exist")  # no-op
    assert buf.drain() == []
    await t.mark_completed("a")
    assert len(buf.drain()) == 1


async def test_sensory_interface():
    t = BatchTrackerSensory()
    assert t.name == "batch_tracker"
    assert t.default_adrenalin is True
    buf = StimulusBuffer()
    await t.start(buf.push)
    await t.stop()  # no-op, must not raise


async def test_trigger_only_once_per_drain_cycle():
    """Once pending empties and the stimulus fires, a subsequent
    register_batch → complete cycle triggers a NEW stimulus."""
    t, buf = await _tracker()
    t.register_batch(["a"])
    await t.mark_completed("a")
    assert len(buf.drain()) == 1

    t.register_batch(["b"])
    await t.mark_completed("b")
    assert len(buf.drain()) == 1
