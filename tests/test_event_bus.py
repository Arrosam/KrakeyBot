"""RuntimeEventBus: typed pub/sub for runtime → dashboard wiring.

Logger stays direct (Runtime calls log.X). EventBus is the *additional*
channel a Dashboard subscribes to; it never gates runtime progress.
"""
import asyncio

import pytest

from krakey.runtime.events.event_types import (
    DecisionEvent, HeartbeatStartEvent, DecisionExecutedEvent, PromptBuiltEvent, StimuliQueuedEvent, TentacleResultEvent, ThinkingEvent,
)
from krakey.runtime.events.event_bus import EventBus


def test_event_dataclasses_carry_typed_fields():
    e = ThinkingEvent(heartbeat_id=3, text="thinking text")
    assert e.heartbeat_id == 3
    assert e.text == "thinking text"

    h = DecisionExecutedEvent(heartbeat_id=3, tentacle_calls_count=2,
                            memory_writes_count=1, memory_updates_count=0,
                            sleep_requested=False)
    assert h.tentacle_calls_count == 2 and h.sleep_requested is False


def test_publish_dispatches_to_all_subscribers():
    bus = EventBus()
    seen_a, seen_b = [], []
    bus.subscribe(seen_a.append)
    bus.subscribe(seen_b.append)
    e = ThinkingEvent(heartbeat_id=1, text="hi")
    bus.publish(e)
    assert seen_a == [e]
    assert seen_b == [e]


def test_subscriber_exception_does_not_break_others_or_publisher():
    bus = EventBus()
    survivor: list = []

    def boom(_e): raise RuntimeError("subscriber blew up")
    def good(e): survivor.append(e)

    bus.subscribe(boom)
    bus.subscribe(good)
    e = ThinkingEvent(heartbeat_id=1, text="x")
    bus.publish(e)  # should not raise
    assert survivor == [e]


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    seen: list = []
    cb = seen.append
    bus.subscribe(cb)
    bus.publish(ThinkingEvent(heartbeat_id=1, text="a"))
    bus.unsubscribe(cb)
    bus.publish(ThinkingEvent(heartbeat_id=2, text="b"))
    assert len(seen) == 1


async def test_async_subscriber_scheduled_as_task():
    bus = EventBus()
    received: list = []

    async def async_cb(e):
        await asyncio.sleep(0)
        received.append(e)

    bus.subscribe(async_cb)
    bus.publish(ThinkingEvent(heartbeat_id=1, text="async"))
    # Give the event loop one tick
    await asyncio.sleep(0.01)
    assert len(received) == 1


async def test_runtime_publishes_phase_events():
    """Integration: a single heartbeat publishes lifecycle events to the
    bus so a Dashboard subscriber sees them."""
    from krakey.runtime.events.event_bus import EventBus
    from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes

    bus = EventBus()
    received = []
    bus.subscribe(received.append)

    self_llm = ScriptedLLM(["[DECISION]\nNo action.\n[HIBERNATE]\n1"])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=ScriptedLLM([]),
    )
    runtime.events = bus
    await runtime.run(iterations=1)
    await runtime.close()

    kinds = {e.kind for e in received}
    assert {"heartbeat_start", "gm_stats", "decision", "hibernate"} <= kinds


def test_event_kind_property_for_serialization():
    """UI WS layer needs a string kind discriminator."""
    from krakey.runtime.events.event_types import GMStatsEvent
    assert ThinkingEvent(1, "x").kind == "thinking"
    assert DecisionEvent(1, "x").kind == "decision"
    assert HeartbeatStartEvent(1, 0).kind == "heartbeat_start"
    assert PromptBuiltEvent(1, {}).kind == "prompt_built"
    assert StimuliQueuedEvent([]).kind == "stimuli_queued"
    assert TentacleResultEvent("action", "x").kind == "tentacle_result"
    # Acronym run preserved as one token
    assert GMStatsEvent(1, 0, 0, 0).kind == "gm_stats"
