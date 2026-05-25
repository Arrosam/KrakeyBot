"""Heartbeat behavior: StimulusReadEvent emission at drain time.

When the heartbeat drains the buffer at the start of a beat, every drained
stimulus that carries a chat_message_id must be reported via a single
StimulusReadEvent (so the dashboard can flip the matching web-chat bubble to
"read"). Beats that drain no chat-tagged stimuli must NOT publish the event.
"""
from datetime import datetime, timedelta

import pytest

from krakey.models.stimulus import Stimulus
from krakey.runtime.events.event_bus import EventBus
from krakey.runtime.events.event_types import StimulusReadEvent
from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes


_QUIET_BEAT = "[THINKING]\n(quiet beat)\n[DECISION]\nNo action.\n[IDLE]\n1"
_T0 = datetime(2026, 5, 26, 0, 0, 0)


def _chat_stim(content, message_id, *, offset_seconds=0):
    return Stimulus(
        type="user_message",
        source="channel:web_chat",
        content=content,
        timestamp=_T0 + timedelta(seconds=offset_seconds),
        chat_message_id=message_id,
    )


async def _run_one_beat_collecting_events(stimuli):
    bus = EventBus()
    received: list = []
    bus.subscribe(received.append)
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([_QUIET_BEAT]), hypo_llm=ScriptedLLM([]),
    )
    runtime.events = bus
    for s in stimuli:
        await runtime.buffer.push(s)
    await runtime.run(iterations=1)
    await runtime.close()
    return [e for e in received if isinstance(e, StimulusReadEvent)]


async def test_drained_chat_stimulus_publishes_stimulus_read():
    reads = await _run_one_beat_collecting_events([_chat_stim("hello", "m1")])
    assert len(reads) == 1
    assert reads[0].chat_message_ids == ["m1"]
    assert reads[0].kind == "stimulus_read"


async def test_multiple_chat_ids_collected_into_one_event():
    reads = await _run_one_beat_collecting_events([
        _chat_stim("a", "m1", offset_seconds=1),
        _chat_stim("b", "m2", offset_seconds=2),
    ])
    assert len(reads) == 1
    assert reads[0].chat_message_ids == ["m1", "m2"]


async def test_stimulus_without_chat_id_does_not_publish_read():
    plain = Stimulus(
        type="user_message", source="channel:cli", content="hi",
        timestamp=_T0,
    )
    reads = await _run_one_beat_collecting_events([plain])
    assert reads == []


async def test_quiet_beat_publishes_no_read_event():
    reads = await _run_one_beat_collecting_events([])
    assert reads == []
