import asyncio
from datetime import datetime

import pytest

from krakey.interfaces.channel import PushCallback, Channel
from krakey.models.stimulus import Stimulus
from krakey.runtime.stimuli.stimulus_buffer import StimulusBuffer


class MockChannel(Channel):
    def __init__(self, name: str, adrenalin: bool, content: str = "tick"):
        self._name = name
        self._adr = adrenalin
        self._content = content
        self.started = 0
        self.stopped = 0
        self._push: PushCallback | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def default_adrenalin(self) -> bool:
        return self._adr

    async def start(self, push: PushCallback) -> None:
        self.started += 1
        self._push = push
        await push(Stimulus(
            type="system_event", source=f"channel:{self._name}",
            content=self._content, timestamp=datetime.now(),
            adrenalin=self._adr,
        ))

    async def stop(self) -> None:
        self.stopped += 1


async def test_register_and_start_all_pushes_stimulus():
    buf = StimulusBuffer()
    m = MockChannel("mock", adrenalin=False, content="hello")
    buf.register(m)
    await buf.start_all()

    drained = buf.drain()
    assert [s.content for s in drained] == ["hello"]
    assert m.started == 1


async def test_pause_non_urgent_stops_only_calm_channels():
    buf = StimulusBuffer()
    calm = MockChannel("calm", adrenalin=False)
    urgent = MockChannel("urgent", adrenalin=True)
    buf.register(calm)
    buf.register(urgent)
    await buf.start_all()
    buf.drain()

    await buf.pause_non_urgent()
    assert calm.stopped == 1
    assert urgent.stopped == 0


async def test_resume_all_restarts_paused():
    buf = StimulusBuffer()
    calm = MockChannel("calm", adrenalin=False)
    buf.register(calm)
    await buf.start_all()
    buf.drain()

    await buf.pause_non_urgent()
    await buf.resume_all()
    assert calm.started == 2


def test_register_duplicate_raises():
    buf = StimulusBuffer()
    buf.register(MockChannel("x", False))
    with pytest.raises(ValueError):
        buf.register(MockChannel("x", False))


def test_get_channel_returns_none_for_unknown():
    buf = StimulusBuffer()
    buf.register(MockChannel("known", False))
    assert buf.get_channel("known") is not None
    assert buf.get_channel("nope") is None
    assert "known" in buf
    assert "nope" not in buf
