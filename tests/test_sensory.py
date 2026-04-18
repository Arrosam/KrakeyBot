import asyncio
from datetime import datetime

import pytest

from src.interfaces.sensory import Sensory, SensoryRegistry
from src.models.stimulus import Stimulus
from src.runtime.stimulus_buffer import StimulusBuffer


class MockSensory(Sensory):
    def __init__(self, name: str, adrenalin: bool, content: str = "tick"):
        self._name = name
        self._adr = adrenalin
        self._content = content
        self.started = 0
        self.stopped = 0
        self._buffer = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def default_adrenalin(self) -> bool:
        return self._adr

    async def start(self, buffer: StimulusBuffer) -> None:
        self.started += 1
        self._buffer = buffer
        await buffer.push(Stimulus(
            type="system_event", source=f"sensory:{self._name}",
            content=self._content, timestamp=datetime.now(),
            adrenalin=self._adr,
        ))

    async def stop(self) -> None:
        self.stopped += 1


async def test_register_and_start_all_pushes_stimulus():
    reg = SensoryRegistry()
    buf = StimulusBuffer()
    m = MockSensory("mock", adrenalin=False, content="hello")
    reg.register(m)
    await reg.start_all(buf)

    drained = buf.drain()
    assert [s.content for s in drained] == ["hello"]
    assert m.started == 1


async def test_pause_non_urgent_stops_only_calm_sensories():
    reg = SensoryRegistry()
    buf = StimulusBuffer()
    calm = MockSensory("calm", adrenalin=False)
    urgent = MockSensory("urgent", adrenalin=True)
    reg.register(calm)
    reg.register(urgent)
    await reg.start_all(buf)
    buf.drain()

    await reg.pause_non_urgent()
    assert calm.stopped == 1
    assert urgent.stopped == 0


async def test_resume_all_restarts_paused():
    reg = SensoryRegistry()
    buf = StimulusBuffer()
    calm = MockSensory("calm", adrenalin=False)
    reg.register(calm)
    await reg.start_all(buf)
    buf.drain()

    await reg.pause_non_urgent()
    await reg.resume_all(buf)
    assert calm.started == 2


def test_register_duplicate_raises():
    reg = SensoryRegistry()
    reg.register(MockSensory("x", False))
    with pytest.raises(ValueError):
        reg.register(MockSensory("x", False))
