import asyncio
from datetime import datetime

import pytest

from src.interfaces.sensory import PushCallback, Sensory
from src.models.stimulus import Stimulus
from src.runtime.stimuli.queue import StimulusQueue
from src.runtime.stimuli.sensory_registry import SensoryRegistry


class MockSensory(Sensory):
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
            type="system_event", source=f"sensory:{self._name}",
            content=self._content, timestamp=datetime.now(),
            adrenalin=self._adr,
        ))

    async def stop(self) -> None:
        self.stopped += 1


def _wired_pair():
    """Build a fresh queue + registry pair the way Runtime composes them."""
    q = StimulusQueue()
    return q, SensoryRegistry(push=q.push)


async def test_register_and_start_all_pushes_stimulus():
    q, reg = _wired_pair()
    m = MockSensory("mock", adrenalin=False, content="hello")
    reg.register(m)
    await reg.start_all()

    drained = q.drain()
    assert [s.content for s in drained] == ["hello"]
    assert m.started == 1


async def test_pause_non_urgent_stops_only_calm_sensories():
    q, reg = _wired_pair()
    calm = MockSensory("calm", adrenalin=False)
    urgent = MockSensory("urgent", adrenalin=True)
    reg.register(calm)
    reg.register(urgent)
    await reg.start_all()
    q.drain()

    await reg.pause_non_urgent()
    assert calm.stopped == 1
    assert urgent.stopped == 0


async def test_resume_all_restarts_paused():
    q, reg = _wired_pair()
    calm = MockSensory("calm", adrenalin=False)
    reg.register(calm)
    await reg.start_all()
    q.drain()

    await reg.pause_non_urgent()
    await reg.resume_all()
    assert calm.started == 2


def test_register_duplicate_raises():
    _, reg = _wired_pair()
    reg.register(MockSensory("x", False))
    with pytest.raises(ValueError):
        reg.register(MockSensory("x", False))


def test_get_sensory_returns_none_for_unknown():
    _, reg = _wired_pair()
    reg.register(MockSensory("known", False))
    assert reg.get_sensory("known") is not None
    assert reg.get_sensory("nope") is None
    assert "known" in reg
    assert "nope" not in reg
