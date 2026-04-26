"""Sensory ABC + Registry (DevSpec §5.4).

Sensory = passive input channel. Registry tracks running state so
Sleep phase-1 can pause_non_urgent() and later resume_all().
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.runtime.stimulus_buffer import StimulusBuffer


class Sensory(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def default_adrenalin(self) -> bool:
        return False

    @abstractmethod
    async def start(self, buffer: StimulusBuffer) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...


class SensoryRegistry:
    def __init__(self):
        self._sensories: dict[str, Sensory] = {}
        self._running: set[str] = set()
        self._paused: set[str] = set()

    def register(self, sensory: Sensory) -> None:
        if sensory.name in self._sensories:
            raise ValueError(f"sensory '{sensory.name}' already registered")
        self._sensories[sensory.name] = sensory

    def get(self, name: str) -> Sensory:
        return self._sensories[name]

    async def start_all(self, buffer: StimulusBuffer) -> None:
        for s in self._sensories.values():
            if s.name not in self._running:
                await s.start(buffer)
                self._running.add(s.name)

    async def pause_non_urgent(self) -> None:
        for s in list(self._sensories.values()):
            if not s.default_adrenalin and s.name in self._running:
                await s.stop()
                self._running.discard(s.name)
                self._paused.add(s.name)

    async def resume_all(self, buffer: StimulusBuffer) -> None:
        for name in list(self._paused):
            s = self._sensories[name]
            await s.start(buffer)
            self._running.add(name)
            self._paused.discard(name)

    async def stop_all(self) -> None:
        for s in list(self._sensories.values()):
            if s.name in self._running:
                await s.stop()
                self._running.discard(s.name)

    def active_buffer(self) -> StimulusBuffer | None:
        """Return the StimulusBuffer that any currently-registered sensory
        was started with, or ``None`` if no sensory has a stashed buffer.

        Sleep's ``resume_all`` needs a buffer to hand to paused sensories
        when they restart; this avoids reaching into ``_sensories`` and
        each sensory's private ``_buffer`` from the orchestrator.
        """
        for s in self._sensories.values():
            buf = getattr(s, "_buffer", None)
            if buf is not None:
                return buf
        return None
