"""Sensory ABC (DevSpec §5.4).

Sensory = passive input channel. Each implementation knows how to
produce ``Stimulus`` objects from its own external surface (Telegram
poll, Web WS receive, batch-completion event, …) and ships each
stimulus by invoking the ``push`` callback handed to it at
``start()``.

Ownership inversion (Samuel 2026-04-26): sensories used to take a
``StimulusBuffer`` reference at start() and call buffer.push()
themselves — making the buffer (a high-level runtime object) a
dependency of every sensory implementation. Now the buffer owns
sensories, hands each one a bare push callback at start(), and the
sensory has no knowledge of (and no import on) the buffer class.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Awaitable, Callable

from src.models.stimulus import Stimulus


PushCallback = Callable[[Stimulus], Awaitable[None]]
"""Async callable a sensory invokes once per stimulus it produces.
The buffer (or any other consumer) supplies it at ``start()``."""


class Sensory(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def default_adrenalin(self) -> bool:
        return False

    @abstractmethod
    async def start(self, push: PushCallback) -> None:
        """Begin producing stimuli. Each call to ``push`` enqueues one
        stimulus into whatever consumer wired this sensory up."""

    @abstractmethod
    async def stop(self) -> None: ...
