"""Stimulus Buffer + Sensory ownership (DevSpec §6.2).

The buffer is the runtime's central queue for stimuli + the owner of
every registered ``Sensory``. Two responsibilities, intentionally
fused so the dependency direction stays clean:

  * **Queue** — ``push`` (sensory → buffer), ``drain`` (buffer →
    heartbeat), ``peek_unrecalled`` (incremental recall scans),
    ``wait_for_*`` (hibernate wake-up).

  * **Sensory ownership** — ``register`` / ``start_all`` /
    ``pause_non_urgent`` / ``resume_all`` / ``stop_all``. Used to
    live in a separate ``SensoryRegistry`` next to the Sensory ABC,
    which forced ``interfaces/sensory.py`` to import this module —
    a "low-level interface depends on high-level runtime" smell.
    Inverted (Samuel 2026-04-26): buffer owns sensories, hands each
    its own ``push`` as a callback at start(), and Sensory.start
    takes ``Callable[[Stimulus], Awaitable[None]]`` instead of
    ``StimulusBuffer``.

Sensory implementations no longer import StimulusBuffer; they hold
the push callable instead. The buffer is the single point of
"what's the live set of sensories" and "what's the live stimulus
queue".
"""
from __future__ import annotations

import asyncio

from src.interfaces.sensory import Sensory
from src.models.stimulus import Stimulus


class StimulusBuffer:
    def __init__(self):
        # Queue
        self._queue: list[Stimulus] = []
        self._recalled_up_to: int = 0
        self._adrenalin_event = asyncio.Event()
        self._new_event = asyncio.Event()
        # Sensory ownership
        self._sensories: dict[str, Sensory] = {}
        self._running: set[str] = set()
        self._paused: set[str] = set()

    # ---- queue ----------------------------------------------------------

    async def push(self, s: Stimulus) -> None:
        self._queue.append(s)
        self._new_event.set()
        if s.adrenalin:
            self._adrenalin_event.set()

    def drain(self) -> list[Stimulus]:
        items = sorted(self._queue, key=lambda s: s.timestamp)
        self._queue = []
        self._recalled_up_to = 0
        self._adrenalin_event.clear()
        self._new_event.clear()
        return items

    def peek_unrecalled(self) -> list[Stimulus]:
        new = self._queue[self._recalled_up_to:]
        self._recalled_up_to = len(self._queue)
        self._new_event.clear()
        return new

    async def wait_for_adrenalin(self) -> None:
        await self._adrenalin_event.wait()

    async def wait_for_any(self) -> None:
        await self._new_event.wait()

    def has_adrenalin(self) -> bool:
        return self._adrenalin_event.is_set()

    # ---- sensory ownership ---------------------------------------------

    def register(self, sensory: Sensory) -> None:
        if sensory.name in self._sensories:
            raise ValueError(f"sensory '{sensory.name}' already registered")
        self._sensories[sensory.name] = sensory

    def get_sensory(self, name: str) -> Sensory | None:
        """Lookup by name — ``None`` if not registered. Used by the
        dashboard to wire up the web_chat sensory's ``push_user_message``
        callback when present (and to fall back to a noop log when
        absent)."""
        return self._sensories.get(name)

    def sensory_names(self) -> list[str]:
        """Snapshot of all registered sensory names — for the dashboard
        plugin report."""
        return list(self._sensories.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._sensories

    async def start_all(self) -> None:
        """Start every registered sensory that isn't already running.
        Hands each its own ``push`` so the sensory has no reference to
        ``self`` beyond the callable."""
        for s in self._sensories.values():
            if s.name not in self._running:
                await s.start(self.push)
                self._running.add(s.name)

    async def pause_non_urgent(self) -> None:
        """Sleep phase-1: stop sensories whose ``default_adrenalin`` is
        False (the calm ones). Urgent sensories (e.g. user-message
        channels) keep running so a wake-up signal can interrupt sleep."""
        for s in list(self._sensories.values()):
            if not s.default_adrenalin and s.name in self._running:
                await s.stop()
                self._running.discard(s.name)
                self._paused.add(s.name)

    async def resume_all(self) -> None:
        """Sleep wake-up: restart everything ``pause_non_urgent`` paused.
        Hands each its push callback again."""
        for name in list(self._paused):
            s = self._sensories[name]
            await s.start(self.push)
            self._running.add(name)
            self._paused.discard(name)

    async def stop_all(self) -> None:
        for s in list(self._sensories.values()):
            if s.name in self._running:
                await s.stop()
                self._running.discard(s.name)
