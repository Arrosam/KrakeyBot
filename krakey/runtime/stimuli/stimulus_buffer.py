"""Stimulus Buffer + Channel ownership (DevSpec §6.2).

The buffer is the runtime's central queue for stimuli + the owner of
every registered ``Channel``. Two responsibilities, intentionally
fused so the dependency direction stays clean:

  * **Queue** — ``push`` (channel → buffer), ``drain`` (buffer →
    heartbeat), ``peek_unrecalled`` (incremental recall scans),
    ``wait_for_*`` (idle wake-up).

  * **Channel ownership** — ``register`` / ``start_all`` /
    ``pause_non_urgent`` / ``resume_all`` / ``stop_all``. Used to
    live in a separate ``ChannelRegistry`` next to the Channel ABC,
    which forced ``interfaces/channel.py`` to import this module —
    a "low-level interface depends on high-level runtime" smell.
    Inverted (Samuel 2026-04-26): buffer owns channels, hands each
    its own ``push`` as a callback at start(), and Channel.start
    takes ``Callable[[Stimulus], Awaitable[None]]`` instead of
    ``StimulusBuffer``.

Channel implementations no longer import StimulusBuffer; they hold
the push callable instead. The buffer is the single point of
"what's the live set of channels" and "what's the live stimulus
queue".
"""
from __future__ import annotations

import asyncio

from krakey.interfaces.channel import Channel
from krakey.models.stimulus import Stimulus


class StimulusBuffer:
    def __init__(self):
        # Queue
        self._queue: list[Stimulus] = []
        self._recalled_up_to: int = 0
        self._adrenalin_event = asyncio.Event()
        self._new_event = asyncio.Event()
        # Channel ownership
        self._channels: dict[str, Channel] = {}
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

    # ---- channel ownership ---------------------------------------------

    def register(self, channel: Channel) -> None:
        if channel.name in self._channels:
            raise ValueError(f"channel '{channel.name}' already registered")
        self._channels[channel.name] = channel

    def get_channel(self, name: str) -> Channel | None:
        """Lookup by name — ``None`` if not registered. Used by the
        dashboard to wire up the web_chat channel's ``push_user_message``
        callback when present (and to fall back to a noop log when
        absent)."""
        return self._channels.get(name)

    def channel_names(self) -> list[str]:
        """Snapshot of all registered channel names — for the dashboard
        plugin report."""
        return list(self._channels.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._channels

    async def start_all(self) -> None:
        """Start every registered channel that isn't already running.
        Hands each its own ``push`` so the channel has no reference to
        ``self`` beyond the callable."""
        for s in self._channels.values():
            if s.name not in self._running:
                await s.start(self.push)
                self._running.add(s.name)

    async def pause_non_urgent(self) -> None:
        """Sleep phase-1: stop channels whose ``default_adrenalin`` is
        False (the calm ones). Urgent channels (e.g. user-message
        channels) keep running so a wake-up signal can interrupt sleep."""
        for s in list(self._channels.values()):
            if not s.default_adrenalin and s.name in self._running:
                await s.stop()
                self._running.discard(s.name)
                self._paused.add(s.name)

    async def resume_all(self) -> None:
        """Sleep wake-up: restart everything ``pause_non_urgent`` paused.
        Hands each its push callback again."""
        for name in list(self._paused):
            s = self._channels[name]
            await s.start(self.push)
            self._running.add(name)
            self._paused.discard(name)

    async def stop_all(self) -> None:
        for s in list(self._channels.values()):
            if s.name in self._running:
                await s.stop()
                self._running.discard(s.name)
