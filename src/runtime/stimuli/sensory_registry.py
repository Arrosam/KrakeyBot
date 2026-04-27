"""Sensory plugin registration + lifecycle.

Owns the live set of registered ``Sensory`` instances and their
running/paused state. Hands each sensory the ``push`` callable from
``StimulusQueue`` at start time so the sensory itself never sees the
queue object — exactly the inversion that lets
``interfaces/sensory.py`` stay free of the runtime import.

Sleep mode uses ``pause_non_urgent`` to silence calm sensories during
the 7-phase sleep cycle, then ``resume_all`` to bring them back.
"""
from __future__ import annotations

from src.interfaces.sensory import PushCallback, Sensory


class SensoryRegistry:
    """Owns the registered ``Sensory`` set + their start/stop lifecycle."""

    def __init__(self, push: PushCallback):
        # The push callable is plumbed into every sensory at start()
        # time. Stored once so we don't re-derive it on each restart
        # (resume_all needs the same callable as start_all).
        self._push = push
        self._sensories: dict[str, Sensory] = {}
        self._running: set[str] = set()
        self._paused: set[str] = set()

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
        """Start every registered sensory that isn't already running."""
        for s in self._sensories.values():
            if s.name not in self._running:
                await s.start(self._push)
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
        """Sleep wake-up: restart everything ``pause_non_urgent`` paused."""
        for name in list(self._paused):
            s = self._sensories[name]
            await s.start(self._push)
            self._running.add(name)
            self._paused.discard(name)

    async def stop_all(self) -> None:
        for s in list(self._sensories.values()):
            if s.name in self._running:
                await s.stop()
                self._running.discard(s.name)
