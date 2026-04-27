"""Runtime event bus for dashboard / external observers (Phase 3 / F).

The Logger stays the runtime's primary console output path. This bus is the
*additional* channel a Dashboard subscribes to. Runtime publishes typed
events; subscribers do whatever (broadcast WS, accumulate metrics, ignore).

Subscribers that raise are logged + quarantined; they cannot block runtime
progress. Async subscribers are scheduled via `asyncio.create_task` when an
event loop is running, so publish never awaits.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

_log = logging.getLogger(__name__)


# ---------------- event types ----------------


_CAMEL_BOUNDARY_AFTER_LOWER = re.compile(r"(?<=[a-z0-9])([A-Z])")
_CAMEL_BOUNDARY_BEFORE_LOWER = re.compile(r"(?<=[A-Z])([A-Z][a-z])")


def _to_snake(name: str) -> str:
    """PascalCase → snake_case, treating acronym runs as one token:
    GMStats → gm_stats, HeartbeatStart → heartbeat_start.
    """
    s = _CAMEL_BOUNDARY_BEFORE_LOWER.sub(r"_\1", name)
    s = _CAMEL_BOUNDARY_AFTER_LOWER.sub(r"_\1", s)
    return s.lower()


@dataclass
class _BaseEvent:
    @property
    def kind(self) -> str:
        name = type(self).__name__
        if name.endswith("Event"):
            name = name[:-5]
        return _to_snake(name)


@dataclass
class HeartbeatStartEvent(_BaseEvent):
    heartbeat_id: int
    stimulus_count: int


@dataclass
class GMStatsEvent(_BaseEvent):
    heartbeat_id: int
    node_count: int
    edge_count: int
    fatigue_pct: int


@dataclass
class StimuliQueuedEvent(_BaseEvent):
    """Snapshot of currently queued (post-drain) stimuli for the UI's
    pending list."""
    stimuli: list[dict[str, Any]]


@dataclass
class PromptBuiltEvent(_BaseEvent):
    heartbeat_id: int
    layers: dict[str, str]   # {dna, self_model, status, recall, history, stimulus}


@dataclass
class ThinkingEvent(_BaseEvent):
    heartbeat_id: int
    text: str


@dataclass
class DecisionEvent(_BaseEvent):
    heartbeat_id: int
    text: str


@dataclass
class NoteEvent(_BaseEvent):
    heartbeat_id: int
    text: str


@dataclass
class HypothalamusEvent(_BaseEvent):
    heartbeat_id: int
    tentacle_calls_count: int
    memory_writes_count: int
    memory_updates_count: int
    sleep_requested: bool
    raw_decision: str = ""


@dataclass
class DispatchEvent(_BaseEvent):
    heartbeat_id: int
    tentacle: str
    intent: str
    adrenalin: bool


@dataclass
class TentacleResultEvent(_BaseEvent):
    tentacle: str
    content: str


@dataclass
class HibernateEvent(_BaseEvent):
    heartbeat_id: int
    interval_seconds: float


@dataclass
class SleepStartEvent(_BaseEvent):
    reason: str


@dataclass
class SleepDoneEvent(_BaseEvent):
    stats: dict[str, Any]


@dataclass
class ChatMessageEvent(_BaseEvent):
    """Bidirectional web chat message (Phase 3.F.3)."""
    sender: str           # "user" | "krakey"
    content: str
    timestamp: str        # ISO format


# ---------------- bus ----------------


Subscriber = Callable[[_BaseEvent], Any]


class EventBus:
    def __init__(self):
        self._subs: list[Subscriber] = []

    def subscribe(self, callback: Subscriber) -> None:
        self._subs.append(callback)

    def unsubscribe(self, callback: Subscriber) -> None:
        try:
            self._subs.remove(callback)
        except ValueError:
            pass

    def publish(self, event: _BaseEvent) -> None:
        for cb in list(self._subs):
            try:
                if inspect.iscoroutinefunction(cb):
                    self._schedule(cb, event)
                else:
                    cb(event)
            except Exception:  # noqa: BLE001
                _log.exception("event subscriber raised; quarantining call")

    @staticmethod
    def _schedule(cb, event):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop — call sync (best-effort)
            try:
                asyncio.run(cb(event))
            except Exception:  # noqa: BLE001
                _log.exception("async subscriber failed (no loop)")
            return
        loop.create_task(cb(event))
