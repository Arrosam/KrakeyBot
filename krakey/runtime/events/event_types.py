"""Typed event dataclasses published on the runtime event bus.

Each event is a dataclass; the base class derives its ``kind`` string
(used for dashboard WS routing) from the class name. Subscribers live
on the dashboard side (see ``src.plugins.dashboard.events``); the runtime
only publishes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


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
class DecisionExecutedEvent(_BaseEvent):
    heartbeat_id: int
    tool_calls_count: int
    memory_writes_count: int
    memory_updates_count: int
    sleep_requested: bool


@dataclass
class DispatchEvent(_BaseEvent):
    heartbeat_id: int
    tool: str
    intent: str
    adrenalin: bool


@dataclass
class ToolResultEvent(_BaseEvent):
    tool: str
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
