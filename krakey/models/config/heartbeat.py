"""Heartbeat-tuning sections: idle cadence + fatigue thresholds."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IdleSection:
    min_interval: int = 2
    max_interval: int = 300
    default_interval: int = 10


@dataclass
class FatigueSection:
    gm_node_soft_limit: int = 1000
    force_sleep_threshold: int = 1200
    thresholds: dict[int, str] = field(default_factory=lambda: {
        50: "(may sleep when not busy)",
        75: "(fatigued; should proactively sleep)",
        100: "(very fatigued; find an opportunity to sleep immediately)",
    })


@dataclass
class SlidingWindowSection:
    """Working-memory persistence (Samuel 2026-05-07).

    The sliding window holds the last N heartbeats' rounds in memory
    AND mirrors them to ``state_path`` after every mutation so a
    process restart restores the rounds exactly. Without persistence
    a restart wipes Self's recent context — only rounds already
    compacted to GM survive, the latest few in-flight ones vanish.

    To opt out (run pure in-memory, pre-2026-05-07 behavior), set
    ``state_path: ""`` (empty string). Empty also opts out via the
    YAML loader.
    """
    state_path: str = "workspace/data/sliding_window.json"


def _build_sliding_window(raw: dict[str, Any]) -> SlidingWindowSection:
    d = SlidingWindowSection()
    return SlidingWindowSection(
        state_path=str(raw.get("state_path", d.state_path)),
    )


def _build_idle(raw: dict[str, Any]) -> IdleSection:
    d = IdleSection()
    return IdleSection(
        min_interval=int(raw.get("min_interval", d.min_interval)),
        max_interval=int(raw.get("max_interval", d.max_interval)),
        default_interval=int(raw.get("default_interval",
                                       d.default_interval)),
    )


def _build_fatigue(raw: dict[str, Any]) -> FatigueSection:
    d = FatigueSection()
    if "thresholds" in raw:
        thresholds = {
            int(k): str(v) for k, v in (raw["thresholds"] or {}).items()
        }
    else:
        thresholds = d.thresholds
    return FatigueSection(
        gm_node_soft_limit=int(raw.get("gm_node_soft_limit",
                                         d.gm_node_soft_limit)),
        force_sleep_threshold=int(raw.get("force_sleep_threshold",
                                             d.force_sleep_threshold)),
        thresholds=thresholds,
    )


def _validate_fatigue_thresholds(f: FatigueSection) -> None:
    bad = [t for t in f.thresholds if t >= f.force_sleep_threshold]
    if bad:
        print(
            f"warning: fatigue threshold(s) {bad} >= force_sleep_threshold "
            f"({f.force_sleep_threshold}); force sleep will fire before hint shows.",
            file=sys.stderr,
        )
