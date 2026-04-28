"""Heartbeat-tuning sections: hibernate cadence + fatigue thresholds."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HibernateSection:
    min_interval: int = 2
    max_interval: int = 300
    default_interval: int = 10


@dataclass
class FatigueSection:
    gm_node_soft_limit: int = 1000
    force_sleep_threshold: int = 1200
    thresholds: dict[int, str] = field(default_factory=lambda: {
        50: "（不繁忙时可以睡眠）",
        75: "（疲劳，需要主动睡眠）",
        100: "（非常疲劳，需要立即找到睡眠的机会）",
    })


def _build_hibernate(raw: dict[str, Any]) -> HibernateSection:
    d = HibernateSection()
    return HibernateSection(
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
