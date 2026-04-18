"""Self-Model YAML store (DevSpec §13)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def default_self_model() -> dict[str, Any]:
    return {
        "identity": {"name": "", "persona": ""},
        "state": {
            "mood_baseline": "neutral",
            "energy_level": 1.0,
            "focus_topic": "",
            "is_sleeping": False,
            "bootstrap_complete": False,
        },
        "goals": {"active": [], "completed": []},
        "relationships": {"users": []},
        "statistics": {
            "total_heartbeats": 0,
            "total_sleep_cycles": 0,
            "uptime_hours": 0.0,
            "first_boot": "",
            "last_heartbeat": "",
            "last_sleep": "",
        },
    }


class SelfModelStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return default_self_model()
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        return raw

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def update(self, delta: dict[str, Any]) -> dict[str, Any]:
        current = self.load()
        merged = _deep_merge(current, delta)
        self.save(merged)
        return merged


def _deep_merge(base: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in delta.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out
