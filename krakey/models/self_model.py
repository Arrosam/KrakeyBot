"""Self-Model YAML store (DevSpec §13).

Schema is deliberately minimal — only what truly persists across runs
and is read by Self every beat:

  * ``identity.name`` / ``identity.persona`` — the unchanging core
    of who Self thinks it is.
  * ``state.bootstrap_complete`` — runtime gate for the Bootstrap
    prompt + the `<self-model>` write path.

Anything Self-authored about its current mental state (focus, mood,
goals) lives in Graph Memory as ``FOCUS`` / ``TARGET`` nodes (already
the canonical home for those concepts). Run-time bookkeeping (sleep
cycles, heartbeat counts, last-X timestamps) lives on the Runtime
object in memory; nothing in Self-model is "statistics".

This is the result of the 2026-04-25 self-model slim refactor —
see docs/design/modifiers-and-self-model.md (Part 1) for the full
rationale.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def default_self_model() -> dict[str, Any]:
    return {
        "identity": {"name": "", "persona": ""},
        "state": {"bootstrap_complete": False},
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
