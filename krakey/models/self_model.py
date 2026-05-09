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


def load_self_model_or_default(
    path: str | Path,
) -> tuple[dict[str, Any], bool]:
    """Load self_model.yaml; create default if missing.

    Returns ``(self_model_dict, is_bootstrap)``. ``is_bootstrap``
    reflects whether the persisted self-model has been marked
    complete.

    Any keys present in the YAML but NOT in the current
    ``default_self_model()`` schema are silently dropped. If anything
    was dropped the cleaned version is rewritten back so the next
    boot is fast and the file matches what's actually in use.
    """
    import logging

    store = SelfModelStore(path)
    p = Path(path)
    if not p.exists():
        data = default_self_model()
        return data, True
    data = store.load()
    merged = _merge_defaults(default_self_model(), data)
    is_bootstrap = not bool(
        merged.get("state", {}).get("bootstrap_complete"),
    )
    if merged != data:
        dropped = _diff_keys(data, merged)
        logging.getLogger(__name__).info(
            "self_model migration: dropped legacy keys %s; rewriting %s",
            dropped, path,
        )
        store.save(merged)
    return merged, is_bootstrap


def _merge_defaults(defaults: dict, loaded: dict) -> dict:
    """Left-bounded deep-merge.

    Only keys present in ``defaults`` survive. Loaded values overlay
    defaults where the key matches; loaded keys not in defaults are
    silently dropped — this is the migration path for the slim-schema
    self-model refactor.
    """
    out: dict[str, Any] = {}
    for k, default_v in defaults.items():
        if k not in loaded:
            out[k] = copy.deepcopy(default_v)
            continue
        loaded_v = loaded[k]
        if isinstance(default_v, dict) and isinstance(loaded_v, dict):
            out[k] = _merge_defaults(default_v, loaded_v)
        else:
            out[k] = loaded_v
    return out


def _diff_keys(loaded: dict, merged: dict, prefix: str = "") -> list[str]:
    """Return dotted-paths for every key that appears in ``loaded``
    but not in ``merged``. Used only for the one-time migration log."""
    out: list[str] = []
    for k, v in loaded.items():
        path = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if k not in merged:
            out.append(path)
        elif isinstance(v, dict) and isinstance(merged.get(k), dict):
            out.extend(_diff_keys(v, merged[k], prefix=path))
    return out
