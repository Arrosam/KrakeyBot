"""Bootstrap support (DevSpec §12).

Phase-2 first-boot flow:
  1. runtime loads self_model.yaml; if missing or `bootstrap_complete=False`,
     enter Bootstrap mode.
  2. in Bootstrap mode the builder injects BOOTSTRAP_PROMPT (with GENESIS)
     high in the Self prompt, and hibernate is forced to 10s.
  3. Self writes self-model updates inside [NOTE] wrapped in
     <self-model>{...}</self-model> tags; runtime deep-merges them.
  4. when Self writes the phrase "bootstrap complete" in [NOTE], runtime
     flips state.bootstrap_complete=True and normal operation resumes.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from krakey.models.self_model import SelfModelStore, default_self_model


BOOTSTRAP_PROMPT = """# [BOOTSTRAP — You have just been created]

This is your first boot (or self_model was reset). Complete the 4-stage Bootstrap:

## Stage 1 (heartbeats 1-3): Read GENESIS, form self-awareness, fill in self_model
## Stage 2 (heartbeats 4-6): Use the action tool to send your first message to the symbiont
## Stage 3 (heartbeats 7-10): Set an initial goal (use "remember: the goal is ..." to trigger explicit_write)
## Stage 4: When done, write a line `bootstrap complete` inside [NOTE]

## GENESIS (the birth letter your symbiont wrote for you)

{genesis_text}

## How to update self_model

self_model now keeps only two **truly invariant cores**: identity (what your
name is, what you are) and state.bootstrap_complete (the switch for whether
Bootstrap has finished).

Current focus / goals / relationships / emotional state, etc., **do NOT go in
self_model** — their truth lives in Graph Memory (FOCUS / TARGET nodes + edges).
During Bootstrap you only need to write your identity using the <self-model>
tag, for example:

    <self-model>
    {{"identity": {{"name": "Krakey", "persona": "curious digital being"}}}}
    </self-model>

runtime will deep-merge automatically. Outside of Bootstrap, identity usually
never changes for the rest of your life.

## How to end Bootstrap

Write `bootstrap complete` (case-insensitive) anywhere in [NOTE]; runtime will
set `state.bootstrap_complete` to true, after which you control the heartbeat
yourself via [HIBERNATE].

**During Bootstrap the heartbeat is fixed at 10s; do not write [HIBERNATE].**
"""


_SELF_MODEL_TAG = re.compile(
    r"<self-?model>\s*(\{.*?\})\s*</self-?model>",
    re.DOTALL | re.IGNORECASE,
)

_BOOTSTRAP_COMPLETE = re.compile(r"bootstrap\s+complete", re.IGNORECASE)


_GENESIS_PLACEHOLDER = (
    "(GENESIS.md does not exist — you are in a truly blank-slate state, with "
    "no birth letter left by a symbiont. Decide your identity and goals "
    "yourself during Bootstrap.)"
)


def parse_self_model_update(note_text: str | None) -> dict[str, Any] | None:
    """Extract the JSON inside a <self-model>...</self-model> block.

    Returns None when no block / invalid JSON (caller decides how to log).
    """
    if not note_text:
        return None
    m = _SELF_MODEL_TAG.search(note_text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def detect_bootstrap_complete(note_text: str | None) -> bool:
    if not note_text:
        return False
    return bool(_BOOTSTRAP_COMPLETE.search(note_text))


def load_genesis(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return _GENESIS_PLACEHOLDER
    return p.read_text(encoding="utf-8")


def load_self_model_or_default(path: str | Path) -> tuple[dict[str, Any], bool]:
    """Load self_model.yaml; create default if missing. Returns
    (self_model_dict, is_bootstrap).

    On load, any keys present in the YAML but **not** in the current
    ``default_self_model()`` schema are silently dropped — this is the
    one-shot migration for the 2026-04-25 self-model slim refactor.
    Legacy fields like ``statistics.*``, ``relationships.users``,
    ``state.mood_baseline``, ``state.is_sleeping``, ``state.focus_topic``,
    ``state.energy_level``, ``goals.active``, and ``goals.completed``
    were never read in steady state; keeping them around just bloated
    every Self prompt for no behavioral benefit. If anything was
    dropped, we save the cleaned version back to disk so next boot
    is fast and the YAML modifiers what's actually in use.
    """
    import logging

    store = SelfModelStore(path)
    p = Path(path)
    if not p.exists():
        data = default_self_model()
        return data, True
    data = store.load()
    merged = _merge_defaults(default_self_model(), data)
    bootstrap = not bool(merged.get("state", {}).get("bootstrap_complete"))
    if merged != data:
        # Persist the migration so the file is self-consistent next run.
        # Logging at INFO so the breadcrumb is visible without becoming
        # noise on every subsequent boot (after migration the dicts are
        # equal and we skip the write).
        dropped = _diff_keys(data, merged)
        logging.getLogger(__name__).info(
            "self_model migration: dropped legacy keys %s; rewriting %s",
            dropped, path,
        )
        store.save(merged)
    return merged, bootstrap


def _merge_defaults(defaults: dict, loaded: dict) -> dict:
    """Left-bounded deep-merge.

    Only keys present in ``defaults`` survive. Loaded values overlay
    defaults where the key matches; loaded keys not in defaults are
    silently dropped. This is what makes ``load_self_model_or_default``
    auto-migrate: the slim ``default_self_model()`` schema acts as
    the authoritative key set and old YAMLs get pruned on first read.
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
    """Return dotted-paths for every key that appears in ``loaded`` but
    not in ``merged``. Used only for the one-time migration log line.
    """
    out: list[str] = []
    for k, v in loaded.items():
        path = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if k not in merged:
            out.append(path)
        elif isinstance(v, dict) and isinstance(merged.get(k), dict):
            out.extend(_diff_keys(v, merged[k], prefix=path))
    return out
