"""Bootstrap state utilities — moved from krakey/bootstrap.py.

Pure helpers used by the BootstrapModifier:

  * ``parse_self_model_update`` — extract the JSON inside a
    ``<self-model>...</self-model>`` block from Self's [NOTE].
  * ``detect_bootstrap_complete`` — case-insensitive scan for the
    "bootstrap complete" completion marker in [NOTE].
  * ``load_genesis`` — read GENESIS.md from disk; return the
    placeholder string when the file is missing so the bot still
    boots.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_SELF_MODEL_TAG = re.compile(
    r"<self-?model>\s*(\{.*?\})\s*</self-?model>",
    re.DOTALL | re.IGNORECASE,
)

_BOOTSTRAP_COMPLETE = re.compile(r"bootstrap\s+complete", re.IGNORECASE)


GENESIS_PLACEHOLDER = (
    "(GENESIS.md does not exist — you are in a truly blank-slate state, with "
    "no birth letter left by a symbiont. Decide your identity and goals "
    "yourself during Bootstrap.)"
)


def parse_self_model_update(note_text: str | None) -> dict[str, Any] | None:
    """Extract the JSON inside a ``<self-model>...</self-model>`` block.

    Returns ``None`` when no block is present OR the JSON is malformed
    — the modifier silently skips bad updates rather than crashing
    the heartbeat.
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
    """Case-insensitive scan for the ``bootstrap complete`` marker."""
    if not note_text:
        return False
    return bool(_BOOTSTRAP_COMPLETE.search(note_text))


def load_genesis(path: str | Path) -> str:
    """Read GENESIS.md or return the built-in placeholder. Never
    raises — a missing/unreadable file just yields the placeholder
    string so the agent boots normally.
    """
    p = Path(path)
    if not p.exists():
        return GENESIS_PLACEHOLDER
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return GENESIS_PLACEHOLDER
