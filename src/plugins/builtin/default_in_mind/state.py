"""``in_mind`` state file persistence.

The state lives at ``workspace/data/in_mind.json`` (Samuel locked
2026-04-25: state files belong under ``workspace/data/`` alongside
``graph_memory.sqlite`` / ``web_chat.jsonl``; per-plugin user *config*
goes elsewhere — state ≠ config).

Format:

    {
      "thoughts": "...",
      "mood": "...",
      "focus": "...",
      "updated_at": "2026-04-25T..."
    }

Robustness:
  * Missing file → empty state.
  * Corrupted JSON → empty state + stderr warning. We don't crash:
    the agent has lost in_mind context but the runtime still runs.
  * Missing fields → empty strings. Forward-compat for adding fields.
  * Atomic write via temp + replace so a kill mid-write doesn't
    leave half-written JSON on disk.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class InMindState:
    thoughts: str = ""
    mood: str = ""
    focus: str = ""
    updated_at: str = ""  # ISO timestamp of last update; "" if never

    def is_empty(self) -> bool:
        """All three content fields blank → no virtual round to inject."""
        return not (self.thoughts or self.mood or self.focus)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def load(path: str | Path) -> InMindState:
    p = Path(path)
    if not p.exists():
        return InMindState()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(
            f"warning: in_mind state at {p} is unreadable ({e}); "
            "starting from empty state. Existing data is preserved on "
            "disk and will be overwritten on the next update_in_mind "
            "call.",
            file=sys.stderr,
        )
        return InMindState()
    if not isinstance(raw, dict):
        print(
            f"warning: in_mind state at {p} is not a JSON object; "
            "starting from empty state.", file=sys.stderr,
        )
        return InMindState()
    return InMindState(
        thoughts=str(raw.get("thoughts", "")),
        mood=str(raw.get("mood", "")),
        focus=str(raw.get("focus", "")),
        updated_at=str(raw.get("updated_at", "")),
    )


def save(state: InMindState, path: str | Path) -> None:
    """Atomic write: serialize, write to a temp sibling, ``os.replace``.

    The replace is atomic on every supported platform (POSIX +
    Windows since Python 3.3), so a crash mid-write either leaves
    the old file intact or the new one — never a half-written file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
    fd, tmp_name = tempfile.mkstemp(
        prefix=p.name + ".", suffix=".tmp", dir=str(p.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(serialized)
        os.replace(tmp_name, p)
    except Exception:
        # Best-effort: clean up the temp on failure so we don't leave
        # litter in workspace/data/.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def now_iso() -> str:
    """Centralized ISO timestamp — stub for future tests that want
    to monkeypatch the clock."""
    return datetime.now().isoformat(timespec="seconds")
