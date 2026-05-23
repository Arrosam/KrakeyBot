"""Canonical helpers for writing and clearing the pause control file.

Owns the file format so every caller (CLI, runtime, tests) reads/writes
the same encoding without duplicating the atomic-write idiom.
"""
from __future__ import annotations

import os
import time
from pathlib import Path


def write_pause_file(path: Path, seconds: int | None = None) -> None:
    """Atomically write the pause control file.

    seconds is None  -> indefinite pause (file contents empty).
    seconds is an int -> auto-resume deadline = time.time() + seconds,
                         stored as a single decimal-float line.

    Creates parent directories if absent. Atomic via temp + os.replace
    so readers never see a half-written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "" if seconds is None else str(time.time() + seconds)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def clear_pause_file(path: Path) -> None:
    """Best-effort unlink; idempotent.

    Does NOT raise on FileNotFoundError or OSError (including
    PermissionError) — all such exceptions are swallowed.
    """
    try:
        path.unlink()
    except (FileNotFoundError, OSError):
        pass
