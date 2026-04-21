"""ANSI color helpers for the heartbeat console.

Convention:
- normal log lines: default (white-ish, no wrapping)
- Self's inner monologue (THINKING / DECISION / NOTE): cyan
- Bot's outward replies (tentacle output that the user sees as chat): green

Color is auto-disabled when stdout isn't a TTY or NO_COLOR is set
(https://no-color.org).
"""
from __future__ import annotations

import os
import sys


_RESET = "\033[0m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"


def _compute_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    stream = sys.stdout
    isatty = getattr(stream, "isatty", None)
    if isatty is None:
        return False
    try:
        return bool(isatty())
    except Exception:  # noqa: BLE001
        return False


_ENABLED = _compute_enabled()


def cyan(text: str) -> str:
    return f"{_CYAN}{text}{_RESET}" if _ENABLED else text


def green(text: str) -> str:
    return f"{_GREEN}{text}{_RESET}" if _ENABLED else text


def yellow(text: str) -> str:
    return f"{_YELLOW}{text}{_RESET}" if _ENABLED else text


def magenta(text: str) -> str:
    return f"{_MAGENTA}{text}{_RESET}" if _ENABLED else text
