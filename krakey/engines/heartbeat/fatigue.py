"""Fatigue helpers (DevSpec §11.2).

Phase-0: hint picker.
Phase-1: calculate_fatigue(node_count, soft_limit) → (pct, hint).
"""
from __future__ import annotations


LOW_FATIGUE_HINT = "(energy is high; no need to sleep)"


def calculate_fatigue(node_count: int, soft_limit: int,
                       thresholds: dict[int, str]) -> tuple[int, str]:
    """Returns (pct, hint).  pct = node_count / soft_limit × 100, integer.
    hint = fatigue_hint(pct, thresholds).  Soft-limit=0 is treated as 0%.
    """
    if soft_limit <= 0:
        pct = 0
    else:
        pct = int((node_count / soft_limit) * 100)
    return pct, fatigue_hint(pct, thresholds)


def fatigue_hint(pct: float, thresholds: dict[int, str]) -> str:
    """Pick the hint matching the highest threshold ≤ pct.

    Below the smallest threshold we emit an explicit low-fatigue hint so
    small models don't default to "idle → sleep".
    Empty thresholds → empty string (caller may have disabled hints).
    """
    if not thresholds:
        return ""
    matched: int | None = None
    for t in sorted(thresholds):
        if pct >= t:
            matched = t
        else:
            break
    if matched is None:
        return LOW_FATIGUE_HINT
    return thresholds[matched]
