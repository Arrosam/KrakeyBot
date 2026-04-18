"""Fatigue helpers (DevSpec §11.2).

Phase-0 scope: only the hint picker for the [STATUS] block, so Self
sees a clear signal like "（精力充沛，无需睡眠）" at low fatigue.
The full `calculate_fatigue()` (node count → pct) lands in Phase 1.
"""
from __future__ import annotations


LOW_FATIGUE_HINT = "（精力充沛，无需睡眠）"


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
