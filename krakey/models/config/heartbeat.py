"""Heartbeat-tuning sections: idle cadence + fatigue thresholds."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IdleSection:
    min_interval: int = 2
    max_interval: int = 300
    default_interval: int = 10
    # Per-call wall-clock ceiling on a single self_llm.chat() inside
    # the heartbeat. Independent of LLM tag's timeout_seconds —
    # protects against a server stuck in an infinite generation loop.
    # When this fires the orchestrator's retry-idle loop counts it as
    # one failed attempt (and waits llm_failure_retry_interval before
    # the next try), NOT as a completed beat.
    self_max_wall_seconds: float = 1800.0
    # When self_llm.chat() raises (network error, timeout, server
    # 5xx, malformed response) the orchestrator sleeps this many
    # seconds and re-tries within the same beat. heartbeat_count does
    # NOT advance during these retries — failed LLM calls don't
    # pollute heartbeat history, fatigue accounting, or sleep
    # distance. Loop exits when chat succeeds OR
    # runtime.stop_requested goes True.
    llm_failure_retry_interval: float = 30.0
    # When Self returns a response but required structured tags
    # ([THINKING], [DECISION]) are missing, the orchestrator retries
    # within the same beat. Fast retries reuse llm_failure_retry_interval;
    # after exhausting the fast budget the loop switches to slow cadence.
    struct_output_fast_retries: int = 3
    struct_output_slow_retry_interval: float = 300.0


@dataclass
class FatigueSection:
    gm_node_soft_limit: int = 1000
    force_sleep_threshold: int = 1200
    thresholds: dict[int, str] = field(default_factory=lambda: {
        50: "(may sleep when not busy)",
        75: "(fatigued; should proactively sleep)",
        100: "(very fatigued; find an opportunity to sleep immediately)",
    })


@dataclass
class SlidingWindowSection:
    """Working-memory persistence (Samuel 2026-05-07).

    The sliding window holds the last N heartbeats' rounds in memory
    AND mirrors them to ``state_path`` after every mutation so a
    process restart restores the rounds exactly. Without persistence
    a restart wipes Self's recent context — only rounds already
    compacted to GM survive, the latest few in-flight ones vanish.

    To opt out (run pure in-memory, pre-2026-05-07 behavior), set
    ``state_path: ""`` (empty string). Empty also opts out via the
    YAML loader.
    """
    state_path: str = "workspace/data/sliding_window.json"
    compact_include_recall: bool = False


def _build_sliding_window(raw: dict[str, Any]) -> SlidingWindowSection:
    d = SlidingWindowSection()
    return SlidingWindowSection(
        state_path=str(raw.get("state_path", d.state_path)),
        compact_include_recall=bool(
            raw.get("compact_include_recall", d.compact_include_recall)
        ),
    )


def _build_idle(raw: dict[str, Any]) -> IdleSection:
    d = IdleSection()
    return IdleSection(
        min_interval=int(raw.get("min_interval", d.min_interval)),
        max_interval=int(raw.get("max_interval", d.max_interval)),
        default_interval=int(raw.get("default_interval",
                                       d.default_interval)),
        self_max_wall_seconds=float(raw.get(
            "self_max_wall_seconds", d.self_max_wall_seconds,
        )),
        llm_failure_retry_interval=float(raw.get(
            "llm_failure_retry_interval", d.llm_failure_retry_interval,
        )),
        struct_output_fast_retries=int(raw.get(
            "struct_output_fast_retries", d.struct_output_fast_retries,
        )),
        struct_output_slow_retry_interval=float(raw.get(
            "struct_output_slow_retry_interval",
            d.struct_output_slow_retry_interval,
        )),
    )


def _build_fatigue(raw: dict[str, Any]) -> FatigueSection:
    d = FatigueSection()
    if "thresholds" in raw:
        thresholds = {
            int(k): str(v) for k, v in (raw["thresholds"] or {}).items()
        }
    else:
        thresholds = d.thresholds
    return FatigueSection(
        gm_node_soft_limit=int(raw.get("gm_node_soft_limit",
                                         d.gm_node_soft_limit)),
        force_sleep_threshold=int(raw.get("force_sleep_threshold",
                                             d.force_sleep_threshold)),
        thresholds=thresholds,
    )


def _validate_fatigue_thresholds(f: FatigueSection) -> None:
    bad = [t for t in f.thresholds if t >= f.force_sleep_threshold]
    if bad:
        print(
            f"warning: fatigue threshold(s) {bad} >= force_sleep_threshold "
            f"({f.force_sleep_threshold}); force sleep will fire before hint shows.",
            file=sys.stderr,
        )
