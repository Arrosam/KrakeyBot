"""Typed input shapes for ``PromptBuilder.build()``.

Three dataclasses crossing the runtime → builder boundary, replacing
the previous loose ``dict[str, Any]`` parameters. Defining them
together (vs scattering by consumer) makes "what does the prompt
builder consume?" answerable from one file.

Every field a producer typoes is a TypeError at construction; every
field a renderer reads wrong is an AttributeError. Both surface long
before the wrong number reaches Self.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SlidingWindowRound:
    """One past heartbeat as it appears in the [HISTORY] layer."""
    heartbeat_id: int
    stimulus_summary: str
    decision_text: str
    note_text: str


@dataclass
class StatusSnapshot:
    """Per-beat runtime numbers rendered in the [STATUS] layer.

    Replaces the previous ``status: dict[str, Any]`` contract. Used to
    be a free dict where producer / consumer / test fixture all kept
    their own copy of the schema; typoing a key (e.g. ``fatigue_pct``
    → ``fatigue_percent``) silently rendered the default value with no
    error. Now: producer constructs a ``StatusSnapshot``; field typo
    is a TypeError at construction.
    """
    gm_node_count: int
    gm_edge_count: int
    fatigue_pct: int
    fatigue_hint: str
    last_sleep_time: str
    heartbeats_since_sleep: int


@dataclass
class CapabilityView:
    """One row in the [CAPABILITIES] layer — name + one-line blurb.

    Replaces ``list[dict[str, Any]]`` with name/description keys.
    """
    name: str
    description: str
