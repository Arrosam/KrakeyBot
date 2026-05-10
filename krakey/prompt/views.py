"""Typed input shapes for the Context Engine.

Two dataclasses crossing the runtime → context boundary, replacing
the previous loose ``dict[str, Any]`` parameters. Defining them
together (vs scattering by consumer) makes "what does the context
engine consume?" answerable from one file.

The history-round dataclass lives next to its producing Engine in
``krakey.interfaces.engines.explicit_history`` (``ExplicitHistoryRound``);
this module re-exports it so callers building prompt inputs still
import every shape from one place.

Every field a producer typoes is a TypeError at construction; every
field a renderer reads wrong is an AttributeError. Both surface long
before the wrong number reaches Self.
"""
from __future__ import annotations

from dataclasses import dataclass

from krakey.interfaces.engines.explicit_history import (  # noqa: F401
    ExplicitHistoryRound,
)


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
