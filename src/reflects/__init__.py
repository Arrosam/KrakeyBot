"""Reflect plugin system — deeper-than-tentacle/sensory extension points.

A Reflect listens at heartbeat boundaries (``on_heartbeat_start`` /
``on_heartbeat_end``) and can also implement one or more ``kind``-specific
hooks that replace or augment a runtime mechanism. Examples:

  * ``kind="hypothalamus"`` — translates Self's natural-language
    ``[DECISION]`` into structured tentacle calls.
  * ``kind="recall_anchor"`` — produces the per-beat recall instance
    used to populate ``[GRAPH MEMORY]``.
  * ``kind="in_mind"`` — owns the persistent thoughts / mood / focus
    state Self can update each beat (planned).

Multiple Reflects of the same ``kind`` are allowed; they execute in
the order they were registered (``config.yaml`` ordering wins). The
chain semantics are kind-specific — see each kind's dispatch in
``ReflectRegistry``.

The 2026-04-25 skeleton refactor wraps the existing in-tree
``Hypothalamus`` and ``IncrementalRecall`` factory as the two default
built-in Reflects, with no behavior change. See
``docs/design/reflects-and-self-model.md`` for the full design.
"""
from src.reflects.protocol import (  # noqa: F401
    HeartbeatContext, HypothalamusReflect, RecallAnchorReflect, Reflect,
)
from src.reflects.registry import ReflectRegistry  # noqa: F401
