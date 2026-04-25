"""Reflect protocol shapes.

Two layers:

  1. ``Reflect`` — the minimum every Reflect satisfies: a name + kind +
     optional heartbeat lifecycle hooks. The registry only depends on
     this shape.

  2. ``HypothalamusReflect`` / ``RecallAnchorReflect`` — kind-specific
     Protocols that runtime call sites depend on. A concrete Reflect
     implements one (or more) of these depending on what mechanisms
     it overrides.

We use ``typing.Protocol`` rather than ABCs so built-in classes (like
the existing ``Hypothalamus``) can satisfy the interface by structure
without inheriting from anything. This keeps the existing classes
free of Reflect-specific imports and avoids a layered dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.hypothalamus import HypothalamusResult
    from src.memory.recall import IncrementalRecall


@dataclass
class HeartbeatContext:
    """Bundle passed to ``on_heartbeat_start`` / ``on_heartbeat_end``.

    Carries enough runtime references that a Reflect can read state
    or schedule side effects without needing the whole Runtime as
    an opaque parameter. Kept intentionally small — Reflects that
    need more should accept a dedicated ``runtime`` reference at
    construction time, not via this context.
    """
    heartbeat_id: int
    phase: str  # "start" | "end"


@runtime_checkable
class Reflect(Protocol):
    """Base shape — every Reflect has a name + kind."""
    name: str
    kind: str  # "hypothalamus" | "recall_anchor" | "in_mind" | ...


@runtime_checkable
class HypothalamusReflect(Protocol):
    """A Reflect that translates Self's [DECISION] text into structured
    tentacle calls. Kind = "hypothalamus".

    Multi-Reflect chain semantics (when more than one is registered):
    each subsequent Reflect can post-process the prior result; the
    chain dispatch in ``ReflectRegistry.translate`` defines the
    composition. The skeleton supports length-1 chains only; chain
    composition is finalized when Reflect #1 (toggle-able
    Hypothalamus + executor engine) lands.
    """
    name: str
    kind: str  # always "hypothalamus"

    async def translate(
        self, decision: str, tentacles: list[dict[str, Any]],
    ) -> "HypothalamusResult": ...


@runtime_checkable
class RecallAnchorReflect(Protocol):
    """A Reflect that builds the per-beat recall instance. Kind =
    "recall_anchor".

    The default in-tree Reflect wraps the existing scripted
    ``IncrementalRecall`` factory. A future LLM-anchor Reflect
    (Reflect #2) will produce a Recall driver that pre-extracts
    anchors from stimuli/history before running vec_search.

    The factory shape (``make_recall(runtime)``) preserves the
    existing per-run lifecycle: Runtime instantiates one Recall at
    ``run()`` start and re-instantiates whenever budget enforcement
    requires a fresh re-recall.
    """
    name: str
    kind: str  # always "recall_anchor"

    def make_recall(self, runtime: Any) -> "IncrementalRecall": ...


@runtime_checkable
class InMindReflect(Protocol):
    """A Reflect that owns Self's persistent "in-mind" state — the
    three short fields (``thoughts`` / ``mood`` / ``focus``) that
    capture what Self currently has at the front of its mental
    workspace. Kind = "in_mind".

    Architecture (see docs/design/reflects-and-self-model.md
    Reflect #3):

      * ``read()`` returns the current state dict; consumed by the
        prompt builder which prepends a "Heartbeat #now (in mind)"
        virtual round at the head of ``[HISTORY]``.
      * ``update(thoughts=, mood=, focus=)`` patches state and
        persists. ``None`` for a field = leave alone; empty string =
        clear; non-empty string = set.
      * ``attach(runtime)`` is the one-time lifecycle hook called by
        ``ReflectRegistry.attach_all`` after registration. The
        in_mind Reflect uses it to register its own ``update_in_mind``
        tentacle into ``runtime.tentacles``. Other Reflect kinds may
        also implement ``attach`` (the protocol allows it
        optionally), but in_mind is the first that needs it.
    """
    name: str
    kind: str  # always "in_mind"

    def read(self) -> dict[str, str]: ...

    def update(
        self,
        thoughts: str | None = None,
        mood: str | None = None,
        focus: str | None = None,
    ) -> dict[str, str]: ...

    def attach(self, runtime: Any) -> None: ...
