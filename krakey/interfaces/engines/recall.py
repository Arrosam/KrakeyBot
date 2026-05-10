"""``RecallEngine`` — per-beat memory recall.

Recall is core flow: every heartbeat needs to know which memory
nodes are relevant to the current stimuli. The default impl
``IncrementalRecallEngine`` feeds stimuli into a session as they
arrive, expands neighbors, scores, applies a token budget, and
finalizes into a ``RecallResult``.

The session model: each heartbeat gets a fresh ``RecallSession`` from
the Engine via ``new_session()``. Sessions are short-lived (one beat),
hold per-beat state (``processed_stimuli``), and are discarded after
``finalize()``. The Engine itself is long-lived (one instance for the
runtime's life) and may carry caches that span sessions.

Two Protocols here (``RecallEngine`` + ``RecallSession``) because the
session lifecycle is part of the contract — heartbeat code calls
``engine.new_session()`` then ``session.add_stimuli(...)`` then
``session.finalize()``. Embedding both into one Protocol would force
session state onto the Engine instance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from krakey.models.stimulus import Stimulus


@dataclass
class RecallResult:
    """Output of a finalized recall session.

    ``nodes`` + ``edges`` populate the [GRAPH MEMORY] prompt layer.
    ``covered_stimuli`` / ``uncovered_stimuli`` partition the input;
    the heartbeat re-pushes uncovered ones (with a retry counter, capped
    by ``MAX_RECALL_RETRIES``) so a stimulus that found no relevant
    memory gets one chance for fresh memory writes to surface context
    on the next beat.
    """
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    covered_stimuli: list[Any] = field(default_factory=list)
    uncovered_stimuli: list[Any] = field(default_factory=list)


@runtime_checkable
class RecallSession(Protocol):
    """One beat's recall session. Created via ``RecallEngine.new_session``,
    fed stimuli incrementally during the idle preheat + drain phases,
    finalized once before prompt assembly.

    ``processed_stimuli`` is the read-only set of Stimulus objects the
    session has already ingested; the heartbeat consults it during
    drain to dedup against what idle's preheat already fed in.
    """

    processed_stimuli: list["Stimulus"]

    async def add_stimuli(self, stimuli: list["Stimulus"]) -> None: ...

    async def finalize(self) -> RecallResult: ...


@runtime_checkable
class RecallEngine(Protocol):
    """Long-lived recall Engine. The heartbeat calls ``new_session()``
    once per beat to get a fresh ``RecallSession``; the Engine instance
    itself outlives sessions and may carry caches across them.
    """

    def new_session(self) -> RecallSession: ...
