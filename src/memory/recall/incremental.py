"""Recall surface — Protocols + null object.

What lives here:
  * ``AsyncEmbedder``  — Protocol for the embedder callable.
  * ``RecallResult``   — dataclass returned by ``finalize()``.
  * ``RecallLike``     — Protocol the runtime types ``self._recall``
                         against. Captures the duck-typed surface
                         consumed by Runtime + HeartbeatOrchestrator
                         + hibernate (``processed_stimuli`` attr,
                         ``add_stimuli``, ``finalize``). Any plugin
                         that implements this Protocol can claim the
                         ``recall_anchor`` Reflect role.
  * ``NoopRecall``     — concrete null-object used when no
                         ``recall_anchor`` Reflect is registered.
                         Honors the additive-plugin invariant
                         (CLAUDE.md): disabling the recall plugin
                         must not break the heartbeat.

What does NOT live here (intentionally):
  * The real ``IncrementalRecall`` driver — it ships with the
    ``default_recall_anchor`` plugin (``src/plugins/default_recall_anchor/
    incremental.py``). Core has zero references to it; disabling the
    plugin removes it entirely from the import graph.

Pure scoring helpers (``rank_candidates``, ``scripted_score``,
``ScoringWeights``, ``Reranker`` Protocol) live in
``src.memory.recall.scoring`` — they're math reusable across any
RecallLike implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class AsyncEmbedder(Protocol):
    async def __call__(self, text: str) -> list[float]: ...


@dataclass
class RecallResult:
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    covered_stimuli: list[Any] = field(default_factory=list)
    uncovered_stimuli: list[Any] = field(default_factory=list)


class RecallLike(Protocol):
    """Duck-typed surface the runtime depends on. Implementations:
    ``NoopRecall`` (in core) and ``IncrementalRecall`` (in the
    default_recall_anchor plugin).

    A read-only ``processed_stimuli`` list is exposed because the
    heartbeat's drain phase uses it to dedup stimuli already fed
    into recall during the prior hibernate's preheat pass.
    """
    processed_stimuli: list[Any]

    async def add_stimuli(self, stimuli: list[Any]) -> None: ...

    async def finalize(self) -> RecallResult: ...


class NoopRecall:
    """No-op stand-in. Returned by ``HeartbeatOrchestrator.new_recall``
    when no ``recall_anchor`` Reflect is registered. Self heartbeats
    with an empty ``[GRAPH MEMORY]`` layer — graceful degradation,
    not a crash.

    Implements the ``RecallLike`` Protocol surface:
    ``processed_stimuli`` (read), ``add_stimuli`` (no-op),
    ``finalize`` (returns empty ``RecallResult``).
    """

    def __init__(self) -> None:
        self.processed_stimuli: list[Any] = []

    async def add_stimuli(self, stimuli: list[Any]) -> None:
        # Track them as "processed" so the dedup logic in
        # _phase_drain_and_seed_recall doesn't keep re-feeding the
        # same Stimulus across beats (which would be harmless here
        # but pointless).
        self.processed_stimuli.extend(stimuli)

    async def finalize(self) -> RecallResult:
        return RecallResult()
