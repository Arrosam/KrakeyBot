"""ReflectRegistry — ordered storage + kind-specific dispatch.

The registry is a thin layer over a kind→list dict. Built-in defaults
register themselves at Runtime construction; future user Reflects
register from ``config.yaml`` declarations (not yet implemented).

Dispatch helpers (``translate``, ``make_recall``) encapsulate the
chain logic per kind so call sites in Runtime don't have to know how
multiple same-kind Reflects compose.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.reflects.protocol import (
    HypothalamusReflect, RecallAnchorReflect, Reflect,
)

if TYPE_CHECKING:
    from src.hypothalamus import HypothalamusResult
    from src.memory.recall import IncrementalRecall


class ReflectRegistry:
    """Ordered, kind-grouped registry for Reflects.

    Order within a kind = registration order. ``register`` appends
    to the kind's list; chain dispatch iterates the list head-to-tail.
    """

    def __init__(self):
        self._by_kind: dict[str, list[Reflect]] = {}

    # ---- registration ------------------------------------------------

    def register(self, reflect: Reflect) -> None:
        """Append a Reflect to its kind's chain."""
        if not getattr(reflect, "kind", None):
            raise ValueError(
                f"Reflect {reflect!r} missing required `kind` attribute"
            )
        if not getattr(reflect, "name", None):
            raise ValueError(
                f"Reflect {reflect!r} missing required `name` attribute"
            )
        self._by_kind.setdefault(reflect.kind, []).append(reflect)

    def by_kind(self, kind: str) -> list[Reflect]:
        """All Reflects of a given kind, in registration order."""
        return list(self._by_kind.get(kind, []))

    def names(self, kind: str | None = None) -> list[str]:
        """Names of all registered Reflects (optionally filtered by kind).
        Used by /status and dashboard display."""
        if kind is not None:
            return [r.name for r in self._by_kind.get(kind, [])]
        return [r.name for kind_list in self._by_kind.values()
                 for r in kind_list]

    # ---- kind-specific dispatch -------------------------------------

    def has_hypothalamus(self) -> bool:
        """Whether any kind=\"hypothalamus\" Reflect is registered.

        Drives the prompt-layer suppression for ``[ACTION FORMAT]``:
        when a hypothalamus Reflect is active it owns the translation
        path, so teaching Self the structured-tag syntax would create
        an interpretive conflict (Self would emit tags, Hypothalamus
        would also try to translate them). Suppress the teaching
        layer when Hypothalamus is on duty.
        """
        return bool(self._by_kind.get("hypothalamus"))

    async def dispatch_decision(
        self, self_text: str, decision: str,
        tentacles: list[dict[str, Any]],
    ) -> "HypothalamusResult":
        """Convert Self's response into structured tentacle calls.

        Picks the path based on registration:
          * hypothalamus Reflect registered → run translate (existing
            LLM-based path; Self's decision is natural language).
          * no hypothalamus Reflect → action executor parses
            ``[ACTION]...[/ACTION]`` JSONL out of ``self_text``.

        ``self_text`` is the raw Self LLM response; ``decision`` is
        the parsed [DECISION] section. Hypothalamus uses ``decision``
        (clean natural language); the action executor uses
        ``self_text`` (the structured tags can appear anywhere).
        """
        chain = self._by_kind.get("hypothalamus") or []
        if not chain:
            return self._dispatch_via_executor(self_text)
        if len(chain) > 1:
            raise NotImplementedError(
                "hypothalamus chain length > 1 — chain composition "
                "semantics will land when Reflect #2 forces them. "
                "For now only one hypothalamus Reflect at a time."
            )
        return await chain[0].translate(decision, tentacles)  # type: ignore[attr-defined]

    def _dispatch_via_executor(self, self_text: str) -> "HypothalamusResult":
        """Parse [ACTION] JSONL, wrap as a HypothalamusResult so the
        downstream call site doesn't care which path produced it.

        Local import — keeps the package's import graph free of a
        runtime → reflects → runtime cycle (action_executor lives
        under runtime/).
        """
        from src.hypothalamus import HypothalamusResult
        from src.runtime.action_executor import parse_action_block

        calls = parse_action_block(self_text)
        return HypothalamusResult(
            tentacle_calls=calls,
            memory_writes=[], memory_updates=[], sleep=False,
        )

    # Back-compat alias used by tests written before dispatch_decision
    # existed. Functionally equivalent to the hypothalamus-only path:
    # this method ignores the executor route on purpose so legacy
    # tests asserting "translate routes through hypothalamus chain"
    # still mean what they meant.
    async def translate(
        self, decision: str, tentacles: list[dict[str, Any]],
    ) -> "HypothalamusResult":
        chain = self._by_kind.get("hypothalamus") or []
        if not chain:
            raise RuntimeError(
                "no hypothalamus Reflect registered — register one or "
                "use dispatch_decision() to fall back to the executor"
            )
        if len(chain) > 1:
            raise NotImplementedError(
                "hypothalamus chain length > 1 — see dispatch_decision"
            )
        return await chain[0].translate(decision, tentacles)  # type: ignore[attr-defined]

    def make_recall(self, runtime: Any) -> "IncrementalRecall":
        """Build a fresh per-beat recall instance via the
        ``recall_anchor`` chain. Skeleton supports length-1 only."""
        chain = self._by_kind.get("recall_anchor") or []
        if not chain:
            raise RuntimeError(
                "no recall_anchor Reflect registered — runtime needs "
                "at least the default built-in"
            )
        if len(chain) > 1:
            raise NotImplementedError(
                "recall_anchor chain length > 1 — semantics will land "
                "with Reflect #2 (LLM anchor extractor). For now only "
                "the default built-in is allowed."
            )
        return chain[0].make_recall(runtime)  # type: ignore[attr-defined]
