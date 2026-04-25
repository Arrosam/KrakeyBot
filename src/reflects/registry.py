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

    async def translate(
        self, decision: str, tentacles: list[dict[str, Any]],
    ) -> "HypothalamusResult":
        """Run the hypothalamus chain. Skeleton supports length-1 only;
        when multi-Reflect chains land (Reflect #1), composition logic
        comes here."""
        chain = self._by_kind.get("hypothalamus") or []
        if not chain:
            raise RuntimeError(
                "no hypothalamus Reflect registered — runtime needs "
                "at least the default built-in"
            )
        if len(chain) > 1:
            raise NotImplementedError(
                "hypothalamus chain length > 1 — semantics will land "
                "with Reflect #1 (toggle-able Hypothalamus). For now "
                "only the default built-in is allowed."
            )
        # Type-narrow: chain[0] is a HypothalamusReflect by registration
        # contract. The Protocol is duck-typed, no isinstance check.
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
