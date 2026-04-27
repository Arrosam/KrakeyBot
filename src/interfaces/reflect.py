"""Reflect plugin interface — protocols + registry.

Sibling to ``sensory.py`` and ``tentacle.py``: defines the contract
runtime depends on and the registry it stores instances in. Concrete
Reflects live under ``src/plugins/builtin/default_*/``.

A Reflect listens at heartbeat boundaries (``on_heartbeat_start`` /
``on_heartbeat_end``) and can implement one or more ``kind``-specific
hooks that replace or augment a runtime mechanism:

  * ``kind="hypothalamus"`` — translates Self's natural-language
    ``[DECISION]`` into structured tentacle calls.
  * ``kind="recall_anchor"`` — produces the per-beat recall instance
    used to populate ``[GRAPH MEMORY]``.
  * ``kind="in_mind"`` — owns the persistent thoughts/mood/focus
    state Self can update each beat.

Multiple Reflects of the same ``kind`` are allowed; they execute in
registration order (``config.yaml`` ordering wins). Chain semantics
are kind-specific — see each kind's dispatch helper below.

Protocols (vs ABCs) so concrete classes don't have to inherit
anything; structural typing keeps plugin code free of interface
imports it doesn't need.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.memory.recall import IncrementalRecall


# --------------------------------------------------------------------
# Contract dataclasses — cross the Reflect ↔ runtime boundary
# --------------------------------------------------------------------


@dataclass
class TentacleCall:
    """Structured tentacle invocation produced by a hypothalamus
    Reflect's ``translate()``. Consumed by ``Runtime._dispatch`` and
    by the script-only action executor (when no hypothalamus is
    registered)."""
    tentacle: str
    intent: str
    params: dict[str, Any] = field(default_factory=dict)
    adrenalin: bool = False


@dataclass
class HypothalamusResult:
    """Aggregate result of one hypothalamus translation pass: the
    tentacle calls to dispatch, plus any memory side-effects and the
    sleep flag."""
    tentacle_calls: list[TentacleCall] = field(default_factory=list)
    memory_writes: list[dict[str, Any]] = field(default_factory=list)
    memory_updates: list[dict[str, Any]] = field(default_factory=list)
    sleep: bool = False


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


# --------------------------------------------------------------------
# Protocols — Reflect shapes the runtime depends on
# --------------------------------------------------------------------


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
    ) -> HypothalamusResult: ...


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

    The companion ``update_in_mind`` tentacle is contributed as a
    sibling component in ``meta.yaml``; the two share the reflect
    instance via ``ctx.plugin_cache`` rather than the reflect
    reaching into the runtime to register the tentacle itself.
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


# --------------------------------------------------------------------
# Registry — ordered storage + kind-specific dispatch
# --------------------------------------------------------------------


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

    def all(self) -> list[Reflect]:
        """Snapshot of every registered Reflect across all kinds, in
        registration order within each kind. Used by observers (e.g.
        the dashboard plugin report) that don't care about the
        kind-grouping."""
        return [r for kind_list in self._by_kind.values()
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

    def in_mind_state(self) -> dict[str, str] | None:
        """Snapshot of the in_mind state, or ``None`` if no in_mind
        Reflect is registered (zero-plugin invariant — runtime keeps
        working, prompt builder just won't insert the virtual
        round / instruction layer).

        Skeleton supports length-1 in_mind chain; multi-Reflect
        in_mind composition isn't a real use case yet — if more than
        one in_mind Reflect is registered, the FIRST one's state is
        returned and a warning is logged on the chain rather than
        raising (better to keep running with the first state than
        crash).
        """
        chain = self._by_kind.get("in_mind") or []
        if not chain:
            return None
        return chain[0].read()  # type: ignore[attr-defined]

    def attach_all(self, runtime: Any) -> None:
        """One-time post-registration lifecycle hook.

        Each registered Reflect that defines an ``attach`` method
        gets called with the runtime so it can wire up its own
        runtime-coupled assets — e.g. the in_mind Reflect uses this
        to register its ``update_in_mind`` tentacle into
        ``runtime.tentacles``.

        Errors in one Reflect's attach must not block the others —
        plugins are strictly additive (CLAUDE.md invariant).
        """
        import logging
        log = logging.getLogger(__name__)
        for kind_chain in self._by_kind.values():
            for reflect in kind_chain:
                attach = getattr(reflect, "attach", None)
                if attach is None:
                    continue
                try:
                    attach(runtime)
                except Exception as e:  # noqa: BLE001
                    log.warning(
                        "Reflect %r attach() raised %s; continuing "
                        "without its runtime hooks",
                        getattr(reflect, "name", "?"), e,
                    )

    async def dispatch_decision(
        self, self_text: str, decision: str,
        tentacles: list[dict[str, Any]],
    ) -> HypothalamusResult:
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

    def _dispatch_via_executor(self, self_text: str) -> HypothalamusResult:
        """Parse [ACTION] JSONL, wrap as a HypothalamusResult so the
        downstream call site doesn't care which path produced it.

        action_executor is imported lazily to avoid a
        ``runtime → interfaces → runtime`` import cycle.
        """
        from src.runtime.heartbeat.action_executor import parse_action_block

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
    ) -> HypothalamusResult:
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
        ``recall_anchor`` chain.

        **No registered Reflect → returns a ``NoopRecall``** (not an
        exception). This is load-bearing per Samuel's 2026-04-25
        principle: disabling any plugin must not break the runtime's
        core loop. Without recall, Self heartbeats with an empty
        ``[GRAPH MEMORY]`` layer — graceful degradation.

        Skeleton supports length-1 chains only; multi-Reflect
        composition lands when Reflect #2 forces semantics.
        """
        from src.memory.recall import NoopRecall

        chain = self._by_kind.get("recall_anchor") or []
        if not chain:
            return NoopRecall()  # type: ignore[return-value]
        if len(chain) > 1:
            raise NotImplementedError(
                "recall_anchor chain length > 1 — semantics will land "
                "with Reflect #2 (LLM anchor extractor). For now only "
                "the default built-in is allowed."
            )
        return chain[0].make_recall(runtime)  # type: ignore[attr-defined]
