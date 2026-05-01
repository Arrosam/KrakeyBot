"""Single owner of "are we still in bootstrap mode" state.

Bootstrap is a runtime-level temporary phase: the agent has just been
created, the GM is empty, the self-model has no learned data. While
active it changes three behaviors:

  1. **Prompt injection**     — BOOTSTRAP_PROMPT (incl. GENESIS text)
                                 prepends every Self prompt.
  2. **Hibernate cadence**    — fixed 10s heartbeat regardless of
                                 Self's [HIBERNATE] tag.
  3. **NOTE-signal parsing**  — Self's [NOTE] is scanned for
                                 ``self_model_update`` JSON and the
                                 ``bootstrap complete`` marker.

These three concerns used to be scattered across Runtime as
``if self.is_bootstrap`` checks. Collecting them into one object
that Runtime composes makes the runtime body uniform (it no longer
branches per-phase on the same flag) and gives Bootstrap a single
home for future evolution (multi-stage bootstrap, custom genesis
sources, etc.).

State changes happen in only two places:
  * ``__init__`` from the loaded self_model + optional override
  * ``refine_from_data`` once GM/KB counts are known
  * ``apply_note_signals`` flips to inactive on the completion marker

There is no public setter — ``runtime.is_bootstrap = True`` in tests
goes through ``Runtime``'s property which forwards to ``force_active``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from krakey.bootstrap import detect_bootstrap_complete, parse_self_model_update

if TYPE_CHECKING:
    from krakey.memory.graph_memory import GraphMemory
    from krakey.memory.knowledge_base import KBRegistry
    from krakey.models.self_model import SelfModelStore


# Bootstrap-mode heartbeat cadence (DevSpec §12.2). Constant; kept
# here so a hypothetical "slower bootstrap" never has to touch
# Runtime to override it — a future BootstrapCoordinator subclass
# can simply override ``hibernate_interval``.
_BOOTSTRAP_HEARTBEAT_SECONDS = 10


class BootstrapCoordinator:
    """Owns the Bootstrap-mode flag + the three behaviors it gates."""

    def __init__(
        self,
        *,
        self_model: dict[str, Any],
        self_model_store: "SelfModelStore",
        override: bool | None = None,
    ):
        self._store = self_model_store
        self._overridden = override is not None
        if override is not None:
            self._active = override
        else:
            # Default: bootstrap is active when the persisted self-model
            # has not been marked complete. The empty-data check in
            # refine_from_data tightens this to "AND no lived data".
            self._active = not _is_marked_complete(self_model)

    # ---- read surface -------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._active

    def should_inject_intro_prompt(self) -> bool:
        """True when the prompt builder should prepend BOOTSTRAP_PROMPT."""
        return self._active

    def hibernate_interval(self, default: int) -> int:
        """Return the bootstrap-cadence override when active, else
        ``default`` (passed in so the caller stays in control of what
        the non-bootstrap fallback is)."""
        return _BOOTSTRAP_HEARTBEAT_SECONDS if self._active else default

    # ---- state mutations ---------------------------------------------

    async def refine_from_data(
        self, gm: "GraphMemory", kb_registry: "KBRegistry",
    ) -> None:
        """Re-derive activity from actual workspace state. Bootstrap
        fires only when the workspace is genuinely empty — zero GM
        nodes AND zero KBs (active or archived). Otherwise the agent
        already has lived experience and shouldn't re-read GENESIS,
        regardless of what self_model.yaml says.

        Override path (constructor's ``override``) wins for tests.
        """
        if self._overridden:
            return
        try:
            n_nodes = await gm.count_nodes()
        except Exception:  # noqa: BLE001
            n_nodes = 0
        try:
            kbs = await kb_registry.list_kbs(include_archived=True)
        except Exception:  # noqa: BLE001
            kbs = []
        empty = n_nodes == 0 and len(kbs) == 0
        self._active = empty and not _is_marked_complete(self._store.load())

    def apply_note_signals(
        self, note: str | None,
    ) -> "_NoteSignalResult":
        """Parse Self's NOTE for self-model patches and the bootstrap-
        completion marker. Both writes go through the SelfModelStore
        so persistence stays transactional. Returns the patch that was
        applied (if any) plus whether the completion marker fired —
        Runtime uses the patch to refresh its self_model snapshot and
        to log the change.
        """
        update = parse_self_model_update(note or "")
        completed = detect_bootstrap_complete(note or "")
        if update:
            self._store.update(update)
        if completed:
            self._store.update({"state": {"bootstrap_complete": True}})
            self._active = False
        return _NoteSignalResult(update=update, completed=completed)

    def force_active(self, value: bool) -> None:
        """Test-only escape hatch matching the legacy
        ``runtime.is_bootstrap = ...`` mutation. Pins the flag without
        touching the persisted self-model."""
        self._active = value


class _NoteSignalResult:
    """Tiny return type so Runtime can branch on what happened without
    re-reading the store."""
    __slots__ = ("update", "completed")

    def __init__(self, update: dict[str, Any] | None, completed: bool):
        self.update = update
        self.completed = completed


def _is_marked_complete(self_model: dict[str, Any]) -> bool:
    return bool(
        (self_model or {}).get("state", {}).get("bootstrap_complete", False)
    )
