"""``BootstrapModifier`` — first-boot self-awareness, fully self-contained.

The modifier owns three behaviors that previously lived inside the
runtime as ``BootstrapCoordinator``:

  1. **Prompt injection** — ``modify_prompt`` writes the
     ``BOOTSTRAP_PROMPT`` (with embedded GENESIS text) into a
     ``bootstrap_intro`` element when the agent is in bootstrap mode.
  2. **NOTE signal parsing** — the modifier subscribes to
     ``NoteEvent`` on the EventBus. When Self's [NOTE] contains a
     ``<self-model>`` JSON block the modifier deep-merges it into the
     persisted self_model; when it contains the ``bootstrap complete``
     marker the modifier flips ``state.bootstrap_complete`` and
     deactivates itself.
  3. **Active-state refinement** — on ``RuntimeReadyEvent`` the
     modifier inspects GM node count + KB count; if both are zero
     AND the persisted self_model isn't marked complete, bootstrap
     is active. Otherwise the agent has lived experience already
     and the modifier deactivates without injecting GENESIS.

Idle cadence is no longer runtime-pinned — Bootstrap teaches Self via
prompt to output ``[IDLE] 10`` while in bootstrap mode. The runtime
honors Self's [IDLE] field as it would in any other state.

GENESIS.md is loaded lazily on first ``modify_prompt`` call (only
relevant during bootstrap; steady-state agents never touch the file).
The path is configurable via the per-plugin ``config.yaml``.

The modifier never imports from the runtime; it touches only the
plugin services dict + the EventBus + the SelfModelStore. Runtime
core has zero references to bootstrap concepts (CLAUDE.md additive-
plugin invariant).
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from krakey.plugins.bootstrap.prompt import BOOTSTRAP_PROMPT
from krakey.plugins.bootstrap.state import (
    detect_bootstrap_complete,
    load_genesis,
    parse_self_model_update,
)

if TYPE_CHECKING:
    from krakey.interfaces.plugin_context import PluginContext
    from krakey.models.self_model import SelfModelStore


_log = logging.getLogger(__name__)


class BootstrapModifier:
    """Bootstrap-mode owner. Self-contained — every behavior reads
    from / writes through the services + events it captured at
    construction time. The runtime touches it only via the standard
    Modifier surface (``modify_prompt`` plus the registry's
    ``attach`` hook for one-time wiring)."""

    name = "bootstrap"
    role = "bootstrap"

    def __init__(
        self,
        *,
        self_model_store: "SelfModelStore",
        memory: Any,
        events: Any,
        genesis_path: str = "workspace/GENESIS.md",
    ):
        self._store = self_model_store
        self._memory = memory
        self._events = events
        self._genesis_path = genesis_path
        # Provisional active state from the persisted self_model.
        # Refined once we can probe gm/kb counts (RuntimeReadyEvent).
        sm = self._safe_load_self_model()
        self._active = not bool(
            (sm or {}).get("state", {}).get("bootstrap_complete", False),
        )
        self._genesis_text: str | None = None
        # Subscribe immediately — EventBus is alive by the time the
        # plugin loader builds this modifier.
        self._events.subscribe(self._on_event)

    # ---- Modifier protocol surface ---------------------------------

    def modify_prompt(self, elements) -> None:
        """Inject the BOOTSTRAP_PROMPT (with GENESIS text) into the
        prompt high above DNA when bootstrap is active. The element
        key ``bootstrap_intro`` doesn't exist in the default
        PromptBuilder element list, so this writes a NEW key — the
        builder appends unrecognized keys at the end of render.

        For the bootstrap intro to land high in the prompt cache
        order it would need to be a known element key in the default
        builder, which would couple the builder to the bootstrap
        plugin. Trade-off accepted: the intro renders late, which
        is suboptimal for prefix-cache hit rate during bootstrap
        but fine in practice (bootstrap is short-lived; the prefix
        cache rebuilds quickly).
        """
        if not self._active:
            return
        elements["bootstrap_intro"] = BOOTSTRAP_PROMPT.format(
            genesis_text=self._get_genesis_text(),
        )

    # ---- Event handlers --------------------------------------------

    def _on_event(self, event) -> None:
        """Single event-bus subscriber that dispatches by event kind.
        Cheaper than registering N handlers each filtering with
        isinstance — same total work, simpler subscription."""
        kind = getattr(event, "kind", None)
        if kind == "note":
            self._handle_note(event)
        elif kind == "runtime_ready":
            # Refine active state from gm/kb counts. async; schedule.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            loop.create_task(self._refine_active())

    def _handle_note(self, event) -> None:
        """Parse Self's [NOTE] for self-model patches + completion
        marker. Both writes go through the SelfModelStore so disk +
        the next-beat reload reflect the change atomically."""
        if not self._active:
            return
        text = getattr(event, "text", "") or ""
        update = parse_self_model_update(text)
        completed = detect_bootstrap_complete(text)
        if update:
            try:
                self._store.update(update)
            except Exception as e:  # noqa: BLE001
                _log.warning(
                    "bootstrap: self_model update raised %s; ignoring",
                    e,
                )
        if completed:
            try:
                self._store.update({"state": {"bootstrap_complete": True}})
            except Exception as e:  # noqa: BLE001
                _log.warning(
                    "bootstrap: completion-marker write raised %s; "
                    "modifier will retry next beat", e,
                )
                return
            self._active = False

    async def _refine_active(self) -> None:
        """RuntimeReadyEvent → re-derive active flag from actual
        workspace state. Bootstrap fires only when the workspace is
        genuinely empty: zero GM nodes AND zero KBs (active or
        archived). Otherwise the agent has lived experience and
        shouldn't re-read GENESIS regardless of what self_model.yaml
        says.
        """
        if not self._active:
            return
        try:
            n_nodes = await self._memory.count_nodes()
        except Exception:  # noqa: BLE001
            n_nodes = 0
        try:
            kbs = await self._memory.list_kbs(include_archived=True)
        except Exception:  # noqa: BLE001
            kbs = []
        empty = n_nodes == 0 and len(kbs) == 0
        sm = self._safe_load_self_model()
        marked_complete = bool(
            (sm or {}).get("state", {}).get("bootstrap_complete", False),
        )
        self._active = empty and not marked_complete

    # ---- internals --------------------------------------------------

    def _get_genesis_text(self) -> str:
        """Lazy-load GENESIS.md on first call. Cached — repeat
        bootstrap-mode beats don't re-read the file."""
        if self._genesis_text is None:
            self._genesis_text = load_genesis(self._genesis_path)
        return self._genesis_text

    def _safe_load_self_model(self) -> dict[str, Any] | None:
        try:
            return self._store.load()
        except Exception:  # noqa: BLE001
            return None

    # ---- introspection (for tests) ---------------------------------

    @property
    def is_active(self) -> bool:
        return self._active

    def force_active(self, value: bool) -> None:
        """Test-only — pin the active flag. Equivalent to the legacy
        ``runtime.is_bootstrap = ...`` setter."""
        self._active = bool(value)


def build_modifier(ctx: "PluginContext") -> BootstrapModifier:
    """Factory invoked by ``load_component``. Pulls SelfModelStore +
    MemoryEngine + EventBus from ``ctx.services`` and the GENESIS
    path from the plugin's own config.yaml.

    A user with a customised workspace layout sets
    ``genesis_path`` in
    ``workspace/plugins/bootstrap/config.yaml``; the default points
    at the repo's standard ``workspace/GENESIS.md``.
    """
    services = ctx.services
    genesis_path = (
        ctx.config.get("genesis_path") if isinstance(ctx.config, dict)
        else None
    ) or "workspace/GENESIS.md"
    return BootstrapModifier(
        self_model_store=services["self_model_store"],
        memory=services.get("memory") or services.get("gm"),
        events=services["events"],
        genesis_path=genesis_path,
    )
