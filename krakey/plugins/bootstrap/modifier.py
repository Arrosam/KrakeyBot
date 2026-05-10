"""``BootstrapModifier`` — first-boot self-awareness, fully self-contained.

The modifier owns three behaviors:

  1. **Prompt injection** — ``modify_prompt`` writes the
     ``BOOTSTRAP_PROMPT`` (with embedded GENESIS text) into the
     ``bootstrap_intro`` element when bootstrap is active.
  2. **NOTE signal parsing** — the modifier subscribes to
     ``NoteEvent`` on the EventBus. When Self's [NOTE] contains a
     ``<self-model>`` JSON block the modifier deep-merges it into
     the persisted self_model. The modifier then checks completion
     criteria and auto-finalizes when met (see below).
  3. **Auto-completion + auto-disable** — the plugin decides when
     bootstrap is done; Self does NOT write any completion marker.
     Criterion: ``self_model.identity.name`` AND
     ``self_model.identity.persona`` both non-empty. On completion
     the plugin:
        * sets ``self_model.state.bootstrap_complete = True``,
        * writes ``bootstrap_done: true`` into its own
          ``workspace/plugins/bootstrap/config.yaml``,
        * deactivates itself for the rest of the session.

If a future runtime starts with the bootstrap plugin enabled in
``cfg.plugins`` AND the plugin's own config has ``bootstrap_done:
true``, the factory emits a stderr warning: bootstrap is already
done; either remove the plugin from ``cfg.plugins`` or set
``bootstrap_done: false`` to re-bootstrap.

Idle cadence is no longer runtime-pinned — Bootstrap teaches Self
via prompt to output ``[IDLE] 10`` while in bootstrap mode. The
runtime honors Self's [IDLE] field as it would in any state.

The modifier never imports from runtime/ outside the standard
plugin services dict + EventBus + the SelfModelStore class. Runtime
core has zero references to bootstrap concepts (CLAUDE.md additive-
plugin invariant).
"""
from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

from krakey.plugin_system.config import FilePluginConfigStore
from krakey.plugins.bootstrap.prompt import BOOTSTRAP_PROMPT
from krakey.plugins.bootstrap.state import (
    load_genesis,
    parse_self_model_update,
)

if TYPE_CHECKING:
    from krakey.interfaces.plugin_context import PluginContext
    from krakey.models.self_model import SelfModelStore


_log = logging.getLogger(__name__)


class BootstrapModifier:
    """Bootstrap-mode owner. Self-contained — every behavior reads
    from / writes through the services + events captured at
    construction time. The runtime touches it only via the standard
    Modifier surface."""

    name = "bootstrap"
    role = "bootstrap"

    def __init__(
        self,
        *,
        self_model_store: "SelfModelStore",
        events: Any,
        plugin_config_store: FilePluginConfigStore,
        plugin_config: dict[str, Any],
        genesis_path: str = "workspace/GENESIS.md",
    ):
        self._store = self_model_store
        self._events = events
        self._plugin_config_store = plugin_config_store
        # Snapshot of the per-plugin config at construction time —
        # other entries (genesis_path, future fields) are preserved
        # when the plugin writes back its bootstrap_done flag.
        self._plugin_config_initial = dict(plugin_config or {})
        self._genesis_path = genesis_path
        # Two independent "already done" markers must both be False
        # for the plugin to be active:
        #   * plugin own config: ``bootstrap_done: true``
        #   * self_model.state.bootstrap_complete: True
        sm = self._safe_load_self_model()
        own_done = bool(self._plugin_config_initial.get("bootstrap_done"))
        sm_done = bool(
            (sm or {}).get("state", {}).get("bootstrap_complete", False),
        )
        self._active = not (own_done or sm_done)
        self._genesis_text: str | None = None
        # Subscribe immediately — EventBus is alive by the time the
        # plugin loader builds this modifier.
        self._events.subscribe(self._on_event)
        if own_done:
            # The plugin is in cfg.plugins (otherwise we wouldn't
            # have been constructed) AND own config says we're done.
            # Either the user re-enabled the plugin without resetting
            # bootstrap_done, or they're keeping it enabled by
            # mistake. Either way: warn loudly so they notice. The
            # plugin then stays inactive for the session.
            print(
                "warning: bootstrap plugin is enabled but already "
                "marked complete (workspace/plugins/bootstrap/"
                "config.yaml has bootstrap_done: true). To "
                "re-bootstrap: set bootstrap_done: false in that "
                "file. Otherwise remove 'bootstrap' from cfg.plugins "
                "to silence this notice. The plugin will stay "
                "inactive until then.",
                file=sys.stderr,
            )

    # ---- Modifier protocol surface ---------------------------------

    def modify_prompt(self, elements) -> None:
        """Inject BOOTSTRAP_PROMPT into the ``bootstrap_intro`` element
        when bootstrap is active. PromptBuilder's
        DEFAULT_ELEMENT_KEYS pre-allocates the key at the head of
        the list so this writes a value into a known position
        rather than appending late."""
        if not self._active:
            return
        elements["bootstrap_intro"] = BOOTSTRAP_PROMPT.format(
            genesis_text=self._get_genesis_text(),
        )

    # ---- Event handlers --------------------------------------------

    def _on_event(self, event) -> None:
        """Single event-bus subscriber dispatching by event kind.
        Cheaper than registering N handlers each filtering with
        isinstance — same total work, simpler subscription."""
        kind = getattr(event, "kind", None)
        if kind == "note":
            self._handle_note(event)
        # RuntimeReadyEvent used to refine ``_active`` from GM/KB
        # emptiness, but that signal is no longer authoritative —
        # the plugin's own config (bootstrap_done) AND the
        # self_model marker fully determine activity at construction
        # time. A user who wants to re-bootstrap an existing agent
        # resets both flags explicitly.

    def _handle_note(self, event) -> None:
        """Parse Self's [NOTE] for self-model patches. After applying,
        check completion criteria — the plugin auto-completes when
        Self has set both identity.name and identity.persona."""
        if not self._active:
            return
        text = getattr(event, "text", "") or ""
        update = parse_self_model_update(text)
        if update:
            try:
                self._store.update(update)
            except Exception as e:  # noqa: BLE001
                _log.warning(
                    "bootstrap: self_model update raised %s; ignoring",
                    e,
                )
        self._maybe_complete()

    # ---- internals --------------------------------------------------

    def _maybe_complete(self) -> None:
        """Auto-finalize bootstrap when self_model.identity has both
        name + persona populated. Atomic-ish: writes self_model first
        (so the bootstrap_complete marker is visible to anything
        watching), then writes the plugin's own bootstrap_done flag,
        then flips the active bit. If the plugin-config write fails
        (disk error), self_model still says complete — next startup
        sees that and stays inactive even without the plugin marker."""
        if not self._active:
            return
        sm = self._safe_load_self_model() or {}
        identity = sm.get("identity") or {}
        name = (identity.get("name") or "").strip()
        persona = (identity.get("persona") or "").strip()
        if not (name and persona):
            return
        # Criteria met — finalize.
        try:
            self._store.update({"state": {"bootstrap_complete": True}})
        except Exception as e:  # noqa: BLE001
            _log.warning(
                "bootstrap: self_model completion write raised %s; "
                "modifier will retry next NoteEvent", e,
            )
            return
        try:
            updated_config = {
                **self._plugin_config_initial, "bootstrap_done": True,
            }
            self._plugin_config_store.write("bootstrap", updated_config)
            self._plugin_config_initial = updated_config
        except Exception as e:  # noqa: BLE001
            _log.warning(
                "bootstrap: own-config write raised %s; self_model "
                "still says complete so next startup will skip "
                "bootstrap regardless", e,
            )
        self._active = False
        _log.info(
            "bootstrap complete — identity set, plugin auto-disabled",
        )

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
        """Test-only — pin the active flag."""
        self._active = bool(value)


def build_modifier(ctx: "PluginContext") -> BootstrapModifier:
    """Factory invoked by ``load_component``. Pulls SelfModelStore +
    EventBus from ``ctx.services`` and the GENESIS path from the
    plugin's own config.yaml.

    Also constructs a ``FilePluginConfigStore`` rooted at the same
    plugin_configs directory the runtime uses, so the modifier can
    persist its ``bootstrap_done`` flag back to its own config file
    when bootstrap auto-completes.
    """
    from pathlib import Path
    services = ctx.services
    plugin_config = (
        ctx.config if isinstance(ctx.config, dict) else {}
    )
    genesis_path = (
        plugin_config.get("genesis_path") or "workspace/GENESIS.md"
    )
    plugin_root = Path(
        ctx.deps.plugin_configs_root or "workspace/plugins",
    )
    plugin_config_store = FilePluginConfigStore(plugin_root)
    return BootstrapModifier(
        self_model_store=services["self_model_store"],
        events=services["events"],
        plugin_config_store=plugin_config_store,
        plugin_config=plugin_config,
        genesis_path=genesis_path,
    )
