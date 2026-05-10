"""``BootstrapModifier`` — first-boot self-awareness, fully self-contained.

The modifier owns three behaviors:

  1. **Prompt injection** — ``modify_prompt`` writes the
     ``BOOTSTRAP_PROMPT`` (with embedded GENESIS text) into the
     ``bootstrap_intro`` element while the modifier is active.
  2. **NOTE signal parsing** — the modifier subscribes to
     ``NoteEvent`` on the EventBus. When Self's [NOTE] contains a
     ``<self-model>`` JSON block the modifier deep-merges it into
     the persisted self_model. The modifier then checks completion
     criteria and auto-finalizes when met (see below).
  3. **Auto-completion + self-disable** — the plugin decides when
     bootstrap is done; Self does NOT write any completion marker.
     Criterion: ``self_model.identity.name`` AND
     ``self_model.identity.persona`` both non-empty. On completion
     the plugin:
        * sets ``self_model.state.bootstrap_complete = True``
          (informational — surfaces in ``/status`` and dashboard),
        * **removes "bootstrap" from the central config.yaml**'s
          ``modifiers:`` and ``plugins:`` lists so the plugin won't
          load on the next start,
        * deactivates itself for the rest of the session.

This means the dashboard's plugin checkbox is the **single source of
truth** for whether bootstrap runs: re-enable it = re-bootstrap. No
separate "bootstrap_done" flag, no double-marker reset dance.

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
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml as _yaml

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
        central_config_path: Path | None = None,
        genesis_path: str = "workspace/GENESIS.md",
    ):
        self._store = self_model_store
        self._events = events
        self._central_config_path = (
            Path(central_config_path) if central_config_path else None
        )
        self._genesis_path = genesis_path
        # Plugin is always active when loaded. The dashboard's plugin
        # checkbox (= presence in central config.yaml's modifiers/
        # plugins list) is the single signal — re-enabling the plugin
        # = re-running bootstrap.
        self._active = True
        self._genesis_text: str | None = None
        # Subscribe immediately — EventBus is alive by the time the
        # plugin loader builds this modifier.
        self._events.subscribe(self._on_event)

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
        name + persona populated. Order:
          1. write ``state.bootstrap_complete: True`` to self_model
             (informational — surfaces in /status and dashboard),
          2. remove "bootstrap" from the central config.yaml's
             modifier/plugin lists so the plugin won't load next start,
          3. flip the active bit.
        Step 2 failure (no config_path, disk error, parse error) is
        warned but not fatal — the worst case is one more bootstrap
        run on the next start, which the user can disable from the
        dashboard."""
        if not self._active:
            return
        sm = self._safe_load_self_model() or {}
        identity = sm.get("identity") or {}
        name = (identity.get("name") or "").strip()
        persona = (identity.get("persona") or "").strip()
        if not (name and persona):
            return
        try:
            self._store.update({"state": {"bootstrap_complete": True}})
        except Exception as e:  # noqa: BLE001
            _log.warning(
                "bootstrap: self_model completion write raised %s; "
                "modifier will retry next NoteEvent", e,
            )
            return
        self._remove_self_from_central_config()
        self._active = False
        _log.info(
            "bootstrap complete — identity set, plugin auto-disabled "
            "for future starts",
        )

    def _remove_self_from_central_config(self) -> None:
        """Remove "bootstrap" from the central config.yaml's
        ``modifiers:`` and ``plugins:`` lists. Round-trips through
        ``yaml.safe_load`` / ``safe_dump`` (loses comments, but the
        dashboard's settings-write path does the same — consistency
        beats comment preservation here)."""
        path = self._central_config_path
        if path is None:
            _log.warning(
                "bootstrap: no central config_path bound; cannot "
                "auto-remove from cfg.modifiers/plugins. Disable "
                "the plugin from the dashboard to prevent re-running",
            )
            return
        if not path.exists():
            _log.warning(
                "bootstrap: central config %s missing; cannot "
                "auto-remove", path,
            )
            return
        try:
            raw = path.read_text(encoding="utf-8")
            data = _yaml.safe_load(raw) or {}
        except (OSError, _yaml.YAMLError) as e:
            _log.warning(
                "bootstrap: cannot parse central config %s: %s; "
                "leaving entry in place", path, e,
            )
            return
        if not isinstance(data, dict):
            _log.warning(
                "bootstrap: central config %s top-level is not a "
                "mapping; leaving entry in place", path,
            )
            return
        changed = False
        for key in ("modifiers", "plugins"):
            current = data.get(key)
            if not isinstance(current, list):
                continue
            filtered = [n for n in current if n != "bootstrap"]
            if len(filtered) != len(current):
                data[key] = filtered
                changed = True
        if not changed:
            return
        try:
            new_raw = _yaml.safe_dump(
                data, allow_unicode=True, sort_keys=False,
            )
            path.write_text(new_raw, encoding="utf-8")
            _log.info(
                "bootstrap: removed 'bootstrap' from %s "
                "(re-enable from the dashboard to re-bootstrap)",
                path,
            )
        except (OSError, _yaml.YAMLError) as e:
            _log.warning(
                "bootstrap: failed to write %s: %s; entry remains "
                "and the plugin will run again next start", path, e,
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
    plugin's own config.yaml. Threads the central config.yaml path
    through so the modifier can self-remove on auto-complete."""
    services = ctx.services
    plugin_config = ctx.config if isinstance(ctx.config, dict) else {}
    genesis_path = (
        plugin_config.get("genesis_path") or "workspace/GENESIS.md"
    )
    central_config_path = ctx.deps.config_path
    return BootstrapModifier(
        self_model_store=services["self_model_store"],
        events=services["events"],
        central_config_path=(
            Path(central_config_path) if central_config_path else None
        ),
        genesis_path=genesis_path,
    )
