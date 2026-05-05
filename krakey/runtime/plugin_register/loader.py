"""PluginLoader — turn ``config.plugins`` names into live components.

The loader is fire-and-forget: ``register_from_config(deps)`` walks the
enabled-plugin list once at runtime startup, opens each plugin's
``meta.yaml`` (no scan), reads its per-plugin config, builds a
``PluginContext`` per declared component, invokes the factory, and
routes the returned instance to the right registry (modifier / tool
/ channel).

Failure modes are isolated per-plugin AND per-component (broken
metadata, factory exception, registry conflict): each writes one line
to stderr and skips. Strictly additive plugin model — a bad plugin
never blocks the rest of startup.

What this loader does NOT do:
  * Read ``llm_purposes`` (or any other plugin-internal field) out of
    the per-plugin config — that's the plugin's job. Loader hands the
    config dict through as ``ctx.config`` verbatim.
  * Write plugin config files — the dashboard owns that surface via
    its own ``FilePluginConfigStore``.
  * Track or report what got loaded — see ``PluginObserver`` in
    ``observer.py`` for that.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from krakey.interfaces.modifier import ModifierRegistry
    from krakey.interfaces.tool import ToolRegistry
    from krakey.models.config import Config
    from krakey.runtime.runtime import RuntimeDeps
    from krakey.runtime.stimuli.stimulus_buffer import StimulusBuffer


class PluginLoader:
    """Loads plugins listed in ``config.plugins`` into the three runtime
    registries.

    Records what it registered as ``(kind, name)`` pairs in
    ``self.registered`` so the observer can distinguish loader-installed
    components from those registered by other paths (e.g. modifier
    ``attach()`` extras, BatchTracker)."""

    def __init__(
        self,
        *,
        config: "Config",
        modifiers: "ModifierRegistry",
        tools: "ToolRegistry",
        channels: "StimulusBuffer",
        services: dict[str, Any],
    ):
        self._config = config
        self._modifiers = modifiers
        self._tools = tools
        self._channels = channels
        self._services = services
        self.registered: set[tuple[str, str]] = set()
        # Plugin-name-level history (separate from per-component
        # ``registered``). The hot-reload path uses this to skip
        # plugins already loaded.
        self.loaded_plugin_names: set[str] = set()

    def register_from_config(self, deps: "RuntimeDeps") -> None:
        """Walk ``config.plugins`` and lazily load each plugin's components.

        For each enabled plugin name:
          1. Look up its ``meta.yaml`` (no plugin code imported yet).
          2. Read its per-plugin config from
             ``<plugin_configs_root>/<name>/config.yaml`` and hand it
             through to the plugin AS-IS via ``ctx.config``.
          3. For each declared component:
             a. Build a ``PluginContext``.
             b. Lazy-import the factory module + invoke factory.
             c. Route the returned instance to the right registry.
        """
        names = self._config.plugins
        if names is None:
            print(
                "config: no `plugins:` section in config.yaml — "
                "starting with zero plugins. Use the dashboard or "
                "edit config.yaml to enable any.",
                file=sys.stderr,
            )
            return
        if not names:
            return  # explicit empty list: respect, no nag

        for plugin_name in names:
            self.register_one(plugin_name, deps)

    def register_one(
        self,
        plugin_name: str,
        deps: "RuntimeDeps",
    ) -> dict[str, Any]:
        """Load + register a single plugin by name. Used for both
        startup (looped via ``register_from_config``) and runtime
        hot-add (the dashboard's "apply changes" path).

        Returns a small report dict::

            {
              "ok":          bool,
              "plugin":      "<name>",
              "components":  [{kind, name}],   # newly registered
              "error":       str | None,       # set when ok=false
            }

        Side effect: every successfully-registered component is
        recorded in ``self.registered`` so subsequent calls + the
        plugin report can label it as plugin-sourced.
        """
        from krakey.interfaces.plugin_context import (
            PluginContext, load_plugin_config,
        )
        from krakey.plugin_system.loader import (
            load_component, load_plugin_meta,
        )

        cfg_root = Path(deps.plugin_configs_root or "workspace/plugins")
        report: dict[str, Any] = {
            "ok":         False,
            "plugin":     plugin_name,
            "components": [],
            "error":      None,
        }

        meta = load_plugin_meta(plugin_name)
        if meta is None:
            msg = (
                f"unknown plugin {plugin_name!r} (no meta.yaml found "
                f"in krakey/plugins/ or workspace/plugins/)"
            )
            print(f"config: {msg}; skipping.", file=sys.stderr)
            report["error"] = msg
            return report

        plugin_cfg = load_plugin_config(plugin_name, cfg_root)
        plugin_cache: dict[str, Any] = {}

        any_registered = False
        any_error: str | None = None
        for component in meta.components:
            ctx = PluginContext(
                deps=deps, plugin_name=plugin_name,
                config=plugin_cfg,
                services=self._services, plugin_cache=plugin_cache,
            )

            try:
                instance = load_component(component, ctx)
            except Exception as e:  # noqa: BLE001
                msg = (
                    f"component {component.kind!r} factory raised: "
                    f"{type(e).__name__}: {e}"
                )
                print(
                    f"config: plugin {plugin_name!r} {msg}; skipping.",
                    file=sys.stderr,
                )
                any_error = msg
                continue
            if instance is None:
                continue  # factory opted out (e.g. unbound LLM)

            registered_ok = self._register_component(
                plugin_name, component, instance,
            )
            if registered_ok:
                any_registered = True
                report["components"].append({
                    "kind": component.kind,
                    "name": getattr(instance, "name", "?"),
                })

        report["ok"] = any_registered
        if any_registered:
            self.loaded_plugin_names.add(plugin_name)
        if not any_registered and any_error is None:
            # Meta parsed and components iterated, but everything
            # opted out (None returns). Not strictly a failure.
            report["error"] = (
                "all components opted out (returned None) — check "
                "plugin config / LLM bindings"
            )
        elif not any_registered:
            report["error"] = any_error
        return report

    def _register_component(
        self, plugin_name: str, component: Any, instance: Any,
    ) -> bool:
        """Route a built component to the right registry by kind. On
        success, record ``(kind, name)`` in ``self.registered`` so
        the observer can label it as plugin-sourced. Returns
        True iff the component was successfully registered."""
        kind = component.kind
        try:
            if kind == "modifier":
                self._modifiers.register(instance)
            elif kind == "tool":
                self._tools.register(instance)
            elif kind == "channel":
                self._channels.register(instance)
            else:
                print(
                    f"config: plugin {plugin_name!r} produced unknown "
                    f"component kind {kind!r}; skipping",
                    file=sys.stderr,
                )
                return False
        except Exception as e:  # noqa: BLE001
            print(
                f"config: plugin {plugin_name!r} {kind} registration "
                f"failed: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return False
        self.registered.add((kind, instance.name))
        return True
