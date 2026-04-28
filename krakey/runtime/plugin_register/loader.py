"""PluginLoader — turn ``config.plugins`` names into live components.

The loader is fire-and-forget: ``register_from_config(deps)`` walks the
enabled-plugin list once at runtime startup, opens each plugin's
``meta.yaml`` (no scan), reads its per-plugin config, builds a
``PluginContext`` per declared component, invokes the factory, and
routes the returned instance to the right registry (reflect / tentacle
/ sensory).

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
    from krakey.interfaces.reflect import ReflectRegistry
    from krakey.interfaces.tentacle import TentacleRegistry
    from krakey.models.config import Config
    from krakey.runtime.runtime import RuntimeDeps
    from krakey.runtime.stimuli.stimulus_buffer import StimulusBuffer


class PluginLoader:
    """Loads plugins listed in ``config.plugins`` into the three runtime
    registries.

    Records what it registered as ``(kind, name)`` pairs in
    ``self.registered`` so the observer can distinguish loader-installed
    components from those registered by other paths (e.g. reflect
    ``attach()`` extras, BatchTracker)."""

    def __init__(
        self,
        *,
        config: "Config",
        reflects: "ReflectRegistry",
        tentacles: "TentacleRegistry",
        sensories: "StimulusBuffer",
        services: dict[str, Any],
    ):
        self._config = config
        self._reflects = reflects
        self._tentacles = tentacles
        self._sensories = sensories
        self._services = services
        self.registered: set[tuple[str, str]] = set()

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
        from krakey.interfaces.plugin_context import (
            PluginContext, load_plugin_config,
        )
        from krakey.plugin_system.loader import (
            load_component, load_plugin_meta,
        )

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

        cfg_root = Path(deps.plugin_configs_root or "workspace/plugins")

        for plugin_name in names:
            meta = load_plugin_meta(plugin_name)
            if meta is None:
                print(
                    f"config: unknown plugin {plugin_name!r} (no "
                    f"meta.yaml found in krakey/plugins/ or "
                    f"workspace/plugins/) — skipping.",
                    file=sys.stderr,
                )
                continue

            plugin_cfg = load_plugin_config(plugin_name, cfg_root)

            # Single ``plugin_cache`` dict shared across this plugin's
            # components — multi-component plugins (telegram /
            # in_mind_note / dashboard) use it to share state
            # between their factories.
            plugin_cache: dict[str, Any] = {}

            for component in meta.components:
                ctx = PluginContext(
                    deps=deps, plugin_name=plugin_name,
                    config=plugin_cfg,
                    services=self._services, plugin_cache=plugin_cache,
                )

                try:
                    instance = load_component(component, ctx)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"config: plugin {plugin_name!r} component "
                        f"{component.kind!r} failed to load: "
                        f"{type(e).__name__}: {e}; skipping.",
                        file=sys.stderr,
                    )
                    continue
                if instance is None:
                    continue  # factory opted out (e.g. unbound LLM)

                self._register_component(plugin_name, component, instance)

    def _register_component(
        self, plugin_name: str, component: Any, instance: Any,
    ) -> None:
        """Route a built component to the right registry by kind. On
        success, record ``(kind, name)`` in ``self.registered`` so
        the observer can label it as plugin-sourced."""
        kind = component.kind
        try:
            if kind == "reflect":
                self._reflects.register(instance)
            elif kind == "tentacle":
                self._tentacles.register(instance)
            elif kind == "sensory":
                self._sensories.register(instance)
            else:
                print(
                    f"config: plugin {plugin_name!r} produced unknown "
                    f"component kind {kind!r}; skipping",
                    file=sys.stderr,
                )
                return
        except Exception as e:  # noqa: BLE001
            print(
                f"config: plugin {plugin_name!r} {kind} registration "
                f"failed: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return
        self.registered.add((kind, instance.name))
