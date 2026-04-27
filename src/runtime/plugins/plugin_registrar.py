"""Plugin registration + dashboard introspection — extracted from Runtime.

Owns two responsibilities that used to live as ``Runtime`` methods
on a 1300-line god class:

  1. **register_from_config(deps)** — walk ``config.plugins`` and
     lazily load each enabled plugin's meta.yaml components into
     the right registry (reflect / tentacle / sensory). The
     plugin's per-plugin config file is read here AND HANDED TO
     THE PLUGIN AS-IS via ``ctx.config`` — runtime does not parse
     ``llm_purposes`` or any other plugin-internal field. Plugins
     that need an LLM read their own ``llm_purposes`` and call
     ``ctx.get_llm_for_tag(tag_name)``.
  2. **derive_plugin_infos()** — build a ``PluginInfo`` list from the
     populated registries so observers (dashboard plugins endpoint)
     can describe what's loaded.

Runtime composes the registrar in its ``__init__``. The two
registry mutators (``_register_plugins_from_config`` and
``_register_component``) survive on Runtime as facades only because
two existing tests call them directly; new code should reach for
``runtime._plugin_registrar.register_from_config(deps)`` instead.

Plugin config WRITES are NOT a runtime concern: the dashboard owns
its own ``FilePluginConfigStore`` and writes plugin config files
directly. The runtime only ever READS them (during registration)
to build ``PluginContext.config`` for the plugin.

Failure modes are isolated per-plugin AND per-component (broken
metadata, factory exceptions, unbound LLM purposes) — they log to
stderr and skip without blocking startup of the rest. Strictly
additive plugin model, per CLAUDE.md.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.interfaces.reflect import ReflectRegistry
    from src.interfaces.tentacle import TentacleRegistry
    from src.models.config import Config
    from src.runtime.runtime import RuntimeDeps
    from src.runtime.stimuli.stimulus_buffer import StimulusBuffer


@dataclass
class PluginInfo:
    """Descriptor for one registered plugin component (reflect /
    tentacle / sensory). Consumed by the dashboard's /api/plugins
    endpoint via ``loaded_plugin_report()``.

    Originally lived in ``src.plugins.loader`` (the legacy MANIFEST
    loader) which has since been removed; the dataclass moved here
    when it became the only used export. Field shape preserved for
    dashboard JS compatibility — ``path`` / ``source`` / ``error``
    are stubs in the meta.yaml flow but kept so the frontend renderer
    doesn't have to special-case missing keys.
    """
    name: str                           # component name
    kind: str                           # "reflect" | "tentacle" | "sensory"
    source: str                         # "builtin" | "plugin"
    path: str                           # module path on disk, "" for core
    project: str = ""                   # containing plugin folder name
    description: str = ""
    config_schema: list[dict[str, Any]] = field(default_factory=list)
    # Always True for entries returned by ``derive_plugin_infos`` — a
    # PluginInfo only gets created if the plugin successfully loaded,
    # which only happens for names listed in ``config.yaml`` ``plugins:``.
    # The dashboard frontend keeps the field for display continuity.
    enabled: bool = True
    error: str | None = None
    # Set by the loader on success. Never JSON-serialised.
    instance: Any = None


class PluginRegistrar:
    """Loads meta.yaml plugins into the runtime's three registries +
    serves the dashboard's read/write surface for plugin config."""

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
        # ``sensories`` is the StimulusBuffer (it owns the registered
        # sensories now). Kept the ``sensories`` parameter name so
        # Runtime's call site stays self-documenting.
        self._sensories = sensories
        self._services = services
        self._infos: list = []

    # ---- registration ---------------------------------------------------

    def register_from_config(self, deps: "RuntimeDeps") -> None:
        """Walk ``config.plugins`` and lazily load each plugin's components.

        For each enabled plugin name:
          1. Look up its metadata (no plugin code imported yet).
          2. Read its per-plugin config from
             ``<plugin_configs_root>/<name>/config.yaml`` and hand
             it to the plugin AS-IS via ``ctx.config``. Runtime does
             not parse ``llm_purposes`` or any other plugin-internal
             field — that's the plugin's job.
          3. For each declared component:
             a. Build a ``PluginContext`` (carries config, services,
                a shared per-plugin cache, and ``get_llm_for_tag``
                for plugins that need an LLM).
             b. Lazy-import the factory module + invoke factory.
             c. Route the returned object to the right registry by
                component kind.
        """
        from src.interfaces.plugin_context import (
            PluginContext, load_plugin_config,
        )
        from src.plugin_system.loader import (
            load_component, load_plugin_meta,
        )

        names = self._config.plugins
        if names is None:
            # No `plugins:` section at all — keep startup quiet (the
            # full available-plugin list is the dashboard's job, not
            # the heartbeat's).
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
            # Load-by-name: open this plugin's meta.yaml directly. No
            # full filesystem scan — Runtime never enumerates plugins
            # the user didn't enable.
            meta = load_plugin_meta(plugin_name)
            if meta is None:
                print(
                    f"config: unknown plugin {plugin_name!r} (no "
                    f"meta.yaml found in src/plugins/ or "
                    f"workspace/plugins/) — skipping.",
                    file=sys.stderr,
                )
                continue

            plugin_cfg = load_plugin_config(plugin_name, cfg_root)

            # Single ``plugin_cache`` dict shared across this plugin's
            # components — multi-component plugins (telegram /
            # web_chat / default_in_mind) use it to share state
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
        """Route a built component to the right registry by kind."""
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
        except Exception as e:  # noqa: BLE001
            print(
                f"config: plugin {plugin_name!r} {kind} registration "
                f"failed: {type(e).__name__}: {e}",
                file=sys.stderr,
            )

    # ---- post-registration introspection -------------------------------

    def derive_plugin_infos(self) -> list:
        """Walk the populated registries and produce the ``PluginInfo``
        list the dashboard's ``/api/plugins`` endpoint expects.

        Used to be ``Runtime._load_plugins``; the legacy MANIFEST loader
        was removed in Phase 2 of the plugin unification, so this is
        now pure introspection. Stored on ``self._infos`` and returned
        for the caller's convenience.
        """
        infos: list = []
        for r in (self._reflects.by_kind("hypothalamus")
                  + self._reflects.by_kind("recall_anchor")
                  + self._reflects.by_kind("in_mind")):
            infos.append(PluginInfo(
                name=r.name, kind="reflect", source="builtin",
                path="", project=r.name, instance=r,
            ))
        for name in sorted(self._tentacles._tentacles.keys()):  # noqa: SLF001
            infos.append(PluginInfo(
                name=name, kind="tentacle", source="builtin",
                path="", project=name,
                instance=self._tentacles._tentacles[name],  # noqa: SLF001
            ))
        for sname in self._sensories.sensory_names():
            s = self._sensories.get_sensory(sname)
            infos.append(PluginInfo(
                name=sname, kind="sensory", source="builtin",
                path="", project=sname, instance=s,
            ))
        self._infos = infos
        return infos

    # ---- runtime observation (for the dashboard's plugins panel) -------

    def loaded_plugin_report(self) -> dict[str, Any]:
        """Pure runtime observation: which tentacles + sensories are
        actually live right now.

        Read-only — does NOT touch any plugin config files. The
        dashboard adapter combines this with its own
        ``FilePluginConfigStore`` reads to assemble the final
        ``/api/plugins`` payload (values come from the store, not
        from the runtime).

        Each entry carries name + kind + project + a ``loaded`` flag.
        Schema and description fields are intentionally omitted —
        those come from the plugin catalogue / meta.yaml on the
        dashboard side, not from the live runtime.
        """
        def _flatten(infos, loaded_names):
            return [{
                "name": i.name,
                "kind": i.kind,
                "source": i.source,
                "project": i.project,
                "loaded": i.name in loaded_names and i.error is None,
                "error": i.error,
            } for i in infos]

        plugin_t_names = {i.name for i in self._infos
                          if i.kind == "tentacle"}
        plugin_s_names = {i.name for i in self._infos
                          if i.kind == "sensory"}
        loaded_t = {n for n in self._tentacles._tentacles}  # noqa: SLF001
        loaded_s = set(self._sensories.sensory_names())

        core_tentacles = [
            {"name": t.name, "kind": "tentacle", "source": "core",
             "project": "", "loaded": True, "error": None}
            for t in self._tentacles._tentacles.values()  # noqa: SLF001
            if t.name not in plugin_t_names
        ]
        core_sensories = [
            {"name": sname, "kind": "sensory", "source": "core",
             "project": "", "loaded": True, "error": None}
            for sname in self._sensories.sensory_names()
            if sname not in plugin_s_names
        ]
        t_infos = [i for i in self._infos if i.kind == "tentacle"]
        s_infos = [i for i in self._infos if i.kind == "sensory"]
        return {
            "tentacles": core_tentacles + _flatten(t_infos, loaded_t),
            "sensories": core_sensories + _flatten(s_infos, loaded_s),
        }
