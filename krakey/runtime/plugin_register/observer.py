"""PluginObserver — read-only snapshot of what's currently registered.

Walks the three registries (modifiers / tools / channels) to
produce a ``PluginInfo`` list and a dashboard-friendly dict. Pure
read; observer holds no state of its own beyond a back-reference to
the loader (used to label each component's ``source`` as either
"builtin" — registered by the loader — or "core" — registered some
other way, e.g. by a Modifier's ``attach()`` hook or directly by
runtime code).

Called every time the dashboard's /api/plugins endpoint hits, so
walking is cheap (registries are in-memory dicts/lists).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from krakey.interfaces.modifier import ModifierRegistry
    from krakey.interfaces.tool import ToolRegistry
    from krakey.runtime.plugin_register.loader import PluginLoader
    from krakey.runtime.stimuli.stimulus_buffer import StimulusBuffer


@dataclass
class PluginInfo:
    """Descriptor for one registered runtime component (modifier /
    tool / channel). Consumed by the dashboard's /api/plugins
    endpoint.

    ``path`` / ``source`` are kept for dashboard-JS compatibility:
    the frontend renderer keys off ``source`` ("builtin" vs "core")
    to decide whether to render an enable toggle. ``path`` is always
    "" in the meta.yaml flow but the JS template doesn't special-case
    missing keys.
    """
    name: str                           # component name
    kind: str                           # "modifier" | "tool" | "channel"
    source: str                         # "builtin" | "core"
    path: str                           # module path on disk, "" today
    project: str = ""                   # containing plugin folder name
    enabled: bool = True                # always True; kept for JS template


class PluginObserver:
    """Pure read of the three runtime registries; produces snapshots
    for the dashboard."""

    def __init__(
        self,
        *,
        modifiers: "ModifierRegistry",
        tools: "ToolRegistry",
        channels: "StimulusBuffer",
        loader: "PluginLoader",
    ):
        self._modifiers = modifiers
        self._tools = tools
        self._channels = channels
        self._loader = loader

    def collect_infos(self) -> list[PluginInfo]:
        """Snapshot every currently-registered component as a
        ``PluginInfo``. ``source`` is "builtin" if the loader
        registered it, "core" if it landed in the registry some other
        way (Modifier ``attach()``, BatchTracker, etc.).

        Walks the live registries fresh each call — cheap, since
        registries are in-memory dicts/lists."""
        infos: list[PluginInfo] = []
        for r in self._modifiers.all():
            infos.append(self._info("modifier", r.name))
        for t in self._tools.all():
            infos.append(self._info("tool", t.name))
        for sname in self._channels.channel_names():
            infos.append(self._info("channel", sname))
        return infos

    def loaded_report(self) -> dict[str, Any]:
        """Dashboard /api/plugins payload: tools + channels with
        a ``loaded`` flag (always True for items in the registry).

        Modifiers are not included in the report because the dashboard's
        plugins panel only renders tools + channels — modifiers
        live in their own panel that uses the catalogue scan, not this
        snapshot."""
        infos = self.collect_infos()
        loaded_t = set(self._tools.names())
        loaded_s = set(self._channels.channel_names())

        def _flatten(infos_subset, loaded_names):
            return [{
                "name": i.name,
                "kind": i.kind,
                "source": i.source,
                "project": i.project,
                "loaded": i.name in loaded_names,
                "error": None,
            } for i in infos_subset]

        return {
            "tools": _flatten(
                [i for i in infos if i.kind == "tool"], loaded_t,
            ),
            "channels": _flatten(
                [i for i in infos if i.kind == "channel"], loaded_s,
            ),
        }

    def _info(self, kind: str, name: str) -> PluginInfo:
        source = "builtin" if (kind, name) in self._loader.registered \
            else "core"
        return PluginInfo(
            name=name, kind=kind, source=source,
            path="", project=self._lookup_project(kind, name),
        )

    def _lookup_project(self, kind: str, name: str) -> str:
        """Resolve which plugin folder owns this ``(kind, name)`` tuple.

        The loader records every successful registration in its
        ``plugin_components`` manifest as ``plugin_name → [(kind,
        instance_name), ...]``. We invert that lookup here so the
        dashboard's per-plugin status badge can match the report
        against the catalog (which is keyed by plugin folder name,
        not component instance name — e.g. the ``duckduckgo_search``
        plugin contributes a tool whose ``.name`` is ``"search"``).

        Falls back to the component name for entries the loader
        didn't install (built-in tools, BatchTracker, modifier
        ``attach()`` extras) — those have no plugin folder, so the
        component name is the only stable identifier.
        """
        for plugin_name, components in self._loader.plugin_components.items():
            if (kind, name) in components:
                return plugin_name
        return name
