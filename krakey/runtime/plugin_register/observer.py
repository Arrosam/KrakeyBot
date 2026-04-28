"""PluginObserver — read-only snapshot of what's currently registered.

Walks the three registries (reflects / tools / channels) to
produce a ``PluginInfo`` list and a dashboard-friendly dict. Pure
read; observer holds no state of its own beyond a back-reference to
the loader (used to label each component's ``source`` as either
"builtin" — registered by the loader — or "core" — registered some
other way, e.g. by a Reflect's ``attach()`` hook or directly by
runtime code).

Called every time the dashboard's /api/plugins endpoint hits, so
walking is cheap (registries are in-memory dicts/lists).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from krakey.interfaces.reflect import ReflectRegistry
    from krakey.interfaces.tool import ToolRegistry
    from krakey.runtime.plugin_register.loader import PluginLoader
    from krakey.runtime.stimuli.stimulus_buffer import StimulusBuffer


@dataclass
class PluginInfo:
    """Descriptor for one registered runtime component (reflect /
    tool / channel). Consumed by the dashboard's /api/plugins
    endpoint.

    ``path`` / ``source`` are kept for dashboard-JS compatibility:
    the frontend renderer keys off ``source`` ("builtin" vs "core")
    to decide whether to render an enable toggle. ``path`` is always
    "" in the meta.yaml flow but the JS template doesn't special-case
    missing keys.
    """
    name: str                           # component name
    kind: str                           # "reflect" | "tool" | "channel"
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
        reflects: "ReflectRegistry",
        tools: "ToolRegistry",
        channels: "StimulusBuffer",
        loader: "PluginLoader",
    ):
        self._reflects = reflects
        self._tools = tools
        self._channels = channels
        self._loader = loader

    def collect_infos(self) -> list[PluginInfo]:
        """Snapshot every currently-registered component as a
        ``PluginInfo``. ``source`` is "builtin" if the loader
        registered it, "core" if it landed in the registry some other
        way (Reflect ``attach()``, BatchTracker, etc.).

        Walks the live registries fresh each call — cheap, since
        registries are in-memory dicts/lists."""
        infos: list[PluginInfo] = []
        for r in self._reflects.all():
            infos.append(self._info("reflect", r.name))
        for t in self._tools.all():
            infos.append(self._info("tool", t.name))
        for sname in self._channels.channel_names():
            infos.append(self._info("channel", sname))
        return infos

    def loaded_report(self) -> dict[str, Any]:
        """Dashboard /api/plugins payload: tools + channels with
        a ``loaded`` flag (always True for items in the registry).

        Reflects are not included in the report because the dashboard's
        plugins panel only renders tools + channels — reflects
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
            path="", project=name,
        )
