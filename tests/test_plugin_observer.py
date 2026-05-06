"""Regression tests for ``PluginObserver.loaded_report``.

The dashboard's plugin panel groups loaded-status by plugin folder
name (the catalog is keyed by folder), so the report's ``project``
field MUST be the plugin folder name, NOT the component instance
name. They diverge whenever a plugin's tool/channel ``.name`` is the
abstract verb (e.g. ``duckduckgo_search`` → tool ``"search"``,
``dashboard`` plugin → channel ``"web_chat"``).

Pre-fix bug: ``_info`` set ``project=name`` (component name), so the
JS lookup ``catalogByName[entry.project]`` missed every such plugin
and rendered "not loaded" — confusing because the plugin WAS running.
"""
from __future__ import annotations

from typing import Any


class _StubLoader:
    """Captures the ``plugin_components`` shape the real loader
    exposes — just enough for the observer to do its lookup."""

    def __init__(
        self,
        *,
        plugin_components: dict[str, list[tuple[str, str]]],
        registered: set[tuple[str, str]] | None = None,
    ):
        self.plugin_components = plugin_components
        self.registered = registered or set()


class _StubModRegistry:
    def __init__(self, modifiers): self._mods = list(modifiers)
    def all(self): return list(self._mods)
    def names(self): return [m.name for m in self._mods]
    def by_role(self, role): return None  # not used here


class _StubToolRegistry:
    def __init__(self, tools): self._tools = tools
    def all(self): return list(self._tools)
    def names(self): return [t.name for t in self._tools]


class _StubBuffer:
    def __init__(self, channel_names: list[str]):
        self._names = list(channel_names)
    def channel_names(self): return list(self._names)


class _NamedThing:
    def __init__(self, name: str): self.name = name


def _make_observer(
    *,
    modifiers=(),
    tools=(),
    channels=(),
    plugin_components: dict[str, list[tuple[str, str]]] | None = None,
    registered=None,
):
    from krakey.runtime.plugin_register.observer import PluginObserver
    return PluginObserver(
        modifiers=_StubModRegistry([_NamedThing(n) for n in modifiers]),
        tools=_StubToolRegistry([_NamedThing(n) for n in tools]),
        channels=_StubBuffer(list(channels)),
        loader=_StubLoader(
            plugin_components=plugin_components or {},
            registered=registered or set(),
        ),
    )


# =====================================================================
# project lookup — the dashboard joins on this field
# =====================================================================


def test_project_field_uses_plugin_folder_when_tool_name_differs():
    """``duckduckgo_search`` plugin contributes a tool whose
    ``.name`` is the abstract verb ``"search"``. The report's
    ``project`` must point back at the plugin folder name so the
    dashboard's catalog lookup hits."""
    obs = _make_observer(
        tools=["search"],
        plugin_components={
            "duckduckgo_search": [("tool", "search")],
        },
        registered={("tool", "search")},
    )
    report = obs.loaded_report()
    assert len(report["tools"]) == 1
    entry = report["tools"][0]
    assert entry["name"] == "search"
    assert entry["project"] == "duckduckgo_search"
    assert entry["loaded"] is True


def test_project_field_uses_plugin_folder_for_channels_too():
    """``dashboard`` plugin contributes a channel ``"web_chat"``;
    catalog row is keyed ``dashboard`` so project must agree."""
    obs = _make_observer(
        channels=["web_chat"],
        plugin_components={
            "dashboard": [("channel", "web_chat")],
        },
        registered={("channel", "web_chat")},
    )
    report = obs.loaded_report()
    assert len(report["channels"]) == 1
    entry = report["channels"][0]
    assert entry["name"] == "web_chat"
    assert entry["project"] == "dashboard"
    assert entry["loaded"] is True


def test_project_field_handles_plugin_with_multiple_components():
    """``in_mind_note`` ships both a modifier and a tool; both
    must resolve back to the same plugin folder."""
    obs = _make_observer(
        modifiers=["in_mind"],
        tools=["update_in_mind"],
        plugin_components={
            "in_mind_note": [
                ("modifier", "in_mind"),
                ("tool", "update_in_mind"),
            ],
        },
        registered={("modifier", "in_mind"), ("tool", "update_in_mind")},
    )
    report = obs.loaded_report()
    # Tool entry
    tool_entries = [e for e in report["tools"] if e["name"] == "update_in_mind"]
    assert len(tool_entries) == 1
    assert tool_entries[0]["project"] == "in_mind_note"


def test_project_falls_back_to_component_name_for_builtin_tools():
    """Built-in tools (InstallTool, SleepTool) aren't in
    ``plugin_components`` because the loader didn't register them.
    Falling back to the component name keeps the existing
    no-plugin-folder behavior — the dashboard catalog never has a
    matching row, so the entry just doesn't show up as a plugin
    status, which is correct."""
    obs = _make_observer(
        tools=["install", "sleep"],
        plugin_components={},  # no plugins loaded
        registered=set(),  # not registered by loader
    )
    report = obs.loaded_report()
    by_name = {e["name"]: e for e in report["tools"]}
    assert by_name["install"]["project"] == "install"
    assert by_name["sleep"]["project"] == "sleep"
    # source label should be "core" (not loader-registered)
    assert by_name["install"]["source"] == "core"


def test_loaded_flag_still_true_for_registered_components():
    """The ``loaded`` field is independent of the project lookup —
    pin that the fix didn't break it."""
    obs = _make_observer(
        tools=["search"],
        plugin_components={"duckduckgo_search": [("tool", "search")]},
        registered={("tool", "search")},
    )
    report = obs.loaded_report()
    assert report["tools"][0]["loaded"] is True


# =====================================================================
# Modifier reporting — modifier-only plugins must surface, otherwise
# the dashboard's "loaded" badge falls through to "not loaded" for
# every such plugin (e.g. hypothalamus).
# =====================================================================


def test_loaded_report_includes_modifiers_bucket():
    """Pre-fix: ``loaded_report`` emitted only ``tools`` + ``channels``,
    so a modifier-only plugin like ``hypothalamus`` couldn't be
    badged correctly. Now emit a ``modifiers`` bucket too."""
    obs = _make_observer(
        modifiers=["hypothalamus"],
        plugin_components={
            "hypothalamus": [("modifier", "hypothalamus")],
        },
        registered={("modifier", "hypothalamus")},
    )
    report = obs.loaded_report()
    assert "modifiers" in report
    assert len(report["modifiers"]) == 1
    entry = report["modifiers"][0]
    assert entry["name"] == "hypothalamus"
    assert entry["kind"] == "modifier"
    assert entry["project"] == "hypothalamus"
    assert entry["loaded"] is True


def test_loaded_report_modifier_project_uses_plugin_folder():
    """Same project-name lookup as tools/channels: a plugin folder
    whose modifier ``.name`` differs from the folder name still
    reports the folder as ``project``."""
    obs = _make_observer(
        modifiers=["recall_anchor"],
        plugin_components={
            "recall": [("modifier", "recall_anchor")],
        },
        registered={("modifier", "recall_anchor")},
    )
    report = obs.loaded_report()
    by_name = {e["name"]: e for e in report["modifiers"]}
    assert by_name["recall_anchor"]["project"] == "recall"


def test_loaded_report_modifier_loaded_flag_reflects_registry_state():
    """Modifier collected by collect_infos but somehow not in the
    registry → loaded=False. (Defensive: the two should always
    match in practice, but the flag must come from the registry.)"""
    # Build observer with registry containing nothing but a
    # plugin_components entry claiming hypothalamus was loaded.
    obs = _make_observer(
        modifiers=[],  # registry empty
        plugin_components={"hypothalamus": [("modifier", "hypothalamus")]},
    )
    # collect_infos walks ``modifiers.all()`` — empty registry → no
    # info object — so the report's modifiers bucket is empty too.
    report = obs.loaded_report()
    assert report["modifiers"] == []
