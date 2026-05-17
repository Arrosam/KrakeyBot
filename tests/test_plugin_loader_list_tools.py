"""Edge tests for the list-of-Tools extension to the runtime PluginLoader.

Spec (new behavior):
  1. Single Tool returned from factory — unchanged regression behavior.
  2. list[Tool] returned — ALL tools individually registered; each
     retrievable by name; all present in names().
  3. unregister_one deregisters ALL tools from a list-return plugin;
     re-register restores them (hot-reload roundtrip).
  4. Empty list [] — registers nothing, does not raise.
  5. Duplicate .name in list — last-registration-wins, no raise.
  6. Strictly-additive invariant — factory crash / None return must not
     block registration of sibling plugins or the core loop.

Tests are written before implementation and are expected to FAIL until
the feature is built. They assert ONLY observable registry state via
``ToolRegistry.get`` / ``names`` / ``__contains__`` and the loader's
public ``register_one`` / ``unregister_one`` entry points (reached
through the runtime or directly on ``runtime._plugin_loader``).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import patch

import pytest

from krakey.interfaces.tool import Tool
from krakey.models.stimulus import Stimulus
from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes


# ---------------------------------------------------------------------------
# Shared fake Tool factories
# ---------------------------------------------------------------------------

class FakeTool(Tool):
    """Minimal concrete Tool with a configurable name."""

    def __init__(self, name: str, description: str = "fake tool"):
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {}

    async def execute(self, intent: str, params: dict[str, Any]) -> Stimulus:
        return Stimulus(
            type="tool_feedback",
            source=f"tool:{self._name}",
            content=f"{self._name} called",
            timestamp=datetime.now(),
        )


def _make_runtime(tmp_path=None):
    """Return a runtime with no pre-loaded plugins so the registry starts
    clean (aside from the built-in SleepTool)."""
    return build_runtime_with_fakes(
        self_llm=ScriptedLLM(),
        modifiers=[],
    )


# ---------------------------------------------------------------------------
# Helpers: patch the plugin-system's load_component + load_plugin_meta so
# that the runtime PluginLoader calls our fake factories without needing
# real meta.yaml files on disk or real importlib paths.
# ---------------------------------------------------------------------------

from krakey.plugin_system.loader import ComponentMetadata, PluginMetadata


def _single_tool_meta(plugin_name: str) -> PluginMetadata:
    """PluginMetadata for a plugin that declares exactly one tool component."""
    return PluginMetadata(
        name=plugin_name,
        description="test plugin with one tool",
        components=[
            ComponentMetadata(
                kind="tool",
                factory_module=f"_fake_{plugin_name}",
                factory_attr="build",
            )
        ],
    )


def _list_tool_meta(plugin_name: str, n_components: int = 1) -> PluginMetadata:
    """PluginMetadata for a plugin whose single factory returns a list.

    Because a list-returning factory is still described by one component
    entry in meta.yaml (the spec says the factory MAY return a list), we
    model it the same way as a single-tool plugin — just one component
    entry whose factory will return a list at call time.
    """
    return PluginMetadata(
        name=plugin_name,
        description="test plugin that returns list[Tool]",
        components=[
            ComponentMetadata(
                kind="tool",
                factory_module=f"_fake_{plugin_name}",
                factory_attr="build",
            )
        ],
    )


def _register_plugin_with_factory(runtime, plugin_name: str, factory_return):
    """Patch load_plugin_meta + load_component so that registering
    ``plugin_name`` calls a factory that returns ``factory_return``
    (either a single Tool or a list[Tool]).

    Uses ``runtime._plugin_loader.register_one`` — the public entry point.
    Returns the sub-report dict from register_one.
    """
    meta = _list_tool_meta(plugin_name)

    def _fake_load_meta(name):
        if name == plugin_name:
            return meta
        return None

    def _fake_load_component(component, ctx):
        return factory_return

    with (
        patch(
            "krakey.plugin_system.loader.load_plugin_meta",
            side_effect=_fake_load_meta,
        ),
        patch(
            "krakey.plugin_system.loader.load_component",
            side_effect=_fake_load_component,
        ),
    ):
        return runtime._plugin_loader.register_one(plugin_name, runtime._deps)


async def _unregister_plugin(runtime, plugin_name: str):
    """Call the async unregister_one entry point directly."""
    return await runtime._plugin_loader.unregister_one(plugin_name)


# ===========================================================================
# Behavior 1 — Single Tool: regression (unchanged)
# ===========================================================================


class TestSingleToolRegression:
    """Single Tool return from a factory — must behave exactly as before."""

    def test_single_tool_appears_in_registry(self):
        """Returning one Tool registers it under its .name."""
        runtime = _make_runtime()
        tool = FakeTool("alpha")
        _register_plugin_with_factory(runtime, "plugin_alpha", tool)
        assert "alpha" in runtime.tools

    def test_single_tool_retrievable_by_get(self):
        """ToolRegistry.get(name) returns the exact object."""
        runtime = _make_runtime()
        tool = FakeTool("alpha")
        _register_plugin_with_factory(runtime, "plugin_alpha", tool)
        retrieved = runtime.tools.get("alpha")
        assert retrieved is tool

    def test_single_tool_present_in_names(self):
        """ToolRegistry.names() lists the registered tool name."""
        runtime = _make_runtime()
        tool = FakeTool("alpha")
        _register_plugin_with_factory(runtime, "plugin_alpha", tool)
        assert "alpha" in runtime.tools.names()

    def test_single_tool_register_one_returns_ok(self):
        """register_one reports success (ok=True) for a single-tool plugin."""
        runtime = _make_runtime()
        result = _register_plugin_with_factory(runtime, "plugin_alpha", FakeTool("alpha"))
        assert result.get("ok") is True

    def test_single_tool_register_one_reports_component(self):
        """register_one's components list includes one entry for the tool."""
        runtime = _make_runtime()
        result = _register_plugin_with_factory(runtime, "plugin_alpha", FakeTool("alpha"))
        components = result.get("components", [])
        tool_comps = [c for c in components if c.get("kind") == "tool"]
        assert len(tool_comps) == 1
        assert tool_comps[0].get("name") == "alpha"


# ===========================================================================
# Behavior 2 — list[Tool]: all tools registered individually
# ===========================================================================


class TestListToolRegistration:
    """list[Tool] from a factory — all tools registered individually."""

    def test_all_tools_in_list_appear_in_registry(self):
        """Every tool in the returned list is present in the registry."""
        runtime = _make_runtime()
        tools = [FakeTool("beta_one"), FakeTool("beta_two"), FakeTool("beta_three")]
        _register_plugin_with_factory(runtime, "plugin_beta", tools)
        for t in tools:
            assert t.name in runtime.tools, (
                f"Expected tool '{t.name}' to be registered after list return"
            )

    def test_each_tool_retrievable_by_get(self):
        """ToolRegistry.get(name) returns the correct object for each list item."""
        runtime = _make_runtime()
        tool_a = FakeTool("list_a")
        tool_b = FakeTool("list_b")
        _register_plugin_with_factory(runtime, "plugin_list", [tool_a, tool_b])
        assert runtime.tools.get("list_a") is tool_a
        assert runtime.tools.get("list_b") is tool_b

    def test_all_tools_appear_in_names(self):
        """ToolRegistry.names() lists all tools from the returned list."""
        runtime = _make_runtime()
        _register_plugin_with_factory(
            runtime, "plugin_list",
            [FakeTool("gamma_one"), FakeTool("gamma_two")],
        )
        names = runtime.tools.names()
        assert "gamma_one" in names
        assert "gamma_two" in names

    def test_two_tools_from_same_plugin_are_independent(self):
        """Two tools from one plugin-list are independently callable via get."""
        runtime = _make_runtime()
        tool_x = FakeTool("independent_x")
        tool_y = FakeTool("independent_y")
        _register_plugin_with_factory(runtime, "plugin_xy", [tool_x, tool_y])
        # Different objects — not an alias of the same instance.
        assert runtime.tools.get("independent_x") is not runtime.tools.get("independent_y")

    def test_register_one_reports_ok_for_list_return(self):
        """register_one must return ok=True when factory returns list[Tool]."""
        runtime = _make_runtime()
        result = _register_plugin_with_factory(
            runtime, "plugin_list", [FakeTool("r1"), FakeTool("r2")],
        )
        assert result.get("ok") is True

    def test_register_one_reports_all_components_for_list_return(self):
        """register_one's components list contains one entry per list Tool."""
        runtime = _make_runtime()
        result = _register_plugin_with_factory(
            runtime, "plugin_list", [FakeTool("c1"), FakeTool("c2"), FakeTool("c3")],
        )
        tool_comps = [
            c for c in result.get("components", []) if c.get("kind") == "tool"
        ]
        comp_names = {c.get("name") for c in tool_comps}
        assert "c1" in comp_names
        assert "c2" in comp_names
        assert "c3" in comp_names

    def test_list_tool_count_matches_registry_growth(self):
        """Registry grows by exactly N entries when factory returns N tools
        (assuming names are distinct and not already registered)."""
        runtime = _make_runtime()
        before = set(runtime.tools.names())
        n = 4
        tools = [FakeTool(f"count_tool_{i}") for i in range(n)]
        _register_plugin_with_factory(runtime, "plugin_count", tools)
        after = set(runtime.tools.names())
        new_names = after - before
        assert len(new_names) == n

    def test_single_element_list_behaves_like_single_tool(self):
        """A list with exactly one Tool must register that one tool — BVA lower bound."""
        runtime = _make_runtime()
        tool = FakeTool("solo_in_list")
        _register_plugin_with_factory(runtime, "plugin_solo", [tool])
        assert "solo_in_list" in runtime.tools
        assert runtime.tools.get("solo_in_list") is tool

    def test_large_list_all_registered(self):
        """A list with many tools (BVA large size) — all must be registered."""
        runtime = _make_runtime()
        n = 20
        tools = [FakeTool(f"bulk_{i}") for i in range(n)]
        _register_plugin_with_factory(runtime, "plugin_bulk", tools)
        for t in tools:
            assert t.name in runtime.tools


# ===========================================================================
# Behavior 3 — State transitions: unregister removes all; re-register
#               restores all (hot-reload roundtrip)
# ===========================================================================


class TestUnregisterAllFromListPlugin:
    """unregister_one must remove ALL tools contributed by a list-return plugin."""

    async def test_unregister_removes_all_list_tools(self):
        """After unregister_one, none of the plugin's list-tools remain."""
        runtime = _make_runtime()
        tool_names = ["deregA", "deregB", "deregC"]
        tools = [FakeTool(n) for n in tool_names]
        _register_plugin_with_factory(runtime, "plugin_deregtest", tools)

        # Pre-condition: all registered.
        for name in tool_names:
            assert name in runtime.tools

        await _unregister_plugin(runtime, "plugin_deregtest")

        for name in tool_names:
            assert name not in runtime.tools, (
                f"Tool '{name}' should have been deregistered with its plugin"
            )

    async def test_unregister_leaves_no_orphaned_tools(self):
        """No tool from the list survives in names() after unregister."""
        runtime = _make_runtime()
        tools = [FakeTool("orphan_a"), FakeTool("orphan_b")]
        _register_plugin_with_factory(runtime, "plugin_orphan", tools)
        await _unregister_plugin(runtime, "plugin_orphan")
        names_after = runtime.tools.names()
        assert "orphan_a" not in names_after
        assert "orphan_b" not in names_after

    async def test_unregister_does_not_remove_other_plugins_tools(self):
        """Unregistering a list-tool plugin leaves other plugins' tools intact."""
        runtime = _make_runtime()
        other_tool = FakeTool("other_plugin_tool")
        _register_plugin_with_factory(runtime, "plugin_other", other_tool)

        list_tools = [FakeTool("victim_a"), FakeTool("victim_b")]
        _register_plugin_with_factory(runtime, "plugin_victim", list_tools)

        await _unregister_plugin(runtime, "plugin_victim")

        # Other plugin's tool must survive.
        assert "other_plugin_tool" in runtime.tools

    async def test_re_register_after_unregister_restores_all_tools(self):
        """Hot-reload round-trip: unregister then re-register restores all tools."""
        runtime = _make_runtime()
        plugin_name = "plugin_reload"
        tool_names = ["rl_x", "rl_y"]
        tools_v1 = [FakeTool(n) for n in tool_names]

        _register_plugin_with_factory(runtime, plugin_name, tools_v1)
        await _unregister_plugin(runtime, plugin_name)

        # Re-register (simulates hot-reload). Use fresh instances.
        tools_v2 = [FakeTool(n) for n in tool_names]
        _register_plugin_with_factory(runtime, plugin_name, tools_v2)

        for name in tool_names:
            assert name in runtime.tools, (
                f"Tool '{name}' should be back after re-register"
            )

    async def test_re_register_tools_are_new_instances(self):
        """After a hot-reload, the registry holds the new instances, not the old ones."""
        runtime = _make_runtime()
        plugin_name = "plugin_swap"
        old_tool = FakeTool("swappable")
        _register_plugin_with_factory(runtime, plugin_name, [old_tool])
        await _unregister_plugin(runtime, plugin_name)

        new_tool = FakeTool("swappable")
        _register_plugin_with_factory(runtime, plugin_name, [new_tool])

        assert runtime.tools.get("swappable") is new_tool
        assert runtime.tools.get("swappable") is not old_tool

    async def test_unregister_reports_all_removed_components(self):
        """unregister_one's sub-report lists every removed tool (not just one)."""
        runtime = _make_runtime()
        tools = [FakeTool("rem_a"), FakeTool("rem_b"), FakeTool("rem_c")]
        _register_plugin_with_factory(runtime, "plugin_rem", tools)
        result = await _unregister_plugin(runtime, "plugin_rem")
        removed_names = {
            c.get("name") for c in result.get("removed", [])
            if c.get("kind") == "tool"
        }
        assert "rem_a" in removed_names
        assert "rem_b" in removed_names
        assert "rem_c" in removed_names


# ===========================================================================
# Behavior 4 — Empty list: no registration, no exception
# ===========================================================================


class TestEmptyListBehavior:
    """Factory returning [] must register nothing and not raise."""

    def test_empty_list_does_not_raise(self):
        """register_one must succeed (not raise) when factory returns []."""
        runtime = _make_runtime()
        # Should not raise.
        _register_plugin_with_factory(runtime, "plugin_empty", [])

    def test_empty_list_adds_no_tools(self):
        """Registry must be unchanged (no new tools) after [] return."""
        runtime = _make_runtime()
        before = set(runtime.tools.names())
        _register_plugin_with_factory(runtime, "plugin_empty", [])
        after = set(runtime.tools.names())
        assert after == before, (
            f"Empty list registered unexpected tools: {after - before}"
        )

    def test_empty_list_returns_ok(self):
        """register_one reports ok=True even for an empty list."""
        runtime = _make_runtime()
        result = _register_plugin_with_factory(runtime, "plugin_empty", [])
        assert result.get("ok") is True

    def test_empty_list_reports_no_components(self):
        """register_one's components list is empty when factory returns []."""
        runtime = _make_runtime()
        result = _register_plugin_with_factory(runtime, "plugin_empty", [])
        tool_comps = [
            c for c in result.get("components", []) if c.get("kind") == "tool"
        ]
        assert len(tool_comps) == 0

    async def test_unregister_empty_plugin_is_harmless(self):
        """Unregistering a plugin that registered nothing must not raise."""
        runtime = _make_runtime()
        _register_plugin_with_factory(runtime, "plugin_empty", [])
        # Should not raise.
        await _unregister_plugin(runtime, "plugin_empty")


# ===========================================================================
# Behavior 5 — Duplicate .name in list: last-registration-wins, no raise
# ===========================================================================


class TestDuplicateNameInList:
    """Two tools in the same list sharing .name — last wins, no exception."""

    def test_duplicate_name_in_list_does_not_raise(self):
        """Registering a list with two tools sharing .name must not raise."""
        runtime = _make_runtime()
        tool_a = FakeTool("dup_name")
        tool_b = FakeTool("dup_name")
        # Must not raise.
        _register_plugin_with_factory(runtime, "plugin_dup", [tool_a, tool_b])

    def test_duplicate_name_in_list_last_wins(self):
        """When two list tools share .name, the last one in the list is registered."""
        runtime = _make_runtime()
        first = FakeTool("contested")
        last = FakeTool("contested")
        _register_plugin_with_factory(runtime, "plugin_dup", [first, last])
        # The registry must hold exactly one entry; it must be the last.
        assert runtime.tools.get("contested") is last

    def test_duplicate_name_in_list_single_entry_in_names(self):
        """Duplicate .name in list must result in exactly one entry in names()."""
        runtime = _make_runtime()
        tools = [FakeTool("once"), FakeTool("once")]
        _register_plugin_with_factory(runtime, "plugin_dup", tools)
        assert runtime.tools.names().count("once") == 1

    def test_duplicate_name_in_list_non_duplicate_still_registered(self):
        """Other tools in the same list (with unique names) are registered correctly."""
        runtime = _make_runtime()
        tools = [
            FakeTool("unique_a"),
            FakeTool("contested_dup"),
            FakeTool("contested_dup"),
            FakeTool("unique_b"),
        ]
        _register_plugin_with_factory(runtime, "plugin_mixed_dup", tools)
        assert "unique_a" in runtime.tools
        assert "unique_b" in runtime.tools
        assert "contested_dup" in runtime.tools


# ===========================================================================
# Behavior 6 — Strictly-additive invariant
# ===========================================================================


class TestStrictlyAdditiveInvariant:
    """Factory crash / None / invalid return must not break other plugins."""

    def test_crashing_factory_lands_in_errors_not_exception(self):
        """A plugin whose factory raises must produce an error entry,
        not propagate the exception to the caller."""
        runtime = _make_runtime()

        def _crashing_factory(component, ctx):
            raise RuntimeError("factory exploded")

        meta = _list_tool_meta("plugin_crash")

        def _fake_load_meta(name):
            if name == "plugin_crash":
                return meta
            return None

        with (
            patch(
                "krakey.plugin_system.loader.load_plugin_meta",
                side_effect=_fake_load_meta,
            ),
            patch(
                "krakey.plugin_system.loader.load_component",
                side_effect=_crashing_factory,
            ),
        ):
            # Must not raise.
            result = runtime._plugin_loader.register_one(
                "plugin_crash", runtime._deps,
            )

        # Either ok=False with an error, or ok=True with partial components.
        # The key assertion: the exception did NOT propagate.
        assert isinstance(result, dict)

    def test_crashing_factory_does_not_prevent_sibling_plugin_registration(self):
        """After a factory crash for plugin A, plugin B can still be registered."""
        runtime = _make_runtime()

        crash_meta = _list_tool_meta("plugin_crash_sib")
        good_tool = FakeTool("good_sib_tool")
        good_meta = _list_tool_meta("plugin_good_sib")

        def _fake_load_meta(name):
            if name == "plugin_crash_sib":
                return crash_meta
            if name == "plugin_good_sib":
                return good_meta
            return None

        def _fake_load_component(component, ctx):
            if component.factory_module == "_fake_plugin_crash_sib":
                raise RuntimeError("crash!")
            return good_tool

        with (
            patch(
                "krakey.plugin_system.loader.load_plugin_meta",
                side_effect=_fake_load_meta,
            ),
            patch(
                "krakey.plugin_system.loader.load_component",
                side_effect=_fake_load_component,
            ),
        ):
            # First: crashing plugin.
            runtime._plugin_loader.register_one("plugin_crash_sib", runtime._deps)
            # Second: should still work.
            runtime._plugin_loader.register_one("plugin_good_sib", runtime._deps)

        assert "good_sib_tool" in runtime.tools

    def test_none_returning_factory_does_not_raise(self):
        """A factory returning None must be handled gracefully — no raise."""
        runtime = _make_runtime()
        meta = _list_tool_meta("plugin_none")

        def _fake_load_meta(name):
            if name == "plugin_none":
                return meta
            return None

        def _fake_load_component(component, ctx):
            return None  # explicitly returns nothing

        with (
            patch(
                "krakey.plugin_system.loader.load_plugin_meta",
                side_effect=_fake_load_meta,
            ),
            patch(
                "krakey.plugin_system.loader.load_component",
                side_effect=_fake_load_component,
            ),
        ):
            result = runtime._plugin_loader.register_one("plugin_none", runtime._deps)

        assert isinstance(result, dict)

    def test_none_returning_factory_registers_no_tools(self):
        """None return from factory leaves the registry unchanged."""
        runtime = _make_runtime()
        before = set(runtime.tools.names())
        meta = _list_tool_meta("plugin_none2")

        def _fake_load_meta(name):
            if name == "plugin_none2":
                return meta
            return None

        def _fake_load_component(component, ctx):
            return None

        with (
            patch(
                "krakey.plugin_system.loader.load_plugin_meta",
                side_effect=_fake_load_meta,
            ),
            patch(
                "krakey.plugin_system.loader.load_component",
                side_effect=_fake_load_component,
            ),
        ):
            runtime._plugin_loader.register_one("plugin_none2", runtime._deps)

        after = set(runtime.tools.names())
        assert after == before

    async def test_hot_reload_with_crashing_list_plugin_reports_error_not_exception(self):
        """hot_reload_plugins must handle a factory crash on a list-tool plugin
        by reporting an error entry — not by raising out of the method."""
        runtime = _make_runtime()

        crash_meta = _list_tool_meta("plugin_crash_hr")

        def _fake_load_meta(name):
            if name == "plugin_crash_hr":
                return crash_meta
            return None

        def _crashing_factory(component, ctx):
            raise ValueError("boom in list factory")

        with (
            patch(
                "krakey.plugin_system.loader.load_plugin_meta",
                side_effect=_fake_load_meta,
            ),
            patch(
                "krakey.plugin_system.loader.load_component",
                side_effect=_crashing_factory,
            ),
        ):
            report = await runtime.hot_reload_plugins(["plugin_crash_hr"])

        error_plugins = [e["plugin"] for e in report.get("errors", [])]
        assert "plugin_crash_hr" in error_plugins

    async def test_hot_reload_with_mixed_good_and_crashing_plugins(self):
        """hot_reload_plugins: a crashing plugin should not prevent the good
        plugin (which returns a list[Tool]) from being registered."""
        runtime = _make_runtime()

        crash_meta = _list_tool_meta("plugin_crash_mix")
        good_tool_a = FakeTool("good_mix_a")
        good_tool_b = FakeTool("good_mix_b")
        good_meta = _list_tool_meta("plugin_good_mix")

        def _fake_load_meta(name):
            if name == "plugin_crash_mix":
                return crash_meta
            if name == "plugin_good_mix":
                return good_meta
            return None

        def _dispatch_factory(component, ctx):
            if component.factory_module == "_fake_plugin_crash_mix":
                raise RuntimeError("crash in mixed test")
            return [good_tool_a, good_tool_b]

        with (
            patch(
                "krakey.plugin_system.loader.load_plugin_meta",
                side_effect=_fake_load_meta,
            ),
            patch(
                "krakey.plugin_system.loader.load_component",
                side_effect=_dispatch_factory,
            ),
        ):
            report = await runtime.hot_reload_plugins(
                ["plugin_crash_mix", "plugin_good_mix"],
            )

        assert "good_mix_a" in runtime.tools
        assert "good_mix_b" in runtime.tools


# ===========================================================================
# Boundary value analysis — sizes / edge shapes
# ===========================================================================


class TestBoundaryValues:
    """BVA across list sizes and mixed registrations."""

    def test_two_element_list_both_registered(self):
        """BVA min+1: two-element list — both tools registered."""
        runtime = _make_runtime()
        _register_plugin_with_factory(
            runtime, "plugin_two", [FakeTool("bva_first"), FakeTool("bva_second")],
        )
        assert "bva_first" in runtime.tools
        assert "bva_second" in runtime.tools

    def test_list_tools_do_not_shadow_previously_registered_single_tool(self):
        """A list-returning plugin with a unique-named set must not overwrite
        a pre-existing tool that happens to share a name with *none* of the list.
        (Regression guard: unrelated tool survives.)"""
        runtime = _make_runtime()
        pre_existing = FakeTool("pre_existing_tool")
        _register_plugin_with_factory(runtime, "plugin_pre", pre_existing)

        _register_plugin_with_factory(
            runtime, "plugin_list_safe",
            [FakeTool("list_safe_a"), FakeTool("list_safe_b")],
        )
        assert "pre_existing_tool" in runtime.tools
        assert runtime.tools.get("pre_existing_tool") is pre_existing

    async def test_multiple_list_plugins_unregistered_independently(self):
        """Two list-tool plugins loaded; unregistering one does not touch
        the other's tools."""
        runtime = _make_runtime()
        _register_plugin_with_factory(
            runtime, "plugin_A", [FakeTool("a_tool_1"), FakeTool("a_tool_2")],
        )
        _register_plugin_with_factory(
            runtime, "plugin_B", [FakeTool("b_tool_1"), FakeTool("b_tool_2")],
        )

        await _unregister_plugin(runtime, "plugin_A")

        # A's tools gone, B's survive.
        assert "a_tool_1" not in runtime.tools
        assert "a_tool_2" not in runtime.tools
        assert "b_tool_1" in runtime.tools
        assert "b_tool_2" in runtime.tools

    def test_mixed_single_and_list_plugins_coexist(self):
        """A single-tool plugin and a list-tool plugin loaded in sequence;
        all tools present in names()."""
        runtime = _make_runtime()
        _register_plugin_with_factory(runtime, "plugin_single_co", FakeTool("single_co"))
        _register_plugin_with_factory(
            runtime, "plugin_list_co",
            [FakeTool("list_co_x"), FakeTool("list_co_y")],
        )
        names = runtime.tools.names()
        assert "single_co" in names
        assert "list_co_x" in names
        assert "list_co_y" in names

    async def test_full_hot_reload_roundtrip_list_tool_plugin(self):
        """hot_reload_plugins end-to-end with a list-tool plugin:
        add → verify → remove → verify gone."""
        runtime = _make_runtime()
        good_tools = [FakeTool("hr_tool_p"), FakeTool("hr_tool_q")]
        good_meta = _list_tool_meta("plugin_hr_list")

        def _fake_load_meta(name):
            if name == "plugin_hr_list":
                return good_meta
            return None

        def _fake_load_component(component, ctx):
            return good_tools

        with (
            patch(
                "krakey.plugin_system.loader.load_plugin_meta",
                side_effect=_fake_load_meta,
            ),
            patch(
                "krakey.plugin_system.loader.load_component",
                side_effect=_fake_load_component,
            ),
        ):
            report_add = await runtime.hot_reload_plugins(["plugin_hr_list"])

        # Both tools registered after hot-add.
        assert "hr_tool_p" in runtime.tools
        assert "hr_tool_q" in runtime.tools
        added_plugins = [a["plugin"] for a in report_add.get("added", [])]
        assert "plugin_hr_list" in added_plugins

        # Now remove it.
        remove_meta = _list_tool_meta("plugin_hr_list")

        def _remove_load_meta(name):
            if name == "plugin_hr_list":
                return remove_meta
            return None

        with patch(
            "krakey.plugin_system.loader.load_plugin_meta",
            side_effect=_remove_load_meta,
        ):
            report_remove = await runtime.hot_reload_plugins([])

        assert "hr_tool_p" not in runtime.tools
        assert "hr_tool_q" not in runtime.tools
        removed_plugins = [r["plugin"] for r in report_remove.get("removed", [])]
        assert "plugin_hr_list" in removed_plugins
