"""Config-driven Reflect registration + discovery laziness.

Architecture invariant being pinned (Samuel 2026-04-25): a Reflect's
Python code must NOT be imported until the user explicitly enables
it. Catalogue scanning lives in
``src.plugins.dashboard.services.plugin_catalogue.list_available_plugins``
(Web UI side); runtime loads by name via
``src.plugin_system.loader.load_plugin_meta``. ``load_component``
is the only path that imports plugin modules — both scanners stay
pure-text.

Three input states for ``config.reflects`` (see Config docstring):
  * None         — field absent: register nothing + stderr nudge.
                   No legacy fallback per "all plugins default off".
  * []           — explicit zero Reflects. Honored silently.
  * [name, ...]  — explicit ordered list; each name resolved via
                   discovery, unknown names skipped with log.
"""
import sys
import textwrap

import pytest

from krakey.plugins.dashboard.services.plugin_catalogue import (
    list_available_plugins as discover_plugins,
)
from krakey.models.config import load_config
from krakey.plugin_system.loader import load_component
from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes


# ---- loader: parsing the YAML field ---------------------------------


def _write(tmp_path, body: str):
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def _minimal_deps_for_runtime(runtime):
    """Re-derive a deps-shaped namespace from a runtime built by the
    helper, so tests can re-invoke ``_register_reflects_from_config``
    with the same plugin-isolation setup the helper provisioned.

    Includes ``config`` because plugin factories now resolve their own
    LLMs via ``ctx.get_llm_for_tag`` → ``resolve_llm_for_tag(deps.config,
    tag, deps.llm_clients_by_tag)``.
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        config=runtime.config,
        plugin_configs_root=runtime._test_reflect_configs_root,
        llm_clients_by_tag=runtime._test_llm_clients_by_tag,
        hypo_llm=ScriptedLLM([]),
        in_mind_state_path=None,
    )


def test_loader_returns_none_when_reflects_key_absent(tmp_path):
    p = _write(tmp_path, """
        llm:
          providers:
            p: {type: "openai_compatible", base_url: "http://x", api_key: "k", models: []}
          tags:
            t1: {provider: "p/claude-sonnet-4-5"}
          core_purposes:
            self_thinking: t1
    """)
    cfg = load_config(p)
    assert cfg.plugins is None


def test_loader_returns_empty_list_when_reflects_is_empty(tmp_path):
    p = _write(tmp_path, """
        llm:
          providers:
            p: {type: "openai_compatible", base_url: "http://x", api_key: "k", models: []}
          tags:
            t1: {provider: "p/claude-sonnet-4-5"}
          core_purposes:
            self_thinking: t1
        reflects: []
    """)
    cfg = load_config(p)
    assert cfg.plugins == []


def test_loader_returns_ordered_list_when_specified(tmp_path):
    p = _write(tmp_path, """
        llm:
          providers:
            p: {type: "openai_compatible", base_url: "http://x", api_key: "k", models: []}
          tags:
            t1: {provider: "p/claude-sonnet-4-5"}
          core_purposes:
            self_thinking: t1
        reflects:
          - recall
          - hypothalamus
    """)
    cfg = load_config(p)
    assert cfg.plugins == [
        "recall", "hypothalamus",
    ]


# ---- discovery: pure text, no imports -------------------------------


def test_discover_finds_builtin_meta_files():
    """The three in-tree built-in plugins (hypothalamus, recall,
    in_mind_note) must each be discoverable as a unified-format
    plugin with at least one component."""
    metas = discover_plugins()
    assert "hypothalamus" in metas
    assert "recall" in metas
    assert "in_mind_note" in metas
    h = metas["hypothalamus"]
    assert len(h.components) >= 1
    refl_comp = next(c for c in h.components if c.kind == "reflect")
    assert refl_comp.role == "hypothalamus"
    assert refl_comp.factory_module == (
        "krakey.plugins.hypothalamus.reflect"
    )
    assert refl_comp.factory_attr == "build_reflect"


def test_discover_does_not_import_plugin_modules():
    """Architectural invariant: scanning meta.yaml must not pull
    plugin code into sys.modules.
    """
    plugin_modules = (
        "krakey.plugins.hypothalamus.reflect",
        "krakey.plugins.recall.reflect",
        "krakey.plugins.recall.tentacle",
    )
    before = {m: m in sys.modules for m in plugin_modules}
    metas = discover_plugins()
    after = {m: m in sys.modules for m in plugin_modules}
    assert "hypothalamus" in metas
    for m in plugin_modules:
        if not before[m]:
            assert not after[m], (
                f"discover_plugins() imported {m} — that's a plugin "
                "module and must stay out of sys.modules until "
                "load_component(component, ctx) is called explicitly"
            )


def test_load_component_imports_and_calls_factory():
    """load_component is the only path that imports plugin modules.
    Plugins now resolve LLMs themselves: the factory reads its own
    ``llm_purposes.translator`` from ``ctx.config`` and calls
    ``ctx.get_llm_for_tag``. We stub the latter on a SimpleNamespace
    so no Runtime is needed."""
    from types import SimpleNamespace

    from krakey.interfaces.plugin_context import PluginContext
    metas = discover_plugins()
    refl_comp = next(c for c in metas["hypothalamus"].components
                     if c.kind == "reflect")
    fake_llm = ScriptedLLM([])
    ctx = PluginContext(
        deps=SimpleNamespace(config=None, llm_clients_by_tag={}),
        plugin_name="hypothalamus",
        config={"llm_purposes": {"translator": "_fake_tag"}},
    )
    # Bypass the runtime resolver — return our scripted LLM whenever
    # the factory asks for the bound tag.
    ctx.get_llm_for_tag = lambda tag: (  # type: ignore[assignment]
        fake_llm if tag == "_fake_tag" else None
    )
    r = load_component(refl_comp, ctx)
    assert r is not None
    assert r.role == "hypothalamus"
    assert r.name == "hypothalamus"


# ---- Runtime registration end-to-end --------------------------------


async def test_runtime_registers_explicit_list_in_order(tmp_path, capsys):
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
        reflects=["hypothalamus", "recall"],
    )
    err = capsys.readouterr().err
    assert "no `plugins:`" not in err
    assert set(runtime.reflects.names()) == {
        "hypothalamus", "recall_anchor",
    }


async def test_runtime_registers_empty_list_with_no_warning(tmp_path, capsys):
    """`reflects: []` → zero plugins registered, NO warning."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
        reflects=[],
    )
    err = capsys.readouterr().err
    assert "no `plugins:`" not in err
    assert runtime.reflects.names() == []


async def test_runtime_warns_when_reflects_field_is_none(tmp_path, capsys):
    """No `reflects:` field → register nothing + stderr nudge.
    No legacy fallback (per all-plugins-default-off principle).
    """
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
        reflects=[],  # helper preset; we'll simulate the None case below
    )
    runtime.reflects._by_role.clear(); runtime.reflects._order.clear()
    runtime.config.plugins = None
    capsys.readouterr()  # discard prior output

    runtime._register_plugins_from_config(_minimal_deps_for_runtime(runtime))
    err = capsys.readouterr().err
    assert "no `plugins:`" in err
    # No legacy default registered — explicit principle.
    assert runtime.reflects.names() == []


async def test_runtime_skips_unknown_reflect_names_loudly(tmp_path, capsys):
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
        reflects=[],
    )
    runtime.reflects._by_role.clear(); runtime.reflects._order.clear()
    runtime.config.plugins = [
        "recall", "typo_reflect", "hypothalamus",
    ]
    capsys.readouterr()

    runtime._register_plugins_from_config(_minimal_deps_for_runtime(runtime))
    err = capsys.readouterr().err
    assert "typo_reflect" in err
    assert set(runtime.reflects.names()) == {
        "recall_anchor", "hypothalamus",
    }
