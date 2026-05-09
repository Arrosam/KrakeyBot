"""Config-driven Modifier registration + discovery laziness.

Architecture invariant being pinned (Samuel 2026-04-25): a Modifier's
Python code must NOT be imported until the user explicitly enables
it. Catalogue scanning lives in
``krakey.plugin_system.catalogue.list_available_plugins``
(shared by onboarding + dashboard); runtime loads by name via
``krakey.plugin_system.loader.load_plugin_meta``. ``load_component``
is the only path that imports plugin modules — both scanners stay
pure-text.

Three input states for ``config.modifiers`` (see Config docstring):
  * None         — field absent: register nothing + stderr nudge.
                   No legacy fallback per "all plugins default off".
  * []           — explicit zero Modifiers. Honored silently.
  * [name, ...]  — explicit ordered list; each name resolved via
                   discovery, unknown names skipped with log.
"""
import sys
import textwrap

import pytest

from krakey.plugin_system.catalogue import (
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
    helper, so tests can re-invoke ``_register_modifiers_from_config``
    with the same plugin-isolation setup the helper provisioned.

    Includes ``config`` because plugin factories now resolve their own
    LLMs via ``ctx.get_llm_for_tag`` → ``resolve_llm_for_tag(deps.config,
    tag, deps.llm_clients_by_tag)``.
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        config=runtime.config,
        plugin_configs_root=runtime._test_modifier_configs_root,
        llm_clients_by_tag=runtime._test_llm_clients_by_tag,
        in_mind_state_path=None,
    )


def test_loader_returns_none_when_modifiers_key_absent(tmp_path):
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


def test_loader_returns_empty_list_when_modifiers_is_empty(tmp_path):
    p = _write(tmp_path, """
        llm:
          providers:
            p: {type: "openai_compatible", base_url: "http://x", api_key: "k", models: []}
          tags:
            t1: {provider: "p/claude-sonnet-4-5"}
          core_purposes:
            self_thinking: t1
        modifiers: []
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
        modifiers:
          - recall
          - in_mind_note
    """)
    cfg = load_config(p)
    assert cfg.plugins == [
        "recall", "in_mind_note",
    ]


# ---- discovery: pure text, no imports -------------------------------


def test_discover_finds_builtin_meta_files():
    """The in-tree built-in plugins must each be discoverable as a
    unified-format plugin with at least one component."""
    metas = discover_plugins()
    assert "recall" in metas
    assert "in_mind_note" in metas
    im = metas["in_mind_note"]
    assert len(im.components) >= 1
    mod_comp = next(c for c in im.components if c.kind == "modifier")
    assert mod_comp.factory_module == (
        "krakey.plugins.in_mind_note.modifier"
    )
    assert mod_comp.factory_attr == "build_modifier"


def test_discover_does_not_import_plugin_modules():
    """Architectural invariant: scanning meta.yaml must not pull
    plugin code into sys.modules.
    """
    plugin_modules = (
        "krakey.plugins.in_mind_note.modifier",
        "krakey.plugins.recall.tool",
    )
    before = {m: m in sys.modules for m in plugin_modules}
    metas = discover_plugins()
    after = {m: m in sys.modules for m in plugin_modules}
    assert "in_mind_note" in metas
    for m in plugin_modules:
        if not before[m]:
            assert not after[m], (
                f"discover_plugins() imported {m} — that's a plugin "
                "module and must stay out of sys.modules until "
                "load_component(component, ctx) is called explicitly"
            )


def test_load_component_imports_and_calls_factory(tmp_path):
    """load_component is the only path that imports plugin modules.
    The in_mind_note Modifier's factory writes a state file path
    pulled from deps; we point it at a tmp file so the test doesn't
    touch the production workspace."""
    from types import SimpleNamespace

    from krakey.interfaces.plugin_context import PluginContext
    metas = discover_plugins()
    mod_comp = next(c for c in metas["in_mind_note"].components
                     if c.kind == "modifier")
    state_path = tmp_path / "in_mind.json"
    ctx = PluginContext(
        deps=SimpleNamespace(
            config=None,
            llm_clients_by_tag={},
            in_mind_state_path=str(state_path),
        ),
        plugin_name="in_mind_note",
        config={},
    )
    r = load_component(mod_comp, ctx)
    assert r is not None
    assert r.role == "in_mind"
    assert r.name == "in_mind_note"


# ---- Runtime registration end-to-end --------------------------------


async def test_runtime_registers_explicit_list_in_order(tmp_path, capsys):
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
        modifiers=["in_mind_note"],
    )
    err = capsys.readouterr().err
    assert "no `plugins:`" not in err
    assert set(runtime.modifiers.names()) == {"in_mind_note"}


async def test_runtime_registers_empty_list_with_no_warning(tmp_path, capsys):
    """`modifiers: []` → zero plugins registered, NO warning."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
        modifiers=[],
    )
    err = capsys.readouterr().err
    assert "no `plugins:`" not in err
    assert runtime.modifiers.names() == []


async def test_runtime_warns_when_modifiers_field_is_none(tmp_path, capsys):
    """No `modifiers:` field → register nothing + stderr nudge.
    No legacy fallback (per all-plugins-default-off principle).
    """
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
        modifiers=[],  # helper preset; we'll simulate the None case below
    )
    runtime.modifiers._by_role.clear(); runtime.modifiers._order.clear()
    runtime.config.plugins = None
    capsys.readouterr()  # discard prior output

    runtime._register_plugins_from_config(_minimal_deps_for_runtime(runtime))
    err = capsys.readouterr().err
    assert "no `plugins:`" in err
    # No legacy default registered — explicit principle.
    assert runtime.modifiers.names() == []


async def test_runtime_skips_unknown_modifier_names_loudly(tmp_path, capsys):
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
        modifiers=[],
    )
    runtime.modifiers._by_role.clear(); runtime.modifiers._order.clear()
    runtime.config.plugins = [
        "typo_modifier", "in_mind_note",
    ]
    capsys.readouterr()

    runtime._register_plugins_from_config(_minimal_deps_for_runtime(runtime))
    err = capsys.readouterr().err
    assert "typo_modifier" in err
    assert set(runtime.modifiers.names()) == {"in_mind_note"}
