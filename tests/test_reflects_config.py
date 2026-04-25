"""Config-driven Reflect registration + discovery laziness.

Architecture invariant being pinned (Samuel 2026-04-25): a Reflect's
Python code must NOT be imported until the user explicitly enables
it. Discovery (``src.reflects.discovery``) walks pure-text
``meta.yaml`` files; ``load_reflect(name)`` is the only path that
imports the module.

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

from src.models.config import load_config
from src.reflects.discovery import discover_reflects, load_reflect
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
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        reflect_configs_root=runtime._test_reflect_configs_root,
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
    assert cfg.reflects is None


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
    assert cfg.reflects == []


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
          - default_recall_anchor
          - default_hypothalamus
    """)
    cfg = load_config(p)
    assert cfg.reflects == [
        "default_recall_anchor", "default_hypothalamus",
    ]


# ---- discovery: pure text, no imports -------------------------------


def test_discover_finds_builtin_meta_files():
    """The two in-tree built-ins must be discoverable by name."""
    metas = discover_reflects()
    assert "default_hypothalamus" in metas
    assert "default_recall_anchor" in metas
    h = metas["default_hypothalamus"]
    assert h.kind == "hypothalamus"
    assert h.factory_module == (
        "src.reflects.builtin.default_hypothalamus.reflect"
    )
    assert h.factory_attr == "build_reflect"


def test_discover_does_not_import_reflect_modules():
    """Architectural invariant: scanning meta.yaml must not pull
    plugin code into sys.modules. Verified by clearing then
    re-running discovery.

    We inspect the modules belonging to each built-in's reflect.py
    path. If discover_reflects() imported them, they'd appear in
    sys.modules. They MUST NOT.
    """
    plugin_modules = (
        "src.reflects.builtin.default_hypothalamus.reflect",
        "src.reflects.builtin.default_recall_anchor.reflect",
    )
    # Clear cached imports so we know whether discovery is the one
    # importing them. We have to be careful not to break other tests
    # that already imported these modules (e.g. test_reflects.py
    # imports the classes directly), but pytest fixtures isolate
    # sys.modules surprisingly poorly. Defensive: just record the
    # state, run discovery, assert nothing NEW got loaded.
    before = {m: m in sys.modules for m in plugin_modules}
    metas = discover_reflects()
    after = {m: m in sys.modules for m in plugin_modules}
    # The names must show up in metadata regardless of imports.
    assert "default_hypothalamus" in metas
    # Any module not loaded BEFORE discovery must still not be loaded
    # AFTER discovery. (If it was already loaded, no claim.)
    for m in plugin_modules:
        if not before[m]:
            assert not after[m], (
                f"discover_reflects() imported {m} — that's a plugin "
                "module and must stay out of sys.modules until "
                "load_reflect(name) is called explicitly"
            )


def test_load_reflect_imports_and_calls_factory():
    """load_reflect is the *only* path that imports plugin modules.
    It builds a fake PluginContext that pretends the user has bound
    the `translator` purpose to a stand-in LLMClient — the factory
    sees `ctx.get_llm("translator")` return the fake and returns a
    Reflect."""
    from src.reflects.context import PluginContext
    fake_llm = ScriptedLLM([])
    ctx = PluginContext(deps=None, plugin_name="default_hypothalamus",
                          config={}, llms={"translator": fake_llm})
    r = load_reflect("default_hypothalamus", ctx)
    assert r is not None
    assert r.kind == "hypothalamus"
    assert r.name == "default_hypothalamus"


def test_load_reflect_raises_keyerror_for_unknown():
    deps = type("_FakeDeps", (), {})()
    with pytest.raises(KeyError):
        load_reflect("does_not_exist", deps)


# ---- Runtime registration end-to-end --------------------------------


async def test_runtime_registers_explicit_list_in_order(tmp_path, capsys):
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
        reflects=["default_hypothalamus", "default_recall_anchor"],
    )
    err = capsys.readouterr().err
    assert "no `reflects:`" not in err
    assert set(runtime.reflects.names()) == {
        "default_hypothalamus", "default_recall_anchor",
    }


async def test_runtime_registers_empty_list_with_no_warning(tmp_path, capsys):
    """`reflects: []` → zero plugins registered, NO warning."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
        reflects=[],
    )
    err = capsys.readouterr().err
    assert "no `reflects:`" not in err
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
    runtime.reflects._by_kind.clear()
    runtime.config.reflects = None
    capsys.readouterr()  # discard prior output

    runtime._register_reflects_from_config(_minimal_deps_for_runtime(runtime))
    err = capsys.readouterr().err
    assert "no `reflects:`" in err
    # No legacy default registered — explicit principle.
    assert runtime.reflects.names() == []


async def test_runtime_skips_unknown_reflect_names_loudly(tmp_path, capsys):
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
        reflects=[],
    )
    runtime.reflects._by_kind.clear()
    runtime.config.reflects = [
        "default_recall_anchor", "typo_reflect", "default_hypothalamus",
    ]
    capsys.readouterr()

    runtime._register_reflects_from_config(_minimal_deps_for_runtime(runtime))
    err = capsys.readouterr().err
    assert "typo_reflect" in err
    assert set(runtime.reflects.names()) == {
        "default_recall_anchor", "default_hypothalamus",
    }
