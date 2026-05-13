"""EngineRegistry resolution mechanism — short-name catalog +
dotted-path fallback paths.

Failure modes covered (all loud — DIP says fail-fast at startup beats
failing 30 minutes into a session with a confusing AttributeError):

  * empty override → falls back to slot's catalog DEFAULT_ENGINE
  * short name not in catalog → ValueError listing the available names
  * malformed dotted path (no ``:``) → ValueError
  * import fails → ImportError annotated with the offending path
  * attribute missing on the imported module → ImportError
  * instantiation TypeError on kwargs mismatch → TypeError
  * resulting object doesn't satisfy the Protocol → TypeError listing
    missing attributes
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pytest

from krakey.engine_system.registry import EngineRegistry, _default_importer
from krakey.models.config import Config
from krakey.models.config.core_impls import CoreImplementations


# --------------------------------------------------------------------
# Test fixtures — fake Protocol + impls that satisfy / violate it
# --------------------------------------------------------------------


@runtime_checkable
class _DummyProto(Protocol):
    def hello(self) -> str: ...


class _GoodImpl:
    def __init__(self, *, greeting: str = "hi"):
        self._greeting = greeting

    def hello(self) -> str:
        return self._greeting


class _BadImpl:
    """Lacks hello(); should fail Protocol validation."""

    def goodbye(self) -> str:
        return "bye"


class _NoKwargsImpl:
    """Doesn't accept any kwargs; passing one would raise TypeError
    pre-filter."""

    def __init__(self):
        pass

    def hello(self) -> str:
        return "hi"


def _registry_with(*, override_for: str = "memory",
                    override_path: str = "",
                    importer=None) -> EngineRegistry:
    """Build an EngineRegistry whose cfg.core_implementations.<slot>
    holds the given override path. Tests stub the importer to skip
    real module loading."""
    cfg = Config(core_implementations=CoreImplementations(
        **{override_for: override_path},
    ))
    return EngineRegistry(cfg, importer=importer)


# --------------------------------------------------------------------
# resolve() — dotted-path fallback (`:` in override)
# --------------------------------------------------------------------


def test_resolve_dotted_path_override_uses_importer():
    """Override containing ``:`` is treated as a dotted path and
    fed to the registry's importer."""
    seen: list[str] = []

    def fake_importer(path: str):
        seen.append(path)
        return _GoodImpl

    reg = _registry_with(
        override_for="memory",
        override_path="user.module:Custom",
        importer=fake_importer,
    )
    instance = reg.resolve(
        "memory", expected_protocol=_DummyProto,
        greeting="from-user",
    )
    assert seen == ["user.module:Custom"]
    assert instance.hello() == "from-user"


def test_resolve_silently_drops_unknown_kwargs():
    """Resolver inspects the class signature and drops kwargs the
    constructor doesn't accept."""
    def fake_importer(path: str):
        return _NoKwargsImpl

    reg = _registry_with(
        override_for="memory",
        override_path="x:NoKwargs",
        importer=fake_importer,
    )
    instance = reg.resolve(
        "memory", expected_protocol=_DummyProto,
        greeting="x",
    )
    assert instance.hello() == "hi"


def test_resolve_raises_when_protocol_violated():
    """Impl class lacks Protocol-required attributes → TypeError
    with a list of the missing attrs."""
    def fake_importer(path: str):
        return _BadImpl

    reg = _registry_with(
        override_for="memory",
        override_path="bad:Impl",
        importer=fake_importer,
    )
    with pytest.raises(TypeError) as exc_info:
        reg.resolve("memory", expected_protocol=_DummyProto)
    assert "does not satisfy" in str(exc_info.value)
    assert "hello" in str(exc_info.value)


# --------------------------------------------------------------------
# resolve() — short-name catalog
# --------------------------------------------------------------------


def test_resolve_uses_slot_default_when_no_override():
    """Empty override → registry resolves the slot's
    ``DEFAULT_ENGINE`` from ``engines/<slot>/BUILTIN_ENGINES``."""
    from krakey.engines.decision import ToolCallParserDecisionEngine
    from krakey.interfaces.engines.decision import DecisionEngine

    reg = _registry_with(override_for="decision", override_path="")
    instance = reg.resolve(
        "decision", expected_protocol=DecisionEngine,
        cfg=None, factory=None,
    )
    # Default for `decision` is the scripted parser.
    assert isinstance(instance, ToolCallParserDecisionEngine)


def test_resolve_short_name_picks_from_catalog():
    """Override = a short name in BUILTIN_ENGINES → registry returns
    that catalog entry's class, not a dotted path."""
    from krakey.engines.decision import HypothalamusDecisionEngine
    from krakey.engines.llm_factory.default import (
        DefaultLLMClientFactoryEngine,
    )
    from krakey.interfaces.engines.decision import DecisionEngine

    cfg_yaml_like = Config(core_implementations=CoreImplementations(
        decision="hypothalamus",
    ))
    reg = EngineRegistry(cfg_yaml_like)
    factory = DefaultLLMClientFactoryEngine(cfg_yaml_like)
    instance = reg.resolve(
        "decision", expected_protocol=DecisionEngine,
        cfg=cfg_yaml_like, factory=factory,
    )
    assert isinstance(instance, HypothalamusDecisionEngine)


def test_resolve_unknown_short_name_raises_with_available_list():
    """Short name not in the catalog → ValueError listing the slot's
    available short names so the user can fix the typo."""
    from krakey.interfaces.engines.decision import DecisionEngine

    cfg = Config(core_implementations=CoreImplementations(
        decision="not_a_real_engine",
    ))
    reg = EngineRegistry(cfg)
    with pytest.raises(ValueError) as exc_info:
        reg.resolve(
            "decision", expected_protocol=DecisionEngine,
            cfg=cfg, factory=None,
        )
    msg = str(exc_info.value)
    assert "not_a_real_engine" in msg
    assert "tool_call_parser" in msg
    assert "hypothalamus" in msg


# --------------------------------------------------------------------
# Plugin-engine catalog
# --------------------------------------------------------------------


def test_resolve_plugin_engine_short_name(monkeypatch):
    """Override = a plugin name → registry consults the plugin-engine
    catalog (parsed from each plugin's meta.yaml) for the dotted path
    and instantiates the same way as a built-in."""
    # Stub list_available_plugins to inject one plugin claiming the
    # `decision` slot — avoids planting a real plugin folder under
    # workspace/plugins/ for the test.
    from krakey.plugin_system import catalogue as cat_mod
    from krakey.plugin_system.loader import (
        ComponentMetadata, PluginMetadata,
    )

    fake_plugin = PluginMetadata(
        name="my_translator",
        description="",
        components=[
            ComponentMetadata(
                kind="engine",
                slot="decision",
                factory_module="my_translator.engine",
                factory_attr="MyEngine",
            ),
        ],
    )
    monkeypatch.setattr(
        cat_mod, "list_available_plugins",
        lambda: {"my_translator": fake_plugin},
    )

    seen: list[str] = []

    def fake_importer(path: str):
        seen.append(path)
        return _GoodImpl  # satisfies _DummyProto

    cfg = Config(core_implementations=CoreImplementations(
        decision="my_translator",
    ))
    reg = EngineRegistry(cfg, importer=fake_importer)
    instance = reg.resolve(
        "decision", expected_protocol=_DummyProto,
        greeting="from-plugin",
    )
    # The plugin's dotted path was looked up via the plugin catalog
    # rather than treated as a literal short name.
    assert seen == ["my_translator.engine:MyEngine"]
    assert instance.hello() == "from-plugin"


def test_resolve_passes_per_engine_config_kwarg(monkeypatch):
    """``cfg.engine_configs.<slot>.<short_name>`` is threaded into the
    resolved engine's constructor as ``config=``. Impls that don't
    declare a ``config`` parameter ignore it via ``_filter_kwargs``
    — pinned here separately."""
    from krakey.engine_system.catalog import EngineImpl
    import krakey.engine_system.registry as reg_mod

    captured: dict = {}

    class _ConfigAwareImpl:
        def __init__(self, *, config=None):
            captured["config"] = config

        def hello(self):
            return "ok"

    cfg = Config(
        core_implementations=CoreImplementations(memory="custom"),
        engine_configs={
            "memory": {
                "custom": {"cache_size_mb": 200},
            },
        },
    )
    monkeypatch.setattr(
        reg_mod, "_load_slot_catalog",
        lambda slot: (
            {"custom": EngineImpl(cls=_ConfigAwareImpl, description="x")},
            "custom",
        ),
    )
    reg = EngineRegistry(cfg)
    instance = reg.resolve("memory", expected_protocol=_DummyProto)
    assert instance.hello() == "ok"
    assert captured["config"] == {"cache_size_mb": 200}


def test_resolve_passes_empty_config_when_user_set_none(monkeypatch):
    """Engine declares config_schema but user hasn't set anything →
    constructor receives an empty dict, not None. Lets impls assume
    ``config`` is always a dict and read fields with .get(...,
    default)."""
    from krakey.engine_system.catalog import EngineImpl
    import krakey.engine_system.registry as reg_mod

    captured: dict = {}

    class _ConfigAwareImpl:
        def __init__(self, *, config=None):
            captured["config"] = config

        def hello(self):
            return "ok"

    cfg = Config(core_implementations=CoreImplementations(memory="custom"))
    monkeypatch.setattr(
        reg_mod, "_load_slot_catalog",
        lambda slot: (
            {"custom": EngineImpl(cls=_ConfigAwareImpl, description="x")},
            "custom",
        ),
    )
    reg = EngineRegistry(cfg)
    reg.resolve("memory", expected_protocol=_DummyProto)
    assert captured["config"] == {}


def test_engine_overlap_hint_recognises_engine_short_names():
    """When a stale ``plugins:`` entry happens to match a built-in
    Engine short-name (the common ``hypothalamus`` carry-over after
    that plugin was retired into an Engine slot), the warning hint
    should call it out so the user knows where to put it instead."""
    from krakey.runtime.plugin_register.loader import (
        _engine_overlap_hint,
    )

    msg = _engine_overlap_hint("hypothalamus")
    assert msg
    assert "core_implementations.decision" in msg
    assert "remove" in msg.lower()
    # Plain unknown name (not in any catalog) → empty string, no
    # spurious hint.
    assert _engine_overlap_hint("totally_made_up_xyz") == ""


def test_engine_components_are_skipped_by_runtime_plugin_loader(tmp_path):
    """``kind: engine`` components live in meta.yaml so the dashboard
    + EngineRegistry can discover them, but the runtime plugin loader
    must NOT try to instantiate them as runtime plugins (they're
    instantiated by the Engine slot resolution path instead). Pinned
    here so a regression in the loader doesn't accidentally re-import
    engine code on plugin enable."""
    from krakey.runtime.plugin_register.loader import PluginLoader

    # Build a meta.yaml with a single ``kind: engine`` component
    # under workspace/plugins/<name>/, then verify register_one
    # doesn't try to load it.
    workspace_root = tmp_path / "workspace" / "plugins"
    plugin_dir = workspace_root / "fake_engine_plugin"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "meta.yaml").write_text(
        "name: fake_engine_plugin\n"
        "description: \"Engine-only plugin (no runtime components)\"\n"
        "components:\n"
        "  - kind: engine\n"
        "    slot: decision\n"
        "    factory_module: nonexistent.module.that.would.error\n"
        "    factory_attr: WouldRaise\n",
        encoding="utf-8",
    )
    # Patch WORKSPACE_ROOT so the loader picks up our temp folder.
    import krakey.plugin_system.loader as ldr
    import importlib
    monkey_orig = ldr.WORKSPACE_ROOT
    ldr.WORKSPACE_ROOT = workspace_root
    try:
        # Sanity: meta parses + the engine component is present.
        meta = ldr.load_plugin_meta("fake_engine_plugin")
        assert meta is not None
        assert meta.components[0].kind == "engine"
        assert meta.components[0].slot == "decision"
        # The runtime plugin loader's register_one must NOT raise on
        # the bogus dotted path because it never imports engine
        # components — proves the skip branch fires.
        from types import SimpleNamespace
        loader = PluginLoader(
            config=SimpleNamespace(plugins=["fake_engine_plugin"]),
            modifiers=SimpleNamespace(register=lambda _i: None),
            tools=SimpleNamespace(register=lambda _i: None),
            channels=SimpleNamespace(register=lambda _i: None),
            services={},
        )
        deps = SimpleNamespace(
            plugin_configs_root=str(tmp_path / "configs"),
            llm_factory=None, config=SimpleNamespace(),
        )
        report = loader.register_one("fake_engine_plugin", deps)
        # No engine ever invoked; loader treats this as "all components
        # skipped, nothing to register".
        assert "WouldRaise" not in str(report.get("error") or "")
    finally:
        ldr.WORKSPACE_ROOT = monkey_orig
        importlib.invalidate_caches()


# --------------------------------------------------------------------
# _default_importer — dotted-path → class resolution
# --------------------------------------------------------------------


def test_default_importer_resolves_real_module():
    import collections

    cls = _default_importer("collections:OrderedDict")
    assert cls is collections.OrderedDict


def test_default_importer_raises_on_missing_separator():
    with pytest.raises(ValueError, match="entry-point style"):
        _default_importer("collections.OrderedDict")


def test_default_importer_raises_on_unknown_module():
    with pytest.raises(ImportError, match="cannot import"):
        _default_importer("krakey_nonexistent_xyz_pkg:Foo")


def test_default_importer_raises_on_missing_attr():
    with pytest.raises(ImportError, match="has no attribute"):
        _default_importer("collections:DefinitelyNotAClass")


# --------------------------------------------------------------------
# Defaults sanity — every FALLBACK_ENGINES entry must be importable
# --------------------------------------------------------------------


def test_fallback_engines_all_resolve():
    """Regression: ``engine_system.defaults.FALLBACK_ENGINES`` is the
    recovery path used when ``meta.yaml`` is missing or malformed. If
    any entry points at a non-existent class, startup fails exactly
    when fallback is supposed to keep the runtime alive — defeating
    the point. Verify every entry can be imported."""
    from krakey.engine_system.defaults import FALLBACK_ENGINES

    failures: list[str] = []
    for slot, path in FALLBACK_ENGINES.items():
        try:
            cls = _default_importer(path)
            assert cls is not None
        except Exception as e:  # noqa: BLE001
            failures.append(f"{slot} → {path}: {type(e).__name__}: {e}")
    assert not failures, (
        "FALLBACK_ENGINES entries that don't resolve:\n  "
        + "\n  ".join(failures)
    )


# --------------------------------------------------------------------
# Bad meta.yaml factory paths fall back to defaults
# --------------------------------------------------------------------


def test_meta_with_bad_factory_module_falls_back_to_defaults(
    tmp_path, monkeypatch, capsys,
):
    """When a slot's ``meta.yaml`` is yaml-valid but the
    ``factory_module`` points at a non-importable module, the
    registry must degrade to ``FALLBACK_ENGINES[slot]`` rather than
    propagate the ImportError. Codex PR #18 review caught the
    earlier behavior: typo in meta crashed instead of degrading."""
    from types import SimpleNamespace

    from krakey.engine_system import meta_loader

    # Stub load_slot_meta to return a one-entry catalog whose factory
    # path points at a non-existent module. The registry's
    # _resolve_class should hit ImportError on _LazyImpl._resolve()
    # and then fall back via FALLBACK_ENGINES[slot].
    def fake_load_slot_meta(slot, *, engines_root=None):
        from krakey.engine_system.catalog import EngineImpl
        return (
            {"broken": EngineImpl(
                cls=meta_loader._LazyImpl(  # type: ignore[arg-type]
                    "krakey_nonexistent_pkg_xyz",
                    "DoesNotMatter",
                ),
                description="bogus factory path",
            )},
            "broken",
        )
    monkeypatch.setattr(meta_loader, "load_slot_meta", fake_load_slot_meta)
    # Also intercept the registry's import of the same name (the
    # registry imports load_slot_meta inside _load_slot_catalog).
    import krakey.engine_system.registry as reg
    monkeypatch.setattr(
        "krakey.engine_system.meta_loader.load_slot_meta",
        fake_load_slot_meta,
    )

    cfg = SimpleNamespace(
        core_implementations=SimpleNamespace(get=lambda _slot: None),
        engine_configs={},
    )
    registry = EngineRegistry(cfg)  # type: ignore[arg-type]
    # Picking "memory" — its FALLBACK_ENGINES entry points at the real
    # GraphMemoryEngine, which imports cleanly.
    cls = registry._resolve_class("memory", "broken")
    from krakey.engines.memory.default import GraphMemoryEngine
    assert cls is GraphMemoryEngine
    # Loud warning fired so the operator notices.
    err = capsys.readouterr().err
    assert "memory" in err
    assert "falling back" in err
