"""EngineRegistry resolution mechanism — single-slot resolve path.

Tests the mechanism in isolation using fake Protocol + fake impl
classes. The full ``resolve_all`` orchestration that wires every
Engine in dependency order is built incrementally as steps 3-11 land
each impl; that integration test arrives in step 12.

Failure modes covered (all loud — DIP says fail-fast at startup beats
failing 30 minutes into a session with a confusing AttributeError):

  * malformed dotted path (no ``:``) → ValueError
  * import fails → ImportError annotated with the offending path
  * attribute missing on the imported module → ImportError
  * instantiation TypeError on kwargs mismatch → TypeError
  * resulting object doesn't satisfy the Protocol → TypeError listing
    missing attributes
  * neither override nor default supplied → ValueError
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pytest

from krakey.engines.registry import EngineRegistry, _default_importer
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
    """Doesn't accept any kwargs; passing one raises TypeError."""

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
    if override_for == "memory" and override_path:
        cfg = Config(core_implementations=CoreImplementations(memory=override_path))
    elif override_for == "decision" and override_path:
        cfg = Config(core_implementations=CoreImplementations(decision=override_path))
    elif not override_path:
        cfg = Config(core_implementations=CoreImplementations())
    else:
        # generic fallback for ad-hoc slots — set the field by name
        cfg = Config(core_implementations=CoreImplementations(**{override_for: override_path}))
    return EngineRegistry(cfg, importer=importer)


# --------------------------------------------------------------------
# resolve() — happy paths
# --------------------------------------------------------------------


def test_resolve_uses_default_when_no_override():
    """No override set → registry uses default_path."""
    def fake_importer(path: str):
        assert path == "default.module:GoodImpl"
        return _GoodImpl

    reg = _registry_with(importer=fake_importer)
    instance = reg.resolve(
        "memory",
        default_path="default.module:GoodImpl",
        expected_protocol=_DummyProto,
        greeting="from-default",
    )
    assert instance.hello() == "from-default"


def test_resolve_uses_override_over_default():
    """User override beats the registry's built-in default."""
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
        "memory",
        default_path="default.module:Default",
        expected_protocol=_DummyProto,
        greeting="from-user",
    )
    assert seen == ["user.module:Custom"]
    assert instance.hello() == "from-user"


def test_resolve_supports_new_engine_slots():
    """The 6 slot fields added for the engine refactor (decision,
    recall, heartbeat, dispatch, context, explicit_history) are
    addressable by name."""
    def fake_importer(path: str):
        return _GoodImpl

    cfg = Config(core_implementations=CoreImplementations(
        decision="x:y", recall="x:y", heartbeat="x:y",
        dispatch="x:y", context="x:y", explicit_history="x:y",
    ))
    reg = EngineRegistry(cfg, importer=fake_importer)
    for slot in ("decision", "recall", "heartbeat", "dispatch",
                 "context", "explicit_history"):
        instance = reg.resolve(
            slot, default_path="", expected_protocol=_DummyProto,
        )
        assert instance.hello() == "hi"


# --------------------------------------------------------------------
# resolve() — failure modes
# --------------------------------------------------------------------


def test_resolve_raises_when_neither_override_nor_default():
    """Empty override + empty default_path → ValueError."""
    reg = _registry_with()
    with pytest.raises(ValueError, match="no impl path"):
        reg.resolve(
            "memory", default_path="",
            expected_protocol=_DummyProto,
        )


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
    with pytest.raises(TypeError, match="does not satisfy"):
        reg.resolve(
            "memory", default_path="",
            expected_protocol=_DummyProto,
        )


def test_resolve_lists_missing_protocol_attrs_in_error():
    """Error message names the specific attributes the impl lacks so
    users can fix their custom class."""
    def fake_importer(path: str):
        return _BadImpl

    reg = _registry_with(
        override_for="memory",
        override_path="bad:Impl",
        importer=fake_importer,
    )
    with pytest.raises(TypeError) as exc_info:
        reg.resolve(
            "memory", default_path="",
            expected_protocol=_DummyProto,
        )
    assert "hello" in str(exc_info.value)


def test_resolve_raises_on_kwargs_mismatch():
    """Impl __init__ rejects supplied kwargs → TypeError naming the
    expected kwargs the slot's contract requires."""
    def fake_importer(path: str):
        return _NoKwargsImpl

    reg = _registry_with(
        override_for="memory",
        override_path="x:NoKwargs",
        importer=fake_importer,
    )
    with pytest.raises(TypeError, match="could not be instantiated"):
        reg.resolve(
            "memory", default_path="",
            expected_protocol=_DummyProto,
            greeting="x",  # _NoKwargsImpl rejects this
        )


# --------------------------------------------------------------------
# _default_importer — dotted-path → class resolution
# --------------------------------------------------------------------


def test_default_importer_resolves_real_module():
    """Sanity: importer pulls a real class out of stdlib via
    'module:Class' syntax."""
    import collections

    cls = _default_importer("collections:OrderedDict")
    assert cls is collections.OrderedDict


def test_default_importer_raises_on_missing_separator():
    """Path without ``:`` is malformed (we use entry-point-style
    dotted paths to disambiguate package modules from class attrs)."""
    with pytest.raises(ValueError, match="entry-point style"):
        _default_importer("collections.OrderedDict")


def test_default_importer_raises_on_unknown_module():
    """Module the user named doesn't exist → ImportError annotated
    with the original path so the operator sees what to fix."""
    with pytest.raises(ImportError, match="cannot import"):
        _default_importer("krakey_nonexistent_xyz_pkg:Foo")


def test_default_importer_raises_on_missing_attr():
    """Module exists but doesn't expose the named class → ImportError
    naming the missing attribute."""
    with pytest.raises(ImportError, match="has no attribute"):
        _default_importer("collections:DefinitelyNotAClass")
