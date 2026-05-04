"""ServiceResolver — slot override mechanism for replaceable core services.

Covers the resolver in isolation (synthetic Protocols + fixtures); the
end-to-end "user replaces embedder, runtime uses it" test lives in
``test_core_impl_swap_e2e.py`` so this file stays a focused unit test.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pytest

from krakey.runtime.service_resolver import (
    ServiceResolver,
    _default_importer,
    _missing_protocol_attrs,
)


# ---- synthetic fixtures ---------------------------------------------


@runtime_checkable
class _Greeter(Protocol):
    def greet(self) -> str: ...


class _DefaultGreeter:
    def greet(self) -> str:
        return "default"


class _OverrideGreeter:
    def greet(self) -> str:
        return "override"


class _BadGreeter:
    """Missing greet() — fails the Protocol check."""
    def shrug(self) -> str:
        return "huh"


# ---- happy paths ----------------------------------------------------


def test_no_override_returns_default():
    """Empty slot → default_factory(**kwargs) is called."""
    r = ServiceResolver.with_overrides({})
    instance = r.resolve(
        "embedder",
        default_factory=_DefaultGreeter,
        expected_protocol=_Greeter,
    )
    assert instance.greet() == "default"


def test_with_override_returns_user_impl():
    """Non-empty slot + valid Protocol → user instance returned."""
    r = ServiceResolver.with_overrides(
        {"embedder": "fake_pkg:OverrideGreeter"},
        importer=lambda path: _OverrideGreeter,
    )
    instance = r.resolve(
        "embedder",
        default_factory=_DefaultGreeter,
        expected_protocol=_Greeter,
    )
    assert instance.greet() == "override"


def test_kwargs_forwarded_to_default():
    """kwargs supplied to resolve() reach default_factory(**kwargs)."""
    captured: dict = {}

    def factory(*, label: str):
        captured["label"] = label
        return _DefaultGreeter()

    r = ServiceResolver.with_overrides({})
    r.resolve(
        "embedder",
        default_factory=factory,
        expected_protocol=_Greeter,
        label="hello",
    )
    assert captured["label"] == "hello"


def test_kwargs_forwarded_to_override():
    """kwargs supplied to resolve() reach the user class's __init__."""
    captured: dict = {}

    class _Capturing:
        def __init__(self, *, label: str):
            captured["label"] = label

        def greet(self) -> str:
            return "ok"

    r = ServiceResolver.with_overrides(
        {"embedder": "fake:_Capturing"},
        importer=lambda path: _Capturing,
    )
    r.resolve(
        "embedder",
        default_factory=_DefaultGreeter,
        expected_protocol=_Greeter,
        label="hi",
    )
    assert captured["label"] == "hi"


# ---- error paths ----------------------------------------------------


def test_override_failing_protocol_raises_typeerror_listing_missing():
    """Bad override → TypeError that NAMES the missing methods."""
    r = ServiceResolver.with_overrides(
        {"embedder": "fake:BadGreeter"},
        importer=lambda path: _BadGreeter,
    )
    with pytest.raises(TypeError, match="does not satisfy") as excinfo:
        r.resolve(
            "embedder",
            default_factory=_DefaultGreeter,
            expected_protocol=_Greeter,
        )
    assert "greet" in str(excinfo.value)


def test_override_kwarg_mismatch_raises_typeerror_with_hint():
    """User class doesn't accept the slot's kwargs → annotated TypeError."""
    class _PickyCls:
        def __init__(self, *, unrelated_kwarg: str):
            ...
        def greet(self): return "ok"

    r = ServiceResolver.with_overrides(
        {"embedder": "fake:Picky"},
        importer=lambda path: _PickyCls,
    )
    with pytest.raises(TypeError, match="kwargs"):
        r.resolve(
            "embedder",
            default_factory=_DefaultGreeter,
            expected_protocol=_Greeter,
            label="hi",
        )


def test_malformed_path_raises_valueerror():
    """Path missing ':ClassName' suffix is rejected up-front."""
    with pytest.raises(ValueError, match="entry-point style"):
        _default_importer("just_a_module_no_colon")


def test_unimportable_module_raises_importerror():
    with pytest.raises(ImportError, match="cannot import"):
        _default_importer("definitely_not_a_real_module_zxqq:Foo")


def test_missing_attr_raises_importerror():
    """Module exists but doesn't expose the named attribute."""
    with pytest.raises(ImportError, match="no attribute"):
        _default_importer(
            "krakey.runtime.service_resolver:DoesNotExist"
        )


def test_missing_protocol_attrs_helper_lists_methods():
    """The helper enumerates Protocol methods absent on the instance."""
    bad = _BadGreeter()
    missing = _missing_protocol_attrs(bad, _Greeter)
    assert "greet" in missing


# ---- config integration --------------------------------------------


def test_resolver_built_from_real_config():
    """ServiceResolver(cfg) should accept a default Config without error."""
    from krakey.models.config import Config
    cfg = Config()
    r = ServiceResolver(cfg)
    # And resolve with no override should return the default.
    out = r.resolve(
        "embedder",
        default_factory=_DefaultGreeter,
        expected_protocol=_Greeter,
    )
    assert out.greet() == "default"


def test_resolver_unknown_slot_falls_through_to_default():
    """A slot name CoreImplementations doesn't have → empty → default."""
    r = ServiceResolver.with_overrides({})
    out = r.resolve(
        "totally_made_up_slot",
        default_factory=_DefaultGreeter,
        expected_protocol=_Greeter,
    )
    assert out.greet() == "default"
