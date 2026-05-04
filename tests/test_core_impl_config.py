"""CoreImplementations config section parsing + round-trip."""
from __future__ import annotations

import textwrap

from krakey.models.config import Config, dump_config, load_config
from krakey.models.config.core_impls import (
    CoreImplementations,
    _build_core_implementations,
)


def _write(tmp_path, body):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_default_config_has_empty_core_implementations():
    cfg = Config()
    assert isinstance(cfg.core_implementations, CoreImplementations)
    assert cfg.core_implementations.prompt_builder == ""
    assert cfg.core_implementations.embedder == ""
    assert cfg.core_implementations.memory == ""


def test_get_returns_empty_for_unknown_slot():
    """CoreImplementations.get() handles slots not declared as fields."""
    impls = CoreImplementations()
    assert impls.get("definitely_not_a_real_slot") == ""


def test_load_config_parses_core_implementations(tmp_path):
    p = _write(tmp_path, """
        core_implementations:
          prompt_builder: "my_pkg.prompts:CustomBuilder"
          embedder: "my_pkg.embed:CustomEmbedder"
    """)
    cfg = load_config(p)
    assert cfg.core_implementations.prompt_builder == \
        "my_pkg.prompts:CustomBuilder"
    assert cfg.core_implementations.embedder == \
        "my_pkg.embed:CustomEmbedder"
    # Unset slots stay empty
    assert cfg.core_implementations.memory == ""


def test_load_config_missing_section_uses_defaults(tmp_path):
    """Config without the section gets a CoreImplementations() with all empty."""
    p = _write(tmp_path, """
        # No core_implementations section at all.
        idle: {min_interval: 1, max_interval: 60, default_interval: 1}
    """)
    cfg = load_config(p)
    assert isinstance(cfg.core_implementations, CoreImplementations)
    assert cfg.core_implementations.prompt_builder == ""


def test_unknown_keys_are_silently_dropped():
    """A typo in core_implementations doesn't crash; the resolver will
    later see '' for the slot and use the default. (This is a deliberate
    permissive choice — a strict mode could be added later.)"""
    impls = _build_core_implementations({
        "prompt_builder": "x:Y",
        "made_up_slot_xyz": "z:W",   # typo or future slot — ignored
    })
    assert impls.prompt_builder == "x:Y"
    assert not hasattr(impls, "made_up_slot_xyz")


def test_dump_round_trip_preserves_overrides(tmp_path):
    """dump_config(cfg) -> load_config(...) preserves core_implementations."""
    cfg = Config(core_implementations=CoreImplementations(
        prompt_builder="my:PB",
        embedder="my:E",
    ))
    path = tmp_path / "rt.yaml"
    path.write_text(dump_config(cfg), encoding="utf-8")
    loaded = load_config(path)
    assert loaded.core_implementations.prompt_builder == "my:PB"
    assert loaded.core_implementations.embedder == "my:E"


def test_non_dict_input_returns_defaults():
    """A scalar value where a dict was expected → fall back to defaults
    rather than crash. (Loud crash would mask other config errors;
    individual slot validation lives in the resolver.)"""
    assert _build_core_implementations("garbage").prompt_builder == ""
    assert _build_core_implementations(None).prompt_builder == ""
    assert _build_core_implementations(42).prompt_builder == ""
