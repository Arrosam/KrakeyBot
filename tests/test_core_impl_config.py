"""CoreImplementations config section parsing + round-trip.

Step 14 (Engine refactor 2026-05) retired four legacy field names:
prompt_builder → context, sliding_window → explicit_history,
kb_registry + sleep_manager merged into memory. These tests now
exercise the canonical names."""
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
    assert cfg.core_implementations.context == ""
    assert cfg.core_implementations.embedder == ""
    assert cfg.core_implementations.memory == ""


def test_get_returns_empty_for_unknown_slot():
    """CoreImplementations.get() handles slots not declared as fields."""
    impls = CoreImplementations()
    assert impls.get("definitely_not_a_real_slot") == ""


def test_load_config_parses_core_implementations(tmp_path):
    p = _write(tmp_path, """
        core_implementations:
          context: "my_pkg.prompts:CustomBuilder"
          embedder: "my_pkg.embed:CustomEmbedder"
    """)
    cfg = load_config(p)
    assert cfg.core_implementations.context == \
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
    assert cfg.core_implementations.context == ""


def test_unknown_keys_are_silently_dropped():
    """A typo in core_implementations doesn't crash; the resolver will
    later see '' for the slot and use the default. (Permissive
    choice — a strict mode could be added later.)

    The four retired legacy keys (prompt_builder / sliding_window /
    kb_registry / sleep_manager) take this path: silently dropped
    by the loader, override silently lost. Documented migration
    path: rename to the new key (no other change needed)."""
    impls = _build_core_implementations({
        "context": "x:Y",
        "made_up_slot_xyz": "z:W",         # typo / future slot
        "prompt_builder": "legacy:LB",     # retired
        "sliding_window": "legacy:LSW",    # retired
        "kb_registry": "legacy:LKB",       # retired
        "sleep_manager": "legacy:LSM",     # retired
    })
    assert impls.context == "x:Y"
    assert not hasattr(impls, "made_up_slot_xyz")
    # Legacy fields are not on the dataclass any more
    assert not hasattr(impls, "prompt_builder")
    assert not hasattr(impls, "sliding_window")
    assert not hasattr(impls, "kb_registry")
    assert not hasattr(impls, "sleep_manager")


def test_dump_round_trip_preserves_overrides(tmp_path):
    """dump_config(cfg) -> load_config(...) preserves core_implementations."""
    cfg = Config(core_implementations=CoreImplementations(
        context="my:PB",
        embedder="my:E",
    ))
    path = tmp_path / "rt.yaml"
    path.write_text(dump_config(cfg), encoding="utf-8")
    loaded = load_config(path)
    assert loaded.core_implementations.context == "my:PB"
    assert loaded.core_implementations.embedder == "my:E"


def test_non_dict_input_returns_defaults():
    """A scalar value where a dict was expected → fall back to defaults
    rather than crash. (Loud crash would mask other config errors;
    individual slot validation lives in the resolver.)"""
    assert _build_core_implementations("garbage").context == ""
    assert _build_core_implementations(None).context == ""
    assert _build_core_implementations(42).context == ""
