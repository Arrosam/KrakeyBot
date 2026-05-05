"""Unit tests for ``krakey.plugin_system.loader.parse_meta`` —
focused on the ``dependencies`` field added 2026-05-05.

Pre-existing fields (name / description / components / config_schema)
were already exercised indirectly by the dashboard catalogue tests
and onboarding tests; this file pins the new field's contract:

  * accepts an absent / empty list → ``dependencies == []``
  * accepts a list of pip-installable strings → preserved verbatim
  * rejects non-list values
  * rejects non-string / empty entries (must be valid spec strings)
  * round-trips through every existing built-in plugin's meta.yaml

so future schema additions can't silently drift the field's shape.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from krakey.plugin_system.loader import (
    BUILTIN_ROOT,
    PluginMetadata,
    load_plugin_meta,
    parse_meta,
)


def _write_meta(tmp_path: Path, body: dict) -> Path:
    p = tmp_path / "meta.yaml"
    p.write_text(yaml.safe_dump(body), encoding="utf-8")
    return p


def _baseline_meta() -> dict:
    """Minimum-viable meta dict — only what's required to parse."""
    return {
        "name":        "test_plugin",
        "description": "for tests",
        "components": [{
            "kind":           "tool",
            "factory_module": "krakey.plugins.cli_exec.tool",
            "factory_attr":   "build_tool",
        }],
    }


# =====================================================================
# Default + happy path
# =====================================================================


def test_parse_meta_default_dependencies_is_empty_list(tmp_path):
    """Field absent in YAML → empty list (not None)."""
    meta = parse_meta(_write_meta(tmp_path, _baseline_meta()))
    assert meta.dependencies == []
    assert isinstance(meta.dependencies, list)


def test_parse_meta_explicit_empty_list(tmp_path):
    body = _baseline_meta()
    body["dependencies"] = []
    meta = parse_meta(_write_meta(tmp_path, body))
    assert meta.dependencies == []


def test_parse_meta_preserves_pip_spec_strings_verbatim(tmp_path):
    """Spec strings round-trip exactly as written — version
    constraints, extras, environment markers all survive."""
    body = _baseline_meta()
    body["dependencies"] = [
        "playwright>=1.40",
        "pyautogui ~= 0.9",
        "uvicorn[standard]>=0.30",
        "aiohttp>=3.9; python_version >= '3.11'",
        "git+https://example.com/foo.git",
    ]
    meta = parse_meta(_write_meta(tmp_path, body))
    assert meta.dependencies == [
        "playwright>=1.40",
        "pyautogui ~= 0.9",
        "uvicorn[standard]>=0.30",
        "aiohttp>=3.9; python_version >= '3.11'",
        "git+https://example.com/foo.git",
    ]


def test_parse_meta_strips_whitespace_around_specs(tmp_path):
    body = _baseline_meta()
    body["dependencies"] = ["  playwright>=1.40  ", "\taiohttp\n"]
    meta = parse_meta(_write_meta(tmp_path, body))
    assert meta.dependencies == ["playwright>=1.40", "aiohttp"]


# =====================================================================
# Reject malformed shapes
# =====================================================================


def test_parse_meta_rejects_non_list_dependencies(tmp_path):
    body = _baseline_meta()
    body["dependencies"] = "playwright"  # string, not list
    with pytest.raises(ValueError) as ei:
        parse_meta(_write_meta(tmp_path, body))
    assert "dependencies" in str(ei.value)
    assert "list" in str(ei.value)


def test_parse_meta_rejects_dict_dependencies(tmp_path):
    body = _baseline_meta()
    body["dependencies"] = {"playwright": "1.40"}
    with pytest.raises(ValueError):
        parse_meta(_write_meta(tmp_path, body))


@pytest.mark.parametrize("bad_entry", [
    None, 0, [], {}, "", "   ", b"playwright",
])
def test_parse_meta_rejects_non_string_or_empty_entries(
    tmp_path, bad_entry,
):
    body = _baseline_meta()
    body["dependencies"] = ["playwright>=1.40", bad_entry]
    with pytest.raises(ValueError) as ei:
        parse_meta(_write_meta(tmp_path, body))
    assert "dependencies[1]" in str(ei.value)


# =====================================================================
# Cross-check: every shipped plugin's meta.yaml parses cleanly
# =====================================================================


def test_every_builtin_plugin_meta_parses():
    """Catches accidental schema drift across all in-tree plugins.
    If a plugin meta.yaml gains a malformed ``dependencies:`` field,
    this test points at the offender."""
    failures: list[tuple[str, Exception]] = []
    for plugin_dir in sorted(BUILTIN_ROOT.iterdir()):
        meta_path = plugin_dir / "meta.yaml"
        if not meta_path.exists():
            continue
        try:
            meta = parse_meta(meta_path)
        except Exception as e:  # noqa: BLE001
            failures.append((plugin_dir.name, e))
            continue
        assert isinstance(meta, PluginMetadata)
        assert isinstance(meta.dependencies, list)
        # Each entry is a non-empty stripped string.
        for d in meta.dependencies:
            assert isinstance(d, str) and d.strip() == d and d
    if failures:
        msg = "\n".join(f"  {name}: {e}" for name, e in failures)
        raise AssertionError(
            f"{len(failures)} plugin meta.yaml(s) failed to parse:\n{msg}"
        )


def test_load_plugin_meta_returns_metadata_for_known_plugin():
    """The high-level load-by-name wrapper threads the
    ``dependencies`` field through correctly."""
    meta = load_plugin_meta("cli_exec")
    assert meta is not None
    assert meta.name == "cli_exec"
    assert isinstance(meta.dependencies, list)
