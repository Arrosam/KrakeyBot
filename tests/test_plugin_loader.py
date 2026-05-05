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


# =====================================================================
# Packaging — pyproject.toml's [tool.setuptools.package-data] table
# must list every plugin's meta.yaml so wheel installs find them
# =====================================================================
#
# Why a dedicated test: BUILTIN_ROOT walks the filesystem directly,
# so ``test_every_builtin_plugin_meta_parses`` passes in a checkout
# regardless of whether ``meta.yaml`` was actually declared in
# ``pyproject.toml``'s package-data table. A pip-installed wheel
# copy WOULDN'T see the meta.yaml unless it's listed there. Routing
# the access through ``importlib.resources`` exercises the same
# resolution a wheel install uses — catches a packaging omission
# that the filesystem-walk tests miss.


_EXPECTED_PLUGINS = (
    # When a new plugin lands, add its name here AND add its
    # meta.yaml entry to pyproject.toml's package-data table.
    # This list catches drift in either direction.
    "browser_exec",
    "cli_exec",
    "dashboard",
    "duckduckgo_search",
    "gui_exec",
    "hypothalamus",
    "in_mind_note",
    "recall",
    "telegram",
)


@pytest.mark.parametrize("plugin_name", _EXPECTED_PLUGINS)
def test_plugin_meta_yaml_reachable_via_importlib_resources(plugin_name):
    """``meta.yaml`` is reachable as a package resource for every
    shipped plugin. Equivalent to what a wheel-installed copy sees,
    so a missing entry in ``pyproject.toml``'s package-data table
    fails this test post-install."""
    from importlib.resources import files

    pkg = f"krakey.plugins.{plugin_name}"
    meta = files(pkg) / "meta.yaml"
    assert meta.is_file(), (
        f"{pkg}/meta.yaml not packaged. Add to pyproject.toml's "
        f"[tool.setuptools.package-data] table."
    )
    text = meta.read_text(encoding="utf-8")
    assert text and "name:" in text


def test_browser_exec_server_source_is_reachable_via_inspect():
    """``snippets.py`` reads ``server.py``'s source via
    ``inspect.getsource`` at import time and embeds it in every
    dispatched snippet. The browser_exec plugin therefore breaks
    entirely if ``server.py`` doesn't survive the wheel build.
    Pin that contract here."""
    import inspect

    from krakey.plugins.browser_exec import server

    src = inspect.getsource(server)
    assert src, "server.py source not reachable via inspect"
    # Recognizable markers — protects against partial / truncated
    # inclusions that ``is_file()`` alone wouldn't catch.
    assert "def serve(" in src
    assert "class BrowserWorker" in src
    assert "/rpc" in src
