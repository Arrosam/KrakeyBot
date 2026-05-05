"""Unit tests for the ``krakey install`` CLI command + the
companion startup dep-changed warning.

The install module's surface:

  * collect_plugin_dependencies() — walks BUILTIN_ROOT (always
    points at the in-tree plugins) + WORKSPACE_ROOT (cwd-relative).
  * collect_core_dependencies() — reads pyproject.toml at repo
    root if present.
  * deps_hash(plugin_deps) — stable hash of the sorted union.
  * has_pending_deps() — compares live hash against
    workspace/data/install_state.json.
  * install(args) — discovers, computes union, dispatches pip
    (or skips on --dry-run), writes install_state.json on
    success.

Tests stub ``subprocess.call`` so pytest never actually pip-
installs anything in CI. Tests use ``monkeypatch.chdir(tmp_path)``
to isolate workspace state per test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml

from krakey.cli import install as install_mod


# =====================================================================
# Helpers
# =====================================================================


def _baseline_meta(name: str, deps: list[str] | None = None) -> dict:
    body = {
        "name":        name,
        "description": "test plugin",
        "components": [{
            "kind":           "tool",
            "factory_module": "krakey.plugins.cli_exec.tool",
            "factory_attr":   "build_tool",
        }],
    }
    if deps is not None:
        body["dependencies"] = deps
    return body


def _make_workspace_plugin(
    workspace: Path, name: str, deps: list[str] | None = None,
) -> Path:
    plugin_dir = workspace / "plugins" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "meta.yaml").write_text(
        yaml.safe_dump(_baseline_meta(name, deps)),
        encoding="utf-8",
    )
    return plugin_dir


@pytest.fixture
def isolated_workspace(monkeypatch, tmp_path: Path):
    """chdir to tmp_path so workspace/ + WORKSPACE_ROOT resolve
    relative to a clean directory. Returns the workspace path."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(tmp_path)
    return workspace


# =====================================================================
# collect_plugin_dependencies — walks both roots
# =====================================================================


def test_collect_plugin_dependencies_includes_builtin_plugins(
    isolated_workspace,
):
    """The in-tree plugins (krakey/plugins/*) are always
    discoverable regardless of cwd because BUILTIN_ROOT is
    computed from __file__."""
    deps = install_mod.collect_plugin_dependencies()
    # browser_exec / dashboard / gui_exec / etc. must show up
    # with their declared deps from commit 73423ab.
    assert "browser_exec" in deps
    assert "playwright>=1.40" in deps["browser_exec"]
    assert "cli_exec" in deps
    assert deps["cli_exec"] == []
    assert "dashboard" in deps
    assert "fastapi>=0.115" in deps["dashboard"]


def test_collect_plugin_dependencies_picks_up_workspace_plugin(
    isolated_workspace,
):
    """A user-installed plugin under workspace/plugins/<name> with
    its own meta.yaml is added to the discovered set."""
    _make_workspace_plugin(
        isolated_workspace, "user_thing",
        deps=["my-package>=1.0"],
    )
    deps = install_mod.collect_plugin_dependencies()
    assert "user_thing" in deps
    assert deps["user_thing"] == ["my-package>=1.0"]


def test_workspace_plugin_overrides_builtin_with_same_name(
    isolated_workspace,
):
    """When a workspace plugin shadows a built-in (same name),
    the workspace version's declared deps win — matches
    load_plugin_meta's same-name semantics."""
    _make_workspace_plugin(
        isolated_workspace, "cli_exec",
        deps=["override-pkg>=2.0"],
    )
    deps = install_mod.collect_plugin_dependencies()
    assert deps["cli_exec"] == ["override-pkg>=2.0"]


def test_collect_plugin_dependencies_skips_malformed_meta(
    isolated_workspace, capsys,
):
    """A plugin folder whose meta.yaml fails to parse is logged
    to stderr and skipped — doesn't abort the discovery (so a
    broken plugin doesn't block installing the others)."""
    bad_dir = isolated_workspace / "plugins" / "broken"
    bad_dir.mkdir(parents=True)
    (bad_dir / "meta.yaml").write_text(
        "name:\n  this is: malformed\nbad indent\n",
        encoding="utf-8",
    )
    # Also drop a good plugin alongside it to confirm the
    # rest still discover.
    _make_workspace_plugin(
        isolated_workspace, "good", deps=["pkg>=1"],
    )

    deps = install_mod.collect_plugin_dependencies()
    assert "broken" not in deps
    assert "good" in deps
    err = capsys.readouterr().err
    assert "broken" in err
    assert "skipping" in err


# =====================================================================
# collect_core_dependencies
# =====================================================================


def test_collect_core_dependencies_reads_pyproject_when_in_checkout():
    """The repo's own pyproject.toml has a `dependencies` block;
    in a checkout we read it."""
    deps = install_mod.collect_core_dependencies()
    # The repo's pyproject lists pyyaml + aiohttp at the very
    # least.
    assert any(d.startswith("pyyaml") for d in deps)
    assert any(d.startswith("aiohttp") for d in deps)


# =====================================================================
# deps_hash — stable, sorted-union, version-sensitive
# =====================================================================


def test_deps_hash_is_stable_across_plugin_order():
    h1 = install_mod.deps_hash({
        "a": ["pkg>=1.0"],
        "b": ["other>=2.0"],
    })
    h2 = install_mod.deps_hash({
        "b": ["other>=2.0"],
        "a": ["pkg>=1.0"],
    })
    assert h1 == h2


def test_deps_hash_collapses_duplicates_across_plugins():
    """If two plugins each declare the same dep, the hash is the
    same as if only one declared it — the union (not the multiset)
    is what matters for "what needs installing"."""
    h_single = install_mod.deps_hash({"a": ["pkg>=1.0"]})
    h_dup = install_mod.deps_hash({
        "a": ["pkg>=1.0"], "b": ["pkg>=1.0"],
    })
    assert h_single == h_dup


def test_deps_hash_changes_when_a_version_pin_changes():
    """A new spec string is a new element of the set → new hash
    → next startup warning fires correctly."""
    h_old = install_mod.deps_hash({"a": ["pkg>=1.0"]})
    h_new = install_mod.deps_hash({"a": ["pkg>=2.0"]})
    assert h_old != h_new


# =====================================================================
# has_pending_deps — startup decision
# =====================================================================


def test_has_pending_deps_true_when_no_state_file(isolated_workspace):
    pending, deps = install_mod.has_pending_deps()
    assert pending is True
    assert "browser_exec" in deps  # discovery still ran


def test_has_pending_deps_false_when_state_matches(isolated_workspace):
    deps = install_mod.collect_plugin_dependencies()
    h = install_mod.deps_hash(deps)
    install_mod.write_install_state({
        "deps_hash": h, "installed": [], "installed_at": "x",
    })
    pending, _ = install_mod.has_pending_deps()
    assert pending is False


def test_has_pending_deps_true_when_hash_drifts(isolated_workspace):
    install_mod.write_install_state({
        "deps_hash": "some-old-hash",
        "installed": [], "installed_at": "x",
    })
    pending, _ = install_mod.has_pending_deps()
    assert pending is True


def test_has_pending_deps_true_when_state_file_corrupt(
    isolated_workspace,
):
    install_mod.INSTALL_STATE_PATH.parent.mkdir(parents=True)
    install_mod.INSTALL_STATE_PATH.write_text(
        "{not-json", encoding="utf-8",
    )
    pending, _ = install_mod.has_pending_deps()
    assert pending is True


# =====================================================================
# install() — dry-run vs real, success vs failure
# =====================================================================


def test_install_dry_run_does_not_invoke_pip(
    isolated_workspace, monkeypatch, capsys,
):
    called = []

    def fake_call(cmd, *a, **kw):
        called.append(cmd)
        return 0

    monkeypatch.setattr(install_mod.subprocess, "call", fake_call)
    rc = install_mod.install(
        argparse.Namespace(dry_run=True, upgrade=False),
    )
    assert rc == 0
    assert called == []
    out = capsys.readouterr().out
    assert "--dry-run" in out
    # State file is NOT written in dry-run.
    assert not install_mod.INSTALL_STATE_PATH.exists()


def test_install_invokes_pip_with_union_of_deps(
    isolated_workspace, monkeypatch,
):
    captured: list[list[str]] = []

    def fake_call(cmd, *a, **kw):
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(install_mod.subprocess, "call", fake_call)
    rc = install_mod.install(
        argparse.Namespace(dry_run=False, upgrade=False),
    )
    assert rc == 0
    assert len(captured) == 1
    cmd = captured[0]
    # Must invoke the active interpreter's pip (so we end up in
    # the right venv).
    assert cmd[:4] == [
        install_mod.sys.executable, "-m", "pip", "install",
    ]
    # Carries plugin deps (e.g. playwright from browser_exec).
    assert "playwright>=1.40" in cmd
    # State file written on success.
    state = json.loads(install_mod.INSTALL_STATE_PATH.read_text(
        encoding="utf-8",
    ))
    assert "deps_hash" in state
    assert "installed" in state
    assert "browser_exec" in state["installed"]


def test_install_passes_upgrade_flag_when_requested(
    isolated_workspace, monkeypatch,
):
    captured: list[list[str]] = []
    monkeypatch.setattr(
        install_mod.subprocess, "call",
        lambda cmd, *a, **kw: (captured.append(list(cmd)) or 0),
    )
    install_mod.install(
        argparse.Namespace(dry_run=False, upgrade=True),
    )
    assert "--upgrade" in captured[0]


def test_install_does_not_write_state_on_pip_failure(
    isolated_workspace, monkeypatch, capsys,
):
    """If pip exits non-zero, install_state.json must NOT be
    updated — otherwise the next startup wouldn't warn the
    operator that install is still pending."""
    monkeypatch.setattr(
        install_mod.subprocess, "call",
        lambda cmd, *a, **kw: 1,
    )
    rc = install_mod.install(
        argparse.Namespace(dry_run=False, upgrade=False),
    )
    assert rc == 1
    assert not install_mod.INSTALL_STATE_PATH.exists()
    err = capsys.readouterr().err
    assert "rc=1" in err


# =====================================================================
# Idempotence — back-to-back installs
# =====================================================================


def test_install_then_has_pending_deps_returns_false(
    isolated_workspace, monkeypatch,
):
    """After a successful install, the startup check transitions
    pending=True → pending=False without further intervention."""
    monkeypatch.setattr(
        install_mod.subprocess, "call",
        lambda cmd, *a, **kw: 0,
    )
    pending_before, _ = install_mod.has_pending_deps()
    assert pending_before is True

    install_mod.install(
        argparse.Namespace(dry_run=False, upgrade=False),
    )

    pending_after, _ = install_mod.has_pending_deps()
    assert pending_after is False


def test_adding_a_new_workspace_plugin_after_install_pends_again(
    isolated_workspace, monkeypatch,
):
    """The whole point of the hash check: when the operator
    enables a new plugin (drops it under workspace/plugins/),
    the next startup must warn even though krakey-install was
    run before."""
    monkeypatch.setattr(
        install_mod.subprocess, "call",
        lambda cmd, *a, **kw: 0,
    )
    install_mod.install(
        argparse.Namespace(dry_run=False, upgrade=False),
    )
    assert install_mod.has_pending_deps()[0] is False

    _make_workspace_plugin(
        isolated_workspace, "new_thing",
        deps=["totally-new-pkg>=1"],
    )
    assert install_mod.has_pending_deps()[0] is True
