"""Smoke tests for the `krakey` CLI: parser wiring, --version, _meta helpers."""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import pytest

from krakey.cli import _meta, main


def test_version_matches_metadata() -> None:
    ver = _meta.version()
    assert ver and ver != "0.0.0+uninstalled", (
        "package must be installed (`pip install -e .`) for tests to run"
    )

    out = io.StringIO()
    with pytest.raises(SystemExit) as exc:
        with redirect_stdout(out):
            main(["--version"])
    assert exc.value.code == 0
    assert ver in out.getvalue()


def test_no_args_prints_help() -> None:
    out = io.StringIO()
    rc = main([])
    # main() returns int when --help-equivalent is invoked via no-args.
    # argparse prints to stdout for print_help; we just check no crash + 0 rc.
    assert rc == 0


def test_unknown_subcommand_exits_nonzero() -> None:
    err = io.StringIO()
    with pytest.raises(SystemExit) as exc:
        with redirect_stderr(err):
            main(["definitely-not-a-command"])
    assert exc.value.code != 0


def test_repo_root_resolves_to_existing_dir() -> None:
    repo = _meta.repo_root()
    assert isinstance(repo, Path)
    assert repo.is_dir()
    assert (repo / "pyproject.toml").is_file()
    assert (repo / "krakey" / "cli" / "__init__.py").is_file()


def test_status_runs_and_returns_zero() -> None:
    # status is side-effect-light: reads pidfile, prints, returns.
    # When no daemon is running it should return 0 (informational).
    rc = main(["status"])
    assert rc == 0


def test_update_falls_back_when_non_editable(monkeypatch, capsys) -> None:
    from krakey.cli import release

    def fake_repo_root() -> Path:
        raise RuntimeError("krakey was installed non-editably; reinstall …")

    monkeypatch.setattr(release._meta, "repo_root", fake_repo_root)
    rc = release.update()
    assert rc == 2
    out = capsys.readouterr().out
    assert "pip install -U krakey" in out


def test_repair_falls_back_when_non_editable(monkeypatch, capsys) -> None:
    from krakey.cli import release

    def fake_repo_root() -> Path:
        raise RuntimeError("krakey was installed non-editably; reinstall …")

    monkeypatch.setattr(release._meta, "repo_root", fake_repo_root)
    rc = release.repair()
    assert rc == 2
    out = capsys.readouterr().out
    assert "pip install -U krakey" in out


def test_banner_renders_with_version_and_tagline(capsys) -> None:
    from krakey.cli import _banner

    _banner.print_banner()
    out = capsys.readouterr().out
    # logo, tagline, and version line are all present
    assert "d8b" in out
    assert "ultimate" in out.replace(" ", "").lower() or "u l t i m a t e" in out
    assert _meta.version() in out


def test_no_args_prints_banner(capsys) -> None:
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "d8b" in out             # logo
    assert "usage: krakey" in out   # argparse help follows banner


def test_runtime_banner_suppressed_when_wizard_ran(capsys) -> None:
    """`krakey run` with no config auto-launches onboarding which has
    its own banner; the runtime startup banner must NOT also print, or
    the user sees TWO banners back-to-back."""
    from krakey.cli import lifecycle

    lifecycle._print_runtime_banner_if_needed(wizard_ran=True)
    assert "d8b" not in capsys.readouterr().out


def test_runtime_banner_prints_when_no_wizard(capsys) -> None:
    """Happy path (`krakey run` with config already present): the
    runtime banner is the one and only banner of the invocation."""
    from krakey.cli import lifecycle

    lifecycle._print_runtime_banner_if_needed(wizard_ran=False)
    assert "d8b" in capsys.readouterr().out
