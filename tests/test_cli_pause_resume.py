"""Edge tests for pause/resume CLI capability.

Covers:
  A) Argument parsing — `-p`/`--pause` on `run`/`start`, new `pause`
     subcommand with optional `seconds`, new `resume` subcommand.
  B) `lifecycle.pause_daemon(seconds)` — liveness check, file creation,
     content format, parent-dir creation.
  C) `lifecycle.resume_daemon()` — liveness check, file deletion,
     idempotent absence.
  D) `lifecycle.status()` — running, running+paused, stopped.

Design choices:
  - We never spawn a real process; daemon liveness is faked via
    monkeypatching `lifecycle._read_pid` and `lifecycle._is_alive`,
    exactly as the established restart tests do.
  - `lifecycle._paths()` is monkeypatched to return tmp_path-based
    paths, so the control file and pidfile never land in the real
    workspace.
  - The `_PAUSE_CONTROL_FILE` path is derived by building it from the
    monkeypatched repo root (tmp_path), matching the spec's convention
    `<repo>/workspace/.krakey.pause`.
"""
from __future__ import annotations

import io
import shutil
import time
from contextlib import redirect_stderr
from pathlib import Path

import pytest

from krakey.cli import main
from krakey.cli import lifecycle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PAUSE_REL = "workspace/.krakey.pause"
_FAKE_PID = 54321
_FAKE_VERSION = "0.1.0"


def _fake_paths(tmp_path: Path):
    """Return a _paths()-compatible tuple rooted at tmp_path."""
    repo = tmp_path
    pidfile = repo / "workspace" / ".krakey.pid"
    logfile = repo / "workspace" / "logs" / "daemon.log"
    return repo, pidfile, logfile


def _pause_file(tmp_path: Path) -> Path:
    """The control file path that pause_daemon should create."""
    return tmp_path / _PAUSE_REL


def _setup_alive(monkeypatch, tmp_path: Path):
    """Monkeypatch lifecycle so the daemon appears alive at _FAKE_PID."""
    repo, pidfile, logfile = _fake_paths(tmp_path)
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text(str(_FAKE_PID), encoding="utf-8")

    monkeypatch.setattr(lifecycle, "_paths",
                        lambda: (repo, pidfile, logfile))
    monkeypatch.setattr(lifecycle, "_read_pid",
                        lambda _p: _FAKE_PID)
    monkeypatch.setattr(lifecycle, "_is_alive",
                        lambda _pid: True)


def _setup_dead(monkeypatch, tmp_path: Path):
    """Monkeypatch lifecycle so no daemon is running (no pidfile)."""
    repo, pidfile, logfile = _fake_paths(tmp_path)

    monkeypatch.setattr(lifecycle, "_paths",
                        lambda: (repo, pidfile, logfile))
    monkeypatch.setattr(lifecycle, "_read_pid",
                        lambda _p: None)
    monkeypatch.setattr(lifecycle, "_is_alive",
                        lambda _pid: False)


# ===========================================================================
# A) Argument parsing
# ===========================================================================

class TestArgParsingRunPauseFlag:
    """Positive: `-p` / `--pause` on the `run` subcommand."""

    def test_run_short_pause_flag_sets_start_paused_true(self):
        from krakey.cli import _build_parser  # type: ignore[attr-defined]
        p = _build_parser()
        ns = p.parse_args(["run", "-p"])
        assert getattr(ns, "start_paused", False) is True

    def test_run_long_pause_flag_sets_start_paused_true(self):
        from krakey.cli import _build_parser  # type: ignore[attr-defined]
        p = _build_parser()
        ns = p.parse_args(["run", "--pause"])
        assert getattr(ns, "start_paused", False) is True

    def test_run_without_pause_flag_start_paused_is_false(self):
        from krakey.cli import _build_parser  # type: ignore[attr-defined]
        p = _build_parser()
        ns = p.parse_args(["run"])
        assert getattr(ns, "start_paused", False) is False


class TestArgParsingStartPauseFlag:
    """`start` subcommand also accepts `-p` / `--pause`."""

    def test_start_short_pause_flag_sets_start_paused_true(self):
        from krakey.cli import _build_parser  # type: ignore[attr-defined]
        p = _build_parser()
        ns = p.parse_args(["start", "-p"])
        assert getattr(ns, "start_paused", False) is True

    def test_start_long_pause_flag_sets_start_paused_true(self):
        from krakey.cli import _build_parser  # type: ignore[attr-defined]
        p = _build_parser()
        ns = p.parse_args(["start", "--pause"])
        assert getattr(ns, "start_paused", False) is True

    def test_start_without_pause_flag_start_paused_is_false(self):
        from krakey.cli import _build_parser  # type: ignore[attr-defined]
        p = _build_parser()
        ns = p.parse_args(["start"])
        assert getattr(ns, "start_paused", False) is False


class TestArgParsingPauseSubcommand:
    """New `pause` subcommand with optional positional `seconds`."""

    def test_pause_with_seconds_parses_integer(self):
        from krakey.cli import _build_parser  # type: ignore[attr-defined]
        p = _build_parser()
        ns = p.parse_args(["pause", "120"])
        assert ns.cmd == "pause"
        assert ns.seconds == 120

    def test_pause_with_seconds_zero_parses_as_zero(self):
        from krakey.cli import _build_parser  # type: ignore[attr-defined]
        p = _build_parser()
        ns = p.parse_args(["pause", "0"])
        assert ns.seconds == 0

    def test_pause_with_large_seconds_value(self):
        from krakey.cli import _build_parser  # type: ignore[attr-defined]
        p = _build_parser()
        ns = p.parse_args(["pause", "86400"])
        assert ns.seconds == 86400

    def test_pause_without_seconds_seconds_is_none(self):
        from krakey.cli import _build_parser  # type: ignore[attr-defined]
        p = _build_parser()
        ns = p.parse_args(["pause"])
        assert ns.cmd == "pause"
        assert ns.seconds is None

    def test_pause_seconds_type_is_int_not_string(self):
        from krakey.cli import _build_parser  # type: ignore[attr-defined]
        p = _build_parser()
        ns = p.parse_args(["pause", "60"])
        assert isinstance(ns.seconds, int)


class TestArgParsingResumeSubcommand:
    """New `resume` subcommand — no extra args."""

    def test_resume_parses_with_no_extra_args(self):
        from krakey.cli import _build_parser  # type: ignore[attr-defined]
        p = _build_parser()
        ns = p.parse_args(["resume"])
        assert ns.cmd == "resume"

    def test_resume_cmd_attribute_present(self):
        from krakey.cli import _build_parser  # type: ignore[attr-defined]
        p = _build_parser()
        ns = p.parse_args(["resume"])
        assert hasattr(ns, "cmd")

    def test_pause_and_resume_appear_in_help(self, capsys):
        rc = main([])
        out = capsys.readouterr().out
        assert "pause" in out
        assert "resume" in out


# ===========================================================================
# B) pause_daemon()
# ===========================================================================

class TestPauseDaemonNotAlive:
    """When no daemon is running: stderr message, non-zero return, no file."""

    def test_returns_nonzero_when_no_pidfile(self, monkeypatch, tmp_path, capsys):
        _setup_dead(monkeypatch, tmp_path)
        rc = lifecycle.pause_daemon(None)
        assert rc != 0

    def test_prints_to_stderr_when_not_alive(self, monkeypatch, tmp_path, capsys):
        _setup_dead(monkeypatch, tmp_path)
        lifecycle.pause_daemon(None)
        err = capsys.readouterr().err
        assert err.strip() != ""

    def test_does_not_create_control_file_when_not_alive(
        self, monkeypatch, tmp_path, capsys
    ):
        _setup_dead(monkeypatch, tmp_path)
        lifecycle.pause_daemon(None)
        assert not _pause_file(tmp_path).exists()

    def test_does_not_create_control_file_timed_when_not_alive(
        self, monkeypatch, tmp_path, capsys
    ):
        _setup_dead(monkeypatch, tmp_path)
        lifecycle.pause_daemon(60)
        assert not _pause_file(tmp_path).exists()


class TestPauseDaemonAliveIndefinite:
    """Daemon alive + `seconds=None`: empty control file, returns 0."""

    def test_returns_zero(self, monkeypatch, tmp_path):
        _setup_alive(monkeypatch, tmp_path)
        rc = lifecycle.pause_daemon(None)
        assert rc == 0

    def test_creates_control_file(self, monkeypatch, tmp_path):
        _setup_alive(monkeypatch, tmp_path)
        lifecycle.pause_daemon(None)
        assert _pause_file(tmp_path).exists()

    def test_control_file_is_empty(self, monkeypatch, tmp_path):
        _setup_alive(monkeypatch, tmp_path)
        lifecycle.pause_daemon(None)
        content = _pause_file(tmp_path).read_text(encoding="utf-8")
        assert content == ""

    def test_idempotent_second_call_still_empty(self, monkeypatch, tmp_path):
        _setup_alive(monkeypatch, tmp_path)
        lifecycle.pause_daemon(None)
        lifecycle.pause_daemon(None)
        content = _pause_file(tmp_path).read_text(encoding="utf-8")
        assert content == ""


class TestPauseDaemonAliveTimed:
    """Daemon alive + `seconds=<int>`: deadline content, returns 0."""

    def test_returns_zero(self, monkeypatch, tmp_path):
        _setup_alive(monkeypatch, tmp_path)
        rc = lifecycle.pause_daemon(60)
        assert rc == 0

    def test_creates_control_file(self, monkeypatch, tmp_path):
        _setup_alive(monkeypatch, tmp_path)
        lifecycle.pause_daemon(60)
        assert _pause_file(tmp_path).exists()

    def test_content_is_parseable_float(self, monkeypatch, tmp_path):
        _setup_alive(monkeypatch, tmp_path)
        lifecycle.pause_daemon(60)
        content = _pause_file(tmp_path).read_text(encoding="utf-8").strip()
        # Must parse without exception
        deadline = float(content)
        assert isinstance(deadline, float)

    def test_deadline_approximately_time_plus_seconds(self, monkeypatch, tmp_path):
        _setup_alive(monkeypatch, tmp_path)
        before = time.time()
        lifecycle.pause_daemon(60)
        after = time.time()
        content = _pause_file(tmp_path).read_text(encoding="utf-8").strip()
        deadline = float(content)
        # Deadline should be within a 5-second tolerance of before+60
        assert before + 60 <= deadline <= after + 60 + 5

    def test_deadline_not_in_the_past(self, monkeypatch, tmp_path):
        _setup_alive(monkeypatch, tmp_path)
        lifecycle.pause_daemon(30)
        content = _pause_file(tmp_path).read_text(encoding="utf-8").strip()
        deadline = float(content)
        assert deadline > time.time() - 1  # generous 1s slack for slow machines

    def test_large_seconds_value_stored_correctly(self, monkeypatch, tmp_path):
        _setup_alive(monkeypatch, tmp_path)
        before = time.time()
        lifecycle.pause_daemon(3600)
        content = _pause_file(tmp_path).read_text(encoding="utf-8").strip()
        deadline = float(content)
        assert deadline >= before + 3600 - 1

    def test_zero_seconds_stores_approx_now(self, monkeypatch, tmp_path):
        _setup_alive(monkeypatch, tmp_path)
        before = time.time()
        lifecycle.pause_daemon(0)
        after = time.time()
        content = _pause_file(tmp_path).read_text(encoding="utf-8").strip()
        deadline = float(content)
        # Deadline should be approximately now (within 5s window)
        assert before - 1 <= deadline <= after + 5

    def test_timed_overwrite_replaces_indefinite(self, monkeypatch, tmp_path):
        """Starting with an indefinite pause, switching to timed must
        produce a non-empty file with a valid deadline."""
        _setup_alive(monkeypatch, tmp_path)
        lifecycle.pause_daemon(None)  # indefinite first
        lifecycle.pause_daemon(120)   # then timed
        content = _pause_file(tmp_path).read_text(encoding="utf-8").strip()
        assert content != ""
        float(content)  # must still parse


class TestPauseDaemonParentDirCreation:
    """Parent dirs of the control file must be created if missing."""

    def test_creates_parent_dirs_when_missing(self, monkeypatch, tmp_path):
        _setup_alive(monkeypatch, tmp_path)
        # Remove the workspace dir _setup_alive created so the control
        # file's parent genuinely does not pre-exist.
        shutil.rmtree(tmp_path / "workspace")
        assert not (tmp_path / "workspace").exists()
        lifecycle.pause_daemon(None)
        assert _pause_file(tmp_path).exists()

    def test_works_when_parent_already_exists(self, monkeypatch, tmp_path):
        _setup_alive(monkeypatch, tmp_path)
        # Pre-create the workspace dir (already created by _setup_alive)
        (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
        rc = lifecycle.pause_daemon(None)
        assert rc == 0
        assert _pause_file(tmp_path).exists()


# ===========================================================================
# C) resume_daemon()
# ===========================================================================

class TestResumeDaemonNotAlive:
    """When no daemon is running: stderr message + non-zero return."""

    def test_returns_nonzero_when_not_alive(self, monkeypatch, tmp_path, capsys):
        _setup_dead(monkeypatch, tmp_path)
        rc = lifecycle.resume_daemon()
        assert rc != 0

    def test_prints_to_stderr_when_not_alive(self, monkeypatch, tmp_path, capsys):
        _setup_dead(monkeypatch, tmp_path)
        lifecycle.resume_daemon()
        err = capsys.readouterr().err
        assert err.strip() != ""

    def test_does_not_raise_if_control_file_present(
        self, monkeypatch, tmp_path, capsys
    ):
        _setup_dead(monkeypatch, tmp_path)
        # Pre-plant a control file; resume should not raise even for dead daemon
        cf = _pause_file(tmp_path)
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text("", encoding="utf-8")
        # Must not raise
        rc = lifecycle.resume_daemon()
        assert rc != 0  # still non-zero: daemon not alive


class TestResumeDaemonAlive:
    """Daemon alive: control file deleted, returns 0."""

    def test_returns_zero_when_alive_and_file_present(
        self, monkeypatch, tmp_path
    ):
        _setup_alive(monkeypatch, tmp_path)
        cf = _pause_file(tmp_path)
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text("", encoding="utf-8")
        rc = lifecycle.resume_daemon()
        assert rc == 0

    def test_deletes_control_file_when_alive(self, monkeypatch, tmp_path):
        _setup_alive(monkeypatch, tmp_path)
        cf = _pause_file(tmp_path)
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text("", encoding="utf-8")
        lifecycle.resume_daemon()
        assert not cf.exists()

    def test_deletes_timed_control_file_when_alive(self, monkeypatch, tmp_path):
        _setup_alive(monkeypatch, tmp_path)
        cf = _pause_file(tmp_path)
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text(str(time.time() + 300), encoding="utf-8")
        lifecycle.resume_daemon()
        assert not cf.exists()


class TestResumeDaemonIdempotent:
    """Calling resume when the control file is already absent must not raise
    and must still return 0."""

    def test_returns_zero_when_alive_and_no_file(self, monkeypatch, tmp_path):
        _setup_alive(monkeypatch, tmp_path)
        assert not _pause_file(tmp_path).exists()
        rc = lifecycle.resume_daemon()
        assert rc == 0

    def test_no_exception_when_alive_and_no_file(self, monkeypatch, tmp_path):
        _setup_alive(monkeypatch, tmp_path)
        # Must not raise FileNotFoundError or anything else
        lifecycle.resume_daemon()

    def test_double_resume_second_call_returns_zero(self, monkeypatch, tmp_path):
        _setup_alive(monkeypatch, tmp_path)
        cf = _pause_file(tmp_path)
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text("", encoding="utf-8")
        lifecycle.resume_daemon()
        # Second call: file is already gone
        rc = lifecycle.resume_daemon()
        assert rc == 0


# ===========================================================================
# D) status() output
# ===========================================================================

class TestStatusOutput:
    """status() output wording under different daemon states.

    Existing running output (from lifecycle.py line 488):
        f"krakey: running  pid={pid}  version={ver}{extra}"
    The paused variant must contain "(paused)" as an additional marker
    while keeping the "running" keyword.
    Stopped output:
        f"krakey: stopped  (version {ver})"  or the stale-pid variant.
    """

    def test_status_stopped_contains_stopped(self, monkeypatch, tmp_path, capsys):
        _setup_dead(monkeypatch, tmp_path)
        monkeypatch.setattr(lifecycle._meta, "version", lambda: _FAKE_VERSION)
        lifecycle.status()
        out = capsys.readouterr().out
        assert "stopped" in out

    def test_status_stopped_returns_zero(self, monkeypatch, tmp_path):
        _setup_dead(monkeypatch, tmp_path)
        monkeypatch.setattr(lifecycle._meta, "version", lambda: _FAKE_VERSION)
        rc = lifecycle.status()
        assert rc == 0

    def test_status_running_contains_running(self, monkeypatch, tmp_path, capsys):
        _setup_alive(monkeypatch, tmp_path)
        monkeypatch.setattr(lifecycle._meta, "version", lambda: _FAKE_VERSION)
        # Suppress psutil uptime lookup failures gracefully
        lifecycle.status()
        out = capsys.readouterr().out
        assert "running" in out

    def test_status_running_no_control_file_no_paused_marker(
        self, monkeypatch, tmp_path, capsys
    ):
        _setup_alive(monkeypatch, tmp_path)
        monkeypatch.setattr(lifecycle._meta, "version", lambda: _FAKE_VERSION)
        # No control file
        assert not _pause_file(tmp_path).exists()
        lifecycle.status()
        out = capsys.readouterr().out
        assert "running" in out
        assert "(paused)" not in out

    def test_status_running_with_control_file_shows_paused(
        self, monkeypatch, tmp_path, capsys
    ):
        _setup_alive(monkeypatch, tmp_path)
        monkeypatch.setattr(lifecycle._meta, "version", lambda: _FAKE_VERSION)
        cf = _pause_file(tmp_path)
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text("", encoding="utf-8")
        lifecycle.status()
        out = capsys.readouterr().out
        assert "running" in out
        assert "(paused)" in out

    def test_status_running_paused_still_contains_pid(
        self, monkeypatch, tmp_path, capsys
    ):
        _setup_alive(monkeypatch, tmp_path)
        monkeypatch.setattr(lifecycle._meta, "version", lambda: _FAKE_VERSION)
        cf = _pause_file(tmp_path)
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text("", encoding="utf-8")
        lifecycle.status()
        out = capsys.readouterr().out
        assert str(_FAKE_PID) in out

    def test_status_running_paused_timed_shows_paused(
        self, monkeypatch, tmp_path, capsys
    ):
        """A timed pause (non-empty control file) is also shown as paused
        — status does not need to interpret the deadline, only the file's
        presence triggers the marker."""
        _setup_alive(monkeypatch, tmp_path)
        monkeypatch.setattr(lifecycle._meta, "version", lambda: _FAKE_VERSION)
        cf = _pause_file(tmp_path)
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text(str(time.time() + 300), encoding="utf-8")
        lifecycle.status()
        out = capsys.readouterr().out
        assert "(paused)" in out

    def test_status_returns_zero_when_running_paused(
        self, monkeypatch, tmp_path
    ):
        _setup_alive(monkeypatch, tmp_path)
        monkeypatch.setattr(lifecycle._meta, "version", lambda: _FAKE_VERSION)
        cf = _pause_file(tmp_path)
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text("", encoding="utf-8")
        rc = lifecycle.status()
        assert rc == 0


# ===========================================================================
# E) CLI dispatch — main() routes pause/resume to lifecycle
# ===========================================================================

class TestMainDispatchPause:
    """`main(["pause"])` and `main(["pause", "60"])` call
    lifecycle.pause_daemon with correct arguments."""

    def test_main_pause_no_seconds_calls_pause_daemon_none(
        self, monkeypatch, capsys
    ):
        calls: list = []
        monkeypatch.setattr(
            lifecycle, "pause_daemon",
            lambda seconds=None: calls.append(seconds) or 0,
        )
        main(["pause"])
        assert calls == [None]

    def test_main_pause_with_seconds_calls_pause_daemon_int(
        self, monkeypatch, capsys
    ):
        calls: list = []
        monkeypatch.setattr(
            lifecycle, "pause_daemon",
            lambda seconds=None: calls.append(seconds) or 0,
        )
        main(["pause", "120"])
        assert calls == [120]

    def test_main_pause_propagates_return_code(self, monkeypatch, capsys):
        monkeypatch.setattr(lifecycle, "pause_daemon", lambda seconds=None: 7)
        rc = main(["pause"])
        assert rc == 7


class TestMainDispatchResume:
    """`main(["resume"])` calls lifecycle.resume_daemon."""

    def test_main_resume_calls_resume_daemon(self, monkeypatch, capsys):
        calls: list = []
        monkeypatch.setattr(
            lifecycle, "resume_daemon",
            lambda: calls.append(1) or 0,
        )
        main(["resume"])
        assert calls == [1]

    def test_main_resume_propagates_return_code(self, monkeypatch, capsys):
        monkeypatch.setattr(lifecycle, "resume_daemon", lambda: 3)
        rc = main(["resume"])
        assert rc == 3


# ===========================================================================
# F) _prepare_pause_file()
# ===========================================================================

def _workspace_dir(tmp_path: Path) -> Path:
    """The workspace directory that the monkeypatched _paths() points to."""
    return tmp_path / "workspace"


class TestPreparePauseFileStartPausedTrue:
    """_prepare_pause_file(True) must ensure the control file exists and is
    empty, regardless of whether it previously existed or was non-empty."""

    def test_creates_empty_control_file_when_none_existed(
        self, monkeypatch, tmp_path
    ):
        """Parent dir missing → created; control file created as empty."""
        _setup_alive(monkeypatch, tmp_path)
        # Remove any workspace dir that _setup_alive may have created so
        # the parent genuinely does not exist.
        import shutil as _shutil
        ws = _workspace_dir(tmp_path)
        if ws.exists():
            _shutil.rmtree(ws)
        assert not ws.exists()

        lifecycle._prepare_pause_file(True)

        pf = _pause_file(tmp_path)
        assert pf.exists(), "control file must be created"
        assert pf.read_text(encoding="utf-8") == "", "control file must be empty"

    def test_parent_dir_is_created_when_missing(self, monkeypatch, tmp_path):
        """The parent directory of the control file must exist after the call."""
        _setup_alive(monkeypatch, tmp_path)
        import shutil as _shutil
        ws = _workspace_dir(tmp_path)
        if ws.exists():
            _shutil.rmtree(ws)

        lifecycle._prepare_pause_file(True)

        assert _pause_file(tmp_path).parent.exists()

    def test_truncates_preexisting_nonempty_control_file(
        self, monkeypatch, tmp_path
    ):
        """A pre-existing control file with content (e.g. a deadline float)
        must be replaced / truncated to empty."""
        _setup_alive(monkeypatch, tmp_path)
        pf = _pause_file(tmp_path)
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text(str(time.time() + 3600), encoding="utf-8")
        assert pf.stat().st_size > 0  # pre-condition: non-empty

        lifecycle._prepare_pause_file(True)

        assert pf.exists()
        assert pf.read_text(encoding="utf-8") == ""

    def test_replaces_preexisting_empty_control_file_remains_empty(
        self, monkeypatch, tmp_path
    ):
        """If the control file already exists and is empty, calling with True
        is idempotent — it stays empty (no exception, no size change)."""
        _setup_alive(monkeypatch, tmp_path)
        pf = _pause_file(tmp_path)
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text("", encoding="utf-8")

        lifecycle._prepare_pause_file(True)

        assert pf.exists()
        assert pf.read_text(encoding="utf-8") == ""

    def test_calling_twice_with_true_is_idempotent(self, monkeypatch, tmp_path):
        """Two consecutive calls with True must not raise and the control
        file must still exist and be empty."""
        _setup_alive(monkeypatch, tmp_path)

        lifecycle._prepare_pause_file(True)
        lifecycle._prepare_pause_file(True)

        pf = _pause_file(tmp_path)
        assert pf.exists()
        assert pf.read_text(encoding="utf-8") == ""


class TestPreparePauseFileStartPausedFalse:
    """_prepare_pause_file(False) must delete a pre-existing (stale) control
    file and be a no-op / not raise when no control file is present."""

    def test_deletes_preexisting_stale_control_file(
        self, monkeypatch, tmp_path
    ):
        """A stale empty control file from a previous run must be removed."""
        _setup_alive(monkeypatch, tmp_path)
        pf = _pause_file(tmp_path)
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text("", encoding="utf-8")
        assert pf.exists()  # pre-condition

        lifecycle._prepare_pause_file(False)

        assert not pf.exists(), "stale control file must be deleted"

    def test_deletes_preexisting_nonempty_stale_control_file(
        self, monkeypatch, tmp_path
    ):
        """A stale timed-pause control file (non-empty) must also be removed."""
        _setup_alive(monkeypatch, tmp_path)
        pf = _pause_file(tmp_path)
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text(str(time.time() + 999), encoding="utf-8")

        lifecycle._prepare_pause_file(False)

        assert not pf.exists()

    def test_noop_when_no_control_file_exists(self, monkeypatch, tmp_path):
        """When no control file is present, calling with False must not raise."""
        _setup_dead(monkeypatch, tmp_path)
        pf = _pause_file(tmp_path)
        assert not pf.exists()  # pre-condition

        # Must not raise (e.g. FileNotFoundError)
        lifecycle._prepare_pause_file(False)

        assert not pf.exists()

    def test_idempotent_double_call_with_false_no_raise(
        self, monkeypatch, tmp_path
    ):
        """Calling twice with False when the file doesn't exist must not
        raise on either call."""
        _setup_dead(monkeypatch, tmp_path)

        lifecycle._prepare_pause_file(False)   # first call
        lifecycle._prepare_pause_file(False)   # second call — idempotent

        assert not _pause_file(tmp_path).exists()

    def test_false_does_not_create_control_file(self, monkeypatch, tmp_path):
        """False must never CREATE the control file; if absent, it stays absent."""
        _setup_alive(monkeypatch, tmp_path)
        assert not _pause_file(tmp_path).exists()

        lifecycle._prepare_pause_file(False)

        assert not _pause_file(tmp_path).exists()


# ===========================================================================
# G) pause_daemon() — atomic write post-conditions
#    Verifies that after pause_daemon() returns 0 (daemon alive):
#      • the control file contains the correct content
#      • no stray .tmp sibling file is left in the workspace dir
# ===========================================================================

def _no_tmp_files_in_workspace(tmp_path: Path) -> bool:
    """Return True iff no file whose name ends with '.tmp' exists directly
    inside the workspace directory (i.e. no leftover atomic-write temp)."""
    ws = _workspace_dir(tmp_path)
    if not ws.exists():
        return True
    return not any(f.name.endswith(".tmp") for f in ws.iterdir() if f.is_file())


class TestPauseDaemonAtomicWriteIndefinite:
    """pause_daemon(None) with daemon alive: empty control file, no .tmp left."""

    def test_control_file_is_empty_after_indefinite_pause(
        self, monkeypatch, tmp_path
    ):
        _setup_alive(monkeypatch, tmp_path)
        rc = lifecycle.pause_daemon(None)
        assert rc == 0
        assert _pause_file(tmp_path).read_text(encoding="utf-8") == ""

    def test_no_tmp_sibling_after_indefinite_pause(
        self, monkeypatch, tmp_path
    ):
        """No file ending in '.tmp' must remain in the workspace dir after
        pause_daemon(None) returns — the atomic tmp file must have been
        renamed away or cleaned up."""
        _setup_alive(monkeypatch, tmp_path)
        lifecycle.pause_daemon(None)
        assert _no_tmp_files_in_workspace(tmp_path), (
            "stray .tmp file found in workspace after pause_daemon(None)"
        )

    def test_no_tmp_sibling_on_second_indefinite_pause(
        self, monkeypatch, tmp_path
    ):
        """Calling pause_daemon(None) twice must leave no .tmp on either call."""
        _setup_alive(monkeypatch, tmp_path)
        lifecycle.pause_daemon(None)
        lifecycle.pause_daemon(None)
        assert _no_tmp_files_in_workspace(tmp_path)


class TestPauseDaemonAtomicWriteTimed:
    """pause_daemon(seconds=N) with daemon alive: float content within tolerance,
    no .tmp left."""

    def test_content_parses_as_float_after_timed_pause(
        self, monkeypatch, tmp_path
    ):
        _setup_alive(monkeypatch, tmp_path)
        rc = lifecycle.pause_daemon(120)
        assert rc == 0
        content = _pause_file(tmp_path).read_text(encoding="utf-8").strip()
        float(content)  # must not raise

    def test_deadline_within_tolerance_after_timed_pause(
        self, monkeypatch, tmp_path
    ):
        """Content must be approximately time.time() + 120 (5s tolerance)."""
        _setup_alive(monkeypatch, tmp_path)
        before = time.time()
        lifecycle.pause_daemon(120)
        after = time.time()
        deadline = float(
            _pause_file(tmp_path).read_text(encoding="utf-8").strip()
        )
        assert before + 120 <= deadline <= after + 120 + 5

    def test_no_tmp_sibling_after_timed_pause(self, monkeypatch, tmp_path):
        """No file ending in '.tmp' must remain in the workspace dir after
        pause_daemon(120) returns."""
        _setup_alive(monkeypatch, tmp_path)
        lifecycle.pause_daemon(120)
        assert _no_tmp_files_in_workspace(tmp_path), (
            "stray .tmp file found in workspace after pause_daemon(120)"
        )

    def test_no_tmp_sibling_after_zero_seconds(self, monkeypatch, tmp_path):
        """Boundary: seconds=0 must also leave no .tmp behind."""
        _setup_alive(monkeypatch, tmp_path)
        lifecycle.pause_daemon(0)
        assert _no_tmp_files_in_workspace(tmp_path)

    def test_no_tmp_sibling_after_large_seconds(self, monkeypatch, tmp_path):
        """Large seconds value (86400) must also leave no .tmp behind."""
        _setup_alive(monkeypatch, tmp_path)
        lifecycle.pause_daemon(86400)
        assert _no_tmp_files_in_workspace(tmp_path)

    def test_no_tmp_sibling_after_overwrite_timed_then_indefinite(
        self, monkeypatch, tmp_path
    ):
        """Chain: timed then indefinite — no .tmp after either call."""
        _setup_alive(monkeypatch, tmp_path)
        lifecycle.pause_daemon(60)
        assert _no_tmp_files_in_workspace(tmp_path), (
            "stray .tmp after first (timed) pause_daemon call"
        )
        lifecycle.pause_daemon(None)
        assert _no_tmp_files_in_workspace(tmp_path), (
            "stray .tmp after second (indefinite) pause_daemon call"
        )
