"""Edge tests for Runtime.request_pause / Runtime.request_resume and the
_pause_io helper module (write_pause_file / clear_pause_file).

Contract under test:
  - krakey/runtime/_pause_io.write_pause_file(path, seconds) -> None
  - krakey/runtime/_pause_io.clear_pause_file(path) -> None
  - Runtime.request_pause(seconds=None) -> bool
  - Runtime.request_resume() -> bool

Tests are written against the BLACK-BOX contract only.  No knowledge of
implementation internals is assumed beyond the observation points stated
in the spec:
  - runtime._pause_file is set/cleared via runtime.set_pause_file(path | None)
  - runtime.paused is a read-only property (bool)
  - runtime.poll_pause_file() exists and is called implicitly by the two
    new methods (its semantics are already verified in test_pause_file.py)
  - the control file on disk is observable

Construction pattern mirrors test_pause_file.py exactly.
"""
from __future__ import annotations

import pathlib
import time

import pytest

from tests._runtime_helpers import (
    NullEmbedder,
    ScriptedLLM,
    build_runtime_with_fakes,
)
from krakey.runtime._pause_io import clear_pause_file, write_pause_file


# ---------------------------------------------------------------------------
# Shared helper — minimal Runtime, no plugins
# ---------------------------------------------------------------------------

def _make_runtime(tmp_path: pathlib.Path):
    """Minimal Runtime with no plugins — sufficient for pause-request tests."""
    return build_runtime_with_fakes(
        self_llm=ScriptedLLM([]),
        hypo_llm=ScriptedLLM([]),
        embedder=NullEmbedder(),
        modifiers=[],
        gm_path=str(tmp_path / "gm.sqlite"),
    )


# ===========================================================================
# SECTION A: Helper unit tests — write_pause_file / clear_pause_file
# (spec section 5: "Helper edge cases")
# ===========================================================================

class TestWritePauseFileHelperPositive:
    """Positive / equivalence tests for write_pause_file."""

    def test_none_seconds_creates_empty_file(self, tmp_path):
        """write_pause_file(path, None) must produce a file whose stripped
        content is an empty string."""
        p = tmp_path / "pause.ctrl"
        write_pause_file(p, None)
        assert p.exists(), "file must be created"
        assert p.read_text(encoding="utf-8").strip() == ""

    def test_positive_seconds_creates_deadline_file(self, tmp_path):
        """write_pause_file(path, 5) must write a float value within ±2 s
        of time.time() + 5."""
        p = tmp_path / "pause.ctrl"
        before = time.time()
        write_pause_file(p, 5)
        after = time.time()
        assert p.exists()
        value = float(p.read_text(encoding="utf-8").strip())
        # Deadline must equal time.time()+5 at call time, bracketed by
        # [before+5, after+5] with a small jitter margin.
        assert (before + 5) - 1.0 <= value <= (after + 5) + 1.0, (
            f"deadline {value} outside expected range "
            f"[{before + 5 - 1.0}, {after + 5 + 1.0}]"
        )

    def test_seconds_5_deadline_in_range(self, tmp_path):
        """float(file.read_text().strip()) - time.time() should be in [4, 6]."""
        p = tmp_path / "pause.ctrl"
        write_pause_file(p, 5)
        parsed = float(p.read_text(encoding="utf-8").strip())
        diff = parsed - time.time()
        assert 4.0 <= diff <= 6.0, f"expected diff in [4,6], got {diff}"

    def test_overwrite_none_after_deadline(self, tmp_path):
        """Calling write_pause_file again on the same path overwrites cleanly.
        Deadline form then None form must leave an empty file."""
        p = tmp_path / "pause.ctrl"
        write_pause_file(p, 30)
        assert p.read_text(encoding="utf-8").strip() != ""
        write_pause_file(p, None)
        assert p.read_text(encoding="utf-8").strip() == ""

    def test_overwrite_deadline_after_none(self, tmp_path):
        """None form then deadline form must leave a parseable float."""
        p = tmp_path / "pause.ctrl"
        write_pause_file(p, None)
        write_pause_file(p, 10)
        val = float(p.read_text(encoding="utf-8").strip())
        assert val > time.time() - 1.0

    def test_creates_parent_directories(self, tmp_path):
        """write_pause_file must create missing parent directories."""
        p = tmp_path / "deep" / "nested" / "pause.ctrl"
        assert not p.parent.exists()
        write_pause_file(p, None)
        assert p.exists()

    def test_returns_none(self, tmp_path):
        """write_pause_file must return None."""
        p = tmp_path / "pause.ctrl"
        result = write_pause_file(p, None)
        assert result is None

    def test_returns_none_with_seconds(self, tmp_path):
        """write_pause_file(path, 5) must return None."""
        p = tmp_path / "pause.ctrl"
        result = write_pause_file(p, 5)
        assert result is None


class TestWritePauseFileBVA:
    """Boundary value tests for write_pause_file."""

    def test_seconds_zero(self, tmp_path):
        """write_pause_file(path, 0) writes a deadline of ~time.time(),
        which is at or just past the expiry boundary."""
        p = tmp_path / "pause.ctrl"
        before = time.time()
        write_pause_file(p, 0)
        after = time.time()
        assert p.exists()
        value = float(p.read_text(encoding="utf-8").strip())
        # deadline must be time.time()+0 — within ±1 s of call window
        assert before - 1.0 <= value <= after + 1.0

    def test_seconds_one(self, tmp_path):
        """write_pause_file(path, 1) writes a future deadline ~1 s from now."""
        p = tmp_path / "pause.ctrl"
        before = time.time()
        write_pause_file(p, 1)
        after = time.time()
        value = float(p.read_text(encoding="utf-8").strip())
        assert (before + 1) - 1.0 <= value <= (after + 1) + 1.0

    def test_large_seconds(self, tmp_path):
        """write_pause_file(path, 86400) must write a deadline roughly 24 h
        in the future without error."""
        p = tmp_path / "pause.ctrl"
        before = time.time()
        write_pause_file(p, 86400)
        value = float(p.read_text(encoding="utf-8").strip())
        assert value > before + 86399

    def test_repeated_writes_same_path(self, tmp_path):
        """Calling write_pause_file five times in succession on the same path
        must leave exactly one file with only the last write's content."""
        p = tmp_path / "pause.ctrl"
        for _ in range(5):
            write_pause_file(p, None)
        content = p.read_text(encoding="utf-8").strip()
        assert content == ""


class TestClearPauseFileHelperPositive:
    """Positive and error-safety tests for clear_pause_file."""

    def test_clear_absent_path_no_exception(self, tmp_path):
        """clear_pause_file on a path that does not exist must not raise."""
        p = tmp_path / "nonexistent.ctrl"
        assert not p.exists()
        clear_pause_file(p)  # must not raise

    def test_clear_existing_file_removes_it(self, tmp_path):
        """clear_pause_file on an existing file must unlink it."""
        p = tmp_path / "pause.ctrl"
        p.write_text("", encoding="utf-8")
        assert p.exists()
        clear_pause_file(p)
        assert not p.exists()

    def test_clear_returns_none(self, tmp_path):
        """clear_pause_file must return None (not True/False/etc.)."""
        p = tmp_path / "pause.ctrl"
        p.write_text("", encoding="utf-8")
        result = clear_pause_file(p)
        assert result is None

    def test_clear_absent_path_returns_none(self, tmp_path):
        """clear_pause_file on an absent path must return None."""
        p = tmp_path / "ghost.ctrl"
        result = clear_pause_file(p)
        assert result is None

    def test_clear_idempotent_double_call(self, tmp_path):
        """Calling clear_pause_file twice on the same path (second call sees
        absent file) must not raise."""
        p = tmp_path / "pause.ctrl"
        p.write_text("", encoding="utf-8")
        clear_pause_file(p)
        clear_pause_file(p)  # must not raise, file already gone

    def test_clear_then_write_then_clear(self, tmp_path):
        """clear → write → clear cycle must leave file absent each time
        clear is called and present after write."""
        p = tmp_path / "pause.ctrl"
        clear_pause_file(p)  # absent → no-op
        assert not p.exists()
        write_pause_file(p, None)
        assert p.exists()
        clear_pause_file(p)
        assert not p.exists()


# ===========================================================================
# SECTION B: Runtime.request_pause — positive / equivalence
# ===========================================================================

class TestRequestPausePositive:
    """Positive equivalence tests for Runtime.request_pause."""

    def test_indefinite_pause_file_created_and_empty(self, tmp_path):
        """request_pause() with no argument must create the pause file with
        empty (stripped) content."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        rt.request_pause()
        assert control.exists(), "pause file must exist after request_pause()"
        assert control.read_text(encoding="utf-8").strip() == ""

    def test_indefinite_pause_sets_paused_true(self, tmp_path):
        """request_pause() must leave runtime.paused True immediately."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        rt.request_pause()
        assert rt.paused is True

    def test_request_pause_returns_true_when_file_set(self, tmp_path):
        """request_pause() must return True when _pause_file is configured."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        result = rt.request_pause()
        assert result is True

    def test_timed_pause_file_exists_and_parseable(self, tmp_path):
        """request_pause(seconds=60) must create a file with a parseable
        float deadline approximately 60 s from now."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        before = time.time()
        rt.request_pause(seconds=60)
        after = time.time()
        assert control.exists()
        value = float(control.read_text(encoding="utf-8").strip())
        assert (before + 60) - 3.0 <= value <= (after + 60) + 3.0, (
            f"deadline {value} not within ±3 s of expected range"
        )

    def test_timed_pause_sets_paused_true(self, tmp_path):
        """request_pause(seconds=60) must set runtime.paused to True
        (deadline is in the future)."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        rt.request_pause(seconds=60)
        assert rt.paused is True

    def test_timed_pause_returns_true(self, tmp_path):
        """request_pause(seconds=60) must return True."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        result = rt.request_pause(seconds=60)
        assert result is True

    def test_indefinite_pause_idempotent(self, tmp_path):
        """Calling request_pause() twice must leave file empty and paused True
        — no accumulation or error."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        rt.request_pause()
        rt.request_pause()
        assert control.exists()
        assert control.read_text(encoding="utf-8").strip() == ""
        assert rt.paused is True


# ===========================================================================
# SECTION C: Runtime.request_resume — positive / equivalence
# ===========================================================================

class TestRequestResumePositive:
    """Positive equivalence tests for Runtime.request_resume."""

    def test_resume_after_pause_file_absent(self, tmp_path):
        """request_resume() after request_pause() must unlink the pause file."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        rt.request_pause()
        assert control.exists()
        rt.request_resume()
        assert not control.exists()

    def test_resume_after_pause_sets_paused_false(self, tmp_path):
        """request_resume() after request_pause() must set runtime.paused False."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        rt.request_pause()
        assert rt.paused is True
        rt.request_resume()
        assert rt.paused is False

    def test_resume_returns_true_when_file_set(self, tmp_path):
        """request_resume() must return True when _pause_file is configured,
        regardless of whether a pause was active."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        result = rt.request_resume()
        assert result is True

    def test_resume_when_not_paused_returns_true(self, tmp_path):
        """request_resume() when no pause is active must return True (file is
        configured) and leave runtime.paused False."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        # Never called request_pause — file absent already
        result = rt.request_resume()
        assert result is True
        assert rt.paused is False

    def test_resume_idempotent_double_call(self, tmp_path):
        """Calling request_resume() twice in succession must not raise; second
        call sees absent file and runtime.paused stays False."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        rt.request_pause()
        rt.request_resume()
        rt.request_resume()  # second call — file already absent
        assert rt.paused is False


# ===========================================================================
# SECTION D: Boundary value analysis
# ===========================================================================

class TestRequestPauseBVA:
    """Boundary value tests for request_pause."""

    def test_seconds_zero_file_written_no_exception(self, tmp_path):
        """request_pause(seconds=0) must not raise. Per the existing
        poll_pause_file semantics, an at-or-past deadline causes the
        post-write poll to set paused=False and unlink the file
        (expired-deadline cleanup), so by the time the call returns the
        file is gone and paused is False. Core invariants: no exception,
        the method returns True (it applied), runtime.paused is a bool."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        applied = rt.request_pause(seconds=0)  # must not raise
        assert applied is True
        # paused may resolve to either outcome at the exact boundary;
        # in practice seconds=0 means the deadline is "now" which is
        # almost certainly already past by the time poll_pause_file
        # parses it — but both are valid.
        assert isinstance(rt.paused, bool)

    def test_seconds_one_pause_active(self, tmp_path):
        """request_pause(seconds=1) must leave runtime.paused True immediately
        (the 1-second deadline is in the future at call time)."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        rt.request_pause(seconds=1)
        assert rt.paused is True

    def test_overwrite_expired_deadline_with_seconds_zero(self, tmp_path):
        """request_pause(seconds=1) then sleep 1.2 s then request_pause(0):
        the second call overwrites the file with an immediately-expired
        deadline, and the subsequent poll_pause_file unlinks the file
        and resolves to paused=False (existing expired-deadline cleanup
        semantics — same as the runtime auto-resuming a stale deadline)."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        rt.request_pause(seconds=1)
        assert rt.paused is True

        time.sleep(1.2)

        applied = rt.request_pause(seconds=0)  # overwrite; deadline is past
        assert applied is True
        # After poll_pause_file on an expired deadline, paused is False
        # and the file has been cleaned up by the expired-deadline path.
        assert rt.paused is False
        assert not control.exists()

    def test_parent_dir_absent_request_pause_succeeds(self, tmp_path):
        """Pause file path inside a nonexistent subdirectory: request_pause()
        must create parent directories and succeed."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "subdir" / "deep" / "pause.ctrl"
        assert not control.parent.exists()
        rt.set_pause_file(control)
        rt.request_pause()  # must not raise
        assert control.exists()

    def test_request_pause_seconds_large(self, tmp_path):
        """request_pause(seconds=86400) must not raise and must write a valid
        future deadline."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        rt.request_pause(seconds=86400)
        assert rt.paused is True
        assert control.exists()
        value = float(control.read_text(encoding="utf-8").strip())
        assert value > time.time() + 86398


# ===========================================================================
# SECTION E: State transitions
# ===========================================================================

class TestPauseResumeStateTransitions:
    """State transition tests for the pause/resume sequence."""

    def test_pause_resume_pause_resume_cycle(self, tmp_path):
        """Full cycle: pause → resume → pause → resume.
        After each pause: paused True, file present.
        After each resume: paused False, file absent.
        No exception at any step."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)

        # First pause
        rt.request_pause()
        assert rt.paused is True
        assert control.exists()

        # First resume
        rt.request_resume()
        assert rt.paused is False
        assert not control.exists()

        # Second pause
        rt.request_pause()
        assert rt.paused is True
        assert control.exists()

        # Second resume
        rt.request_resume()
        assert rt.paused is False
        assert not control.exists()

    def test_indefinite_then_timed_overwrite(self, tmp_path):
        """request_pause() (indefinite) followed by request_pause(seconds=30)
        must overwrite the file with the deadline form, leaving paused True.
        Then request_resume() clears everything."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)

        rt.request_pause()  # indefinite
        assert control.read_text(encoding="utf-8").strip() == ""
        assert rt.paused is True

        rt.request_pause(seconds=30)  # overwrite with timed deadline
        content = control.read_text(encoding="utf-8").strip()
        assert content != "", "file must now contain a deadline float, not be empty"
        float(content)  # must be parseable — raises if not
        assert rt.paused is True

        rt.request_resume()
        assert not control.exists()
        assert rt.paused is False

    def test_timed_then_indefinite_overwrite(self, tmp_path):
        """request_pause(seconds=30) followed by request_pause() (indefinite)
        must overwrite the file with empty content, leaving paused True."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)

        rt.request_pause(seconds=30)
        assert float(control.read_text(encoding="utf-8").strip()) > time.time()

        rt.request_pause()  # overwrite to indefinite
        assert control.read_text(encoding="utf-8").strip() == ""
        assert rt.paused is True

    def test_resume_after_timed_pause_still_paused(self, tmp_path):
        """request_pause(seconds=60) then request_resume():
        file absent, paused False."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        rt.request_pause(seconds=60)
        rt.request_resume()
        assert not control.exists()
        assert rt.paused is False

    def test_pause_returns_true_each_call(self, tmp_path):
        """request_pause() must return True on every call when file is set."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        for _ in range(3):
            result = rt.request_pause()
            assert result is True

    def test_resume_returns_true_each_call(self, tmp_path):
        """request_resume() must return True on every call when file is set,
        even after multiple resume calls with no intervening pause."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        for _ in range(3):
            result = rt.request_resume()
            assert result is True


# ===========================================================================
# SECTION F: Negative — no-file-configured cases
# ===========================================================================

class TestNoPauseFileConfigured:
    """Negative tests: _pause_file is None — both methods must be no-ops
    returning False without touching disk or state."""

    def test_request_pause_returns_false_when_no_file(self, tmp_path):
        """request_pause() with no pause file configured must return False."""
        rt = _make_runtime(tmp_path)
        result = rt.request_pause()
        assert result is False

    def test_request_pause_no_file_io_when_no_file(self, tmp_path):
        """request_pause() with no pause file configured must not create any
        file in the tmp directory."""
        rt = _make_runtime(tmp_path)
        files_before = set(tmp_path.rglob("*.ctrl"))
        rt.request_pause()
        files_after = set(tmp_path.rglob("*.ctrl"))
        assert files_before == files_after, "no .ctrl file should appear after no-op"

    def test_request_pause_no_exception_when_no_file(self, tmp_path):
        """request_pause() with no pause file must not raise."""
        rt = _make_runtime(tmp_path)
        rt.request_pause()  # must not raise

    def test_request_pause_paused_stays_false_when_no_file(self, tmp_path):
        """runtime.paused must remain False after request_pause() with no file."""
        rt = _make_runtime(tmp_path)
        rt.request_pause()
        assert rt.paused is False

    def test_request_resume_returns_false_when_no_file(self, tmp_path):
        """request_resume() with no pause file configured must return False."""
        rt = _make_runtime(tmp_path)
        result = rt.request_resume()
        assert result is False

    def test_request_resume_no_exception_when_no_file(self, tmp_path):
        """request_resume() with no pause file must not raise."""
        rt = _make_runtime(tmp_path)
        rt.request_resume()  # must not raise

    def test_request_resume_paused_stays_false_when_no_file(self, tmp_path):
        """runtime.paused must remain False after request_resume() with no file."""
        rt = _make_runtime(tmp_path)
        rt.request_resume()
        assert rt.paused is False

    def test_request_pause_and_resume_both_false_when_no_file(self, tmp_path):
        """Both calls in sequence with no file: both return False, no raises,
        paused stays False throughout."""
        rt = _make_runtime(tmp_path)
        r1 = rt.request_pause()
        r2 = rt.request_resume()
        assert r1 is False
        assert r2 is False
        assert rt.paused is False

    def test_cleared_file_path_makes_both_return_false(self, tmp_path):
        """After set_pause_file(None) clears the path, both methods must
        return False and leave paused False."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        rt.set_pause_file(None)  # clear the path

        r1 = rt.request_pause()
        r2 = rt.request_resume()
        assert r1 is False
        assert r2 is False
        assert rt.paused is False

    def test_cleared_file_path_no_file_created(self, tmp_path):
        """After set_pause_file(None), request_pause() must not create any
        file — even though a path was previously registered."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        rt.set_pause_file(None)
        rt.request_pause()
        assert not control.exists()

    def test_resume_when_file_configured_but_file_absent_returns_true(self, tmp_path):
        """request_resume() when _pause_file IS configured but the file does not
        exist on disk (never paused): must return True, no raise, paused False."""
        rt = _make_runtime(tmp_path)
        control = tmp_path / "pause.ctrl"
        rt.set_pause_file(control)
        assert not control.exists()
        result = rt.request_resume()
        assert result is True
        assert rt.paused is False
        assert not control.exists()
