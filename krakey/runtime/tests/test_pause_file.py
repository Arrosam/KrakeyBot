"""Behavioral tests for Runtime.set_pause_file / poll_pause_file / .paused.

Spec: a user-controllable pause capability driven by a control file that
the runtime watches.  No internal implementation details are assumed beyond
the three public API members documented in the spec.

Construction pattern follows the established repo convention:
  build_runtime_with_fakes(modifiers=[], ...)  — lightest valid Runtime,
  all plugins disabled, in-memory graph DB, real tmp_path for the pause file.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runtime(tmp_path: pathlib.Path):
    """Minimal Runtime with no plugins — sufficient for pause-file tests."""
    return build_runtime_with_fakes(
        self_llm=ScriptedLLM([]),
        hypo_llm=ScriptedLLM([]),
        embedder=NullEmbedder(),
        modifiers=[],
        gm_path=str(tmp_path / "gm.sqlite"),
    )


# ---------------------------------------------------------------------------
# Positive — valid inputs / expected transitions
# ---------------------------------------------------------------------------


def test_fresh_runtime_paused_is_false(tmp_path):
    """A freshly constructed Runtime must report paused=False before any poll."""
    rt = _make_runtime(tmp_path)
    assert rt.paused is False


def test_no_pause_file_set_poll_is_noop(tmp_path):
    """With no pause file path set, poll_pause_file() does nothing.
    paused stays False and no exception is raised."""
    rt = _make_runtime(tmp_path)
    rt.poll_pause_file()
    assert rt.paused is False


def test_pause_file_path_set_but_absent_paused_false(tmp_path):
    """Path configured, file does not exist -> paused becomes/stays False."""
    rt = _make_runtime(tmp_path)
    control = tmp_path / "pause.ctrl"
    # File deliberately not created.
    rt.set_pause_file(control)
    rt.poll_pause_file()
    assert rt.paused is False


def test_empty_file_triggers_indefinite_pause(tmp_path):
    """An empty control file signals indefinite pause -> paused becomes True."""
    rt = _make_runtime(tmp_path)
    control = tmp_path / "pause.ctrl"
    control.write_text("", encoding="utf-8")
    rt.set_pause_file(control)
    rt.poll_pause_file()
    assert rt.paused is True


def test_non_numeric_content_triggers_indefinite_pause(tmp_path):
    """Non-parseable file content is treated as indefinite pause (not a crash)."""
    rt = _make_runtime(tmp_path)
    control = tmp_path / "pause.ctrl"
    control.write_text("please pause forever\n", encoding="utf-8")
    rt.set_pause_file(control)
    rt.poll_pause_file()
    assert rt.paused is True


def test_future_deadline_pauses_and_file_survives(tmp_path):
    """A future epoch deadline -> paused True, file must still exist after poll."""
    rt = _make_runtime(tmp_path)
    control = tmp_path / "pause.ctrl"
    deadline = time.time() + 86_400  # 24 h in the future
    control.write_text(str(deadline), encoding="utf-8")
    rt.set_pause_file(control)
    rt.poll_pause_file()
    assert rt.paused is True
    assert control.exists(), "file should be left in place while deadline is in the future"


def test_past_deadline_clears_pause_and_deletes_file(tmp_path):
    """A past epoch deadline -> paused becomes False AND the stale file is deleted."""
    rt = _make_runtime(tmp_path)
    control = tmp_path / "pause.ctrl"
    deadline = time.time() - 10.0  # 10 seconds in the past
    control.write_text(str(deadline), encoding="utf-8")
    rt.set_pause_file(control)
    rt.poll_pause_file()
    assert rt.paused is False
    assert not control.exists(), "stale past-deadline file must be unlinked after poll"


# ---------------------------------------------------------------------------
# State transition tests
# ---------------------------------------------------------------------------


def test_transition_file_present_then_removed(tmp_path):
    """Idempotent re-evaluation: file present (paused True), file then removed,
    second poll produces paused False."""
    rt = _make_runtime(tmp_path)
    control = tmp_path / "pause.ctrl"
    control.write_text("", encoding="utf-8")
    rt.set_pause_file(control)

    rt.poll_pause_file()
    assert rt.paused is True, "should be paused while file exists"

    control.unlink()
    rt.poll_pause_file()
    assert rt.paused is False, "should unpause once file is gone"


def test_set_pause_file_none_after_path_makes_poll_noop(tmp_path):
    """set_pause_file(None) after having a path -> subsequent poll is a no-op
    and paused is False regardless of prior state."""
    rt = _make_runtime(tmp_path)
    control = tmp_path / "pause.ctrl"
    control.write_text("", encoding="utf-8")
    rt.set_pause_file(control)
    rt.poll_pause_file()
    assert rt.paused is True  # confirm pause was active

    # Clear the path reference.
    rt.set_pause_file(None)
    rt.poll_pause_file()
    assert rt.paused is False


def test_set_pause_file_none_after_path_poll_is_noop_file_still_exists(tmp_path):
    """After set_pause_file(None), poll must not touch the old file even though
    it is still present on disk (the runtime no longer watches it)."""
    rt = _make_runtime(tmp_path)
    control = tmp_path / "pause.ctrl"
    control.write_text("", encoding="utf-8")
    rt.set_pause_file(control)
    rt.poll_pause_file()
    assert rt.paused is True

    rt.set_pause_file(None)
    rt.poll_pause_file()
    # File should be untouched — the runtime has no path to act on.
    assert control.exists(), "runtime must not delete file it is no longer watching"
    assert rt.paused is False


# ---------------------------------------------------------------------------
# BVA — boundary value tests
# ---------------------------------------------------------------------------


def test_deadline_exactly_now_is_treated_as_expired(tmp_path):
    """A deadline equal to time.time() at read time (now or just past) ->
    paused False and file deleted.  We write a value a few milliseconds in
    the past to reliably satisfy time.time() >= deadline after the write."""
    rt = _make_runtime(tmp_path)
    control = tmp_path / "pause.ctrl"
    # Subtract a small delta so it is guaranteed past by poll time.
    deadline = time.time() - 0.05
    control.write_text(repr(deadline), encoding="utf-8")
    rt.set_pause_file(control)
    rt.poll_pause_file()
    assert rt.paused is False
    assert not control.exists()


def test_whitespace_only_content_treated_as_indefinite_pause(tmp_path):
    """Whitespace-only file content (no parseable number) -> indefinite pause."""
    rt = _make_runtime(tmp_path)
    control = tmp_path / "pause.ctrl"
    control.write_text("   \n\t\n", encoding="utf-8")
    rt.set_pause_file(control)
    rt.poll_pause_file()
    assert rt.paused is True


def test_deadline_with_surrounding_whitespace_parsed_correctly(tmp_path):
    """A deadline with leading/trailing whitespace must still be parsed as a
    valid number (strip + float conversion)."""
    rt = _make_runtime(tmp_path)
    control = tmp_path / "pause.ctrl"
    deadline = time.time() - 5.0  # past
    control.write_text(f"  {deadline}  \n", encoding="utf-8")
    rt.set_pause_file(control)
    rt.poll_pause_file()
    assert rt.paused is False
    assert not control.exists()


def test_zero_as_deadline_is_past(tmp_path):
    """Epoch 0 is deep in the past -> paused False and file deleted."""
    rt = _make_runtime(tmp_path)
    control = tmp_path / "pause.ctrl"
    control.write_text("0", encoding="utf-8")
    rt.set_pause_file(control)
    rt.poll_pause_file()
    assert rt.paused is False
    assert not control.exists()


def test_negative_deadline_is_past(tmp_path):
    """A negative epoch value is also in the past -> paused False and file deleted."""
    rt = _make_runtime(tmp_path)
    control = tmp_path / "pause.ctrl"
    control.write_text("-1.0", encoding="utf-8")
    rt.set_pause_file(control)
    rt.poll_pause_file()
    assert rt.paused is False
    assert not control.exists()


# ---------------------------------------------------------------------------
# Negative / error-guessing tests
# ---------------------------------------------------------------------------


def test_poll_idempotent_when_file_absent(tmp_path):
    """Multiple polls with the file absent must all leave paused=False.
    No exception, no state accumulation."""
    rt = _make_runtime(tmp_path)
    control = tmp_path / "pause.ctrl"
    rt.set_pause_file(control)
    for _ in range(5):
        rt.poll_pause_file()
    assert rt.paused is False


def test_poll_idempotent_with_empty_file(tmp_path):
    """Multiple polls with an empty file must all leave paused=True.
    No exception, no escalating side-effects."""
    rt = _make_runtime(tmp_path)
    control = tmp_path / "pause.ctrl"
    control.write_text("", encoding="utf-8")
    rt.set_pause_file(control)
    for _ in range(5):
        rt.poll_pause_file()
    assert rt.paused is True
    assert control.exists(), "file must not be deleted for indefinite pause"


def test_poll_with_past_deadline_idempotent_after_deletion(tmp_path):
    """After a past-deadline file is deleted by the first poll, subsequent
    polls must behave as if the file is absent (paused=False, no crash)."""
    rt = _make_runtime(tmp_path)
    control = tmp_path / "pause.ctrl"
    deadline = time.time() - 10.0
    control.write_text(str(deadline), encoding="utf-8")
    rt.set_pause_file(control)

    rt.poll_pause_file()
    assert rt.paused is False
    assert not control.exists()

    # Second poll — file is already gone.
    rt.poll_pause_file()
    assert rt.paused is False


def test_non_parseable_multiline_junk_does_not_crash(tmp_path):
    """Multiline junk content must be handled gracefully (treated as
    indefinite pause) and must not raise any exception."""
    rt = _make_runtime(tmp_path)
    control = tmp_path / "pause.ctrl"
    control.write_bytes(b"not a number\r\nstill not\x00\xff\n")
    rt.set_pause_file(control)
    rt.poll_pause_file()
    assert rt.paused is True


def test_paused_is_false_after_set_none_never_polled(tmp_path):
    """set_pause_file(None) on a runtime that has never polled must leave
    paused=False (initial invariant preserved)."""
    rt = _make_runtime(tmp_path)
    rt.set_pause_file(None)
    assert rt.paused is False


def test_changing_path_mid_lifecycle(tmp_path):
    """set_pause_file can be called multiple times; the most recent path wins.
    Switching from a live pause file to an absent one must clear the pause
    on the next poll."""
    rt = _make_runtime(tmp_path)

    # First path — empty file, induces pause.
    first = tmp_path / "pause_a.ctrl"
    first.write_text("", encoding="utf-8")
    rt.set_pause_file(first)
    rt.poll_pause_file()
    assert rt.paused is True

    # Switch to a second path that does not exist.
    second = tmp_path / "pause_b.ctrl"
    rt.set_pause_file(second)
    rt.poll_pause_file()
    assert rt.paused is False
