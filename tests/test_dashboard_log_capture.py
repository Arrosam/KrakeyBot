"""Tee + ring-buffer behaviour for the Log tab's backend."""
from __future__ import annotations

import io

from krakey.plugins.dashboard.log_capture import LogCapture, _LogTee


def test_log_tee_mirrors_writes_to_original():
    """Original stream still receives every write; line callback fires
    once per completed line, ignoring partial fragments until EOL."""
    sink = io.StringIO()
    captured: list[str] = []
    tee = _LogTee(sink, captured.append)

    tee.write("hello ")
    tee.write("world\n")
    tee.write("partial")  # no newline → not delivered yet
    assert sink.getvalue() == "hello world\npartial"
    assert captured == ["hello world"]

    tee.write("-tail\nnext\n")
    assert captured == ["hello world", "partial-tail", "next"]


def test_log_tee_isatty_passes_through():
    class _FakeTTY:
        def write(self, s): return len(s)
        def flush(self): pass
        def isatty(self): return True
    tee = _LogTee(_FakeTTY(), lambda _l: None)
    assert tee.isatty() is True


def test_log_capture_ring_buffer_caps():
    """Ring drops oldest lines past history_size."""
    cap = LogCapture(history_size=3)
    sink = io.StringIO()
    tee = _LogTee(sink, cap._on_line)
    for i in range(5):
        tee.write(f"line {i}\n")
    assert cap.recent() == ["line 2", "line 3", "line 4"]


def test_log_capture_subscribers_get_each_line():
    cap = LogCapture(history_size=10)
    received: list[str] = []
    cap.subscribe(received.append)
    sink = io.StringIO()
    tee = _LogTee(sink, cap._on_line)
    tee.write("alpha\nbeta\n")
    assert received == ["alpha", "beta"]


def test_log_capture_unsubscribe_stops_delivery():
    cap = LogCapture(history_size=10)
    received: list[str] = []
    cap.subscribe(received.append)
    sink = io.StringIO()
    tee = _LogTee(sink, cap._on_line)
    tee.write("first\n")
    cap.unsubscribe(received.append)
    tee.write("second\n")
    assert received == ["first"]


def test_log_capture_install_is_idempotent():
    """A second install() doesn't double-tee (which would dup every
    line in the ring)."""
    import sys
    cap = LogCapture(history_size=5)
    orig_stdout = sys.stdout
    try:
        cap.install()
        cap.install()
        # Double-installation would make a write produce 2 ring entries.
        sys.stdout.write("once\n")
        sys.stdout.flush()
        # Allow a couple of ring entries from earlier startup logs;
        # the contract is just that "once" appears at most once.
        assert cap.recent().count("once") == 1
    finally:
        sys.stdout = orig_stdout
        sys.stderr = sys.__stderr__
