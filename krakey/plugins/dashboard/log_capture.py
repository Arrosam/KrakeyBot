"""Tee sys.stdout / sys.stderr into a ring buffer + WS broadcaster.

Why
---
The dashboard's "Log" tab shows the running daemon's heartbeat log
live in the browser, with ANSI colours preserved (parsed client-side
into CSS-classed spans). Useful when the user runs ``krakey start``
(daemonized, no terminal) and otherwise has to ``tail -f`` a log
file just to watch the runtime tick.

The tee is non-destructive:
  * the original stream still receives every write — terminal output
    when interactive, the daemon's log file when daemonized,
  * isatty() / fileno() / flush() pass through to the underlying
    stream so callers that probe them keep working,
  * partial writes are buffered until a newline arrives so we don't
    fan out half-formed lines.

Thread-safety: stdout writes happen on whatever thread runs the
print() (mostly the runtime's main thread). Subscribers are typically
managed on the dashboard's loop and dispatched via
``run_coroutine_threadsafe`` by the WS route. We don't lock — the
shared lists / deque are short and the rare race (a line landing
mid-iteration) is benign.
"""
from __future__ import annotations

import sys
from collections import deque
from typing import Callable


class _LogTee:
    """File-like proxy that mirrors writes to ``original`` while
    also passing each completed line to ``on_line``."""

    def __init__(self, original, on_line: Callable[[str], None]):
        self._orig = original
        self._on_line = on_line
        self._buf = ""

    def write(self, s: str) -> int:
        try:
            self._orig.write(s)
        except Exception:  # noqa: BLE001 — never break a print()
            pass
        if not isinstance(s, str):
            return 0
        self._buf += s
        # Drain complete lines; whatever's after the last newline
        # stays buffered until the next write rounds it out.
        while "\n" in self._buf:
            line, _, self._buf = self._buf.partition("\n")
            try:
                self._on_line(line)
            except Exception:  # noqa: BLE001
                pass
        return len(s)

    def flush(self):
        try:
            self._orig.flush()
        except Exception:  # noqa: BLE001
            pass

    def isatty(self):
        try:
            return self._orig.isatty()
        except Exception:  # noqa: BLE001
            return False

    def fileno(self):
        return self._orig.fileno()

    @property
    def encoding(self):
        return getattr(self._orig, "encoding", "utf-8")


class LogCapture:
    """Ring buffer + subscriber list for tee'd log lines."""

    def __init__(self, *, history_size: int = 500):
        self._recent: deque[str] = deque(maxlen=history_size)
        self._subscribers: list[Callable[[str], None]] = []
        self._installed = False
        self._orig_stdout = None
        self._orig_stderr = None

    def install(self) -> None:
        """Install tees on ``sys.stdout`` and ``sys.stderr``. Idempotent
        (a second call is a no-op) so a runtime that re-loads the
        dashboard plugin doesn't double-tee."""
        if self._installed:
            return
        self._installed = True
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _LogTee(sys.stdout, self._on_line)
        sys.stderr = _LogTee(sys.stderr, self._on_line)

    def _on_line(self, line: str) -> None:
        self._recent.append(line)
        for sub in list(self._subscribers):
            try:
                sub(line)
            except Exception:  # noqa: BLE001
                pass

    def recent(self) -> list[str]:
        return list(self._recent)

    def subscribe(self, fn: Callable[[str], None]) -> None:
        self._subscribers.append(fn)

    def unsubscribe(self, fn: Callable[[str], None]) -> None:
        try:
            self._subscribers.remove(fn)
        except ValueError:
            pass
