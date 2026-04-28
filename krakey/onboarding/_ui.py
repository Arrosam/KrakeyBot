"""Terminal-UI helpers for the onboarding wizard.

Two responsibilities, both fail-safe:

  * ANSI-color formatting (`bold`, `dim`, `red`, `green`, `yellow`,
    `cyan`, `magenta`). Codes are emitted only when stdout is an
    interactive TTY — when the wizard's `output_fn` is a test fake
    that appends to a list, the helpers degrade to plain strings.
  * Single-key raw reads (`read_key`) for the interactive plugin
    picker — used by the arrow-key UI. Falls back gracefully
    everywhere; callers check `is_interactive()` first.

Stdlib only. The wizard refuses to depend on a TUI library so it
keeps working on every Python install Krakey runs on.
"""
from __future__ import annotations

import os
import sys


# --------------------------------------------------------------------
# Capability detection
# --------------------------------------------------------------------

def is_interactive() -> bool:
    """True when both stdin and stdout are connected to a real TTY.
    The interactive features (colors, arrow-key picker) only kick in
    when this is True; tests with stub I/O always evaluate False."""
    return bool(
        getattr(sys.stdin, "isatty", lambda: False)()
        and getattr(sys.stdout, "isatty", lambda: False)()
    )


_VT_ENABLED = False


def enable_vt_on_windows() -> None:
    """Turn on ANSI escape processing for legacy Windows consoles.

    Windows Terminal / VS Code / modern PowerShell already process VT
    sequences. Older cmd.exe doesn't unless the console is flipped
    into VT mode via `SetConsoleMode`. Cheap idempotent call — no-op
    on Unix, no-op on second invocation."""
    global _VT_ENABLED
    if _VT_ENABLED or os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes  # noqa: F401

        kernel32 = ctypes.windll.kernel32
        STD_OUTPUT_HANDLE = -11
        ENABLE_VT = 0x0004
        handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | ENABLE_VT)
        _VT_ENABLED = True
    except Exception:  # noqa: BLE001
        # Any failure → ANSI may not render, but the wizard's text
        # is still readable. Don't crash the install.
        pass


# --------------------------------------------------------------------
# ANSI color formatting
# --------------------------------------------------------------------

_RESET = "\033[0m"
_CODES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "bright_cyan": "\033[96m",
    "bright_yellow": "\033[93m",
    "bright_green": "\033[92m",
    "bright_red": "\033[91m",
}


def color(text: str, *names: str) -> str:
    """Wrap `text` in ANSI codes for the named styles. Stacks (e.g.
    `color("hi", "bold", "cyan")`). Plain string when not interactive."""
    if not is_interactive() or not names:
        return text
    prefix = "".join(_CODES[n] for n in names if n in _CODES)
    if not prefix:
        return text
    return f"{prefix}{text}{_RESET}"


# Convenience wrappers — same signature, easier to read at call sites.
def bold(s: str) -> str: return color(s, "bold")
def dim(s: str) -> str: return color(s, "dim")
def red(s: str) -> str: return color(s, "red")
def green(s: str) -> str: return color(s, "green")
def yellow(s: str) -> str: return color(s, "yellow")
def cyan(s: str) -> str: return color(s, "cyan")
def magenta(s: str) -> str: return color(s, "magenta")


# --------------------------------------------------------------------
# Raw single-key reads (for the interactive plugin picker)
# --------------------------------------------------------------------

# Token strings returned by `read_key` so callers don't deal with
# platform-specific byte sequences.
KEY_UP = "up"
KEY_DOWN = "down"
KEY_LEFT = "left"
KEY_RIGHT = "right"
KEY_SPACE = "space"
KEY_ENTER = "enter"
KEY_ESC = "esc"


def read_key() -> str:
    """Block until one keystroke arrives; return a token like
    `'up' / 'down' / 'space' / 'enter' / 'esc'`, or the literal
    character for printables. Raw mode — no Enter required."""
    if os.name == "nt":
        return _read_key_windows()
    return _read_key_posix()


def _read_key_windows() -> str:
    import msvcrt
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):
        ch2 = msvcrt.getwch()
        return {
            "H": KEY_UP, "P": KEY_DOWN,
            "K": KEY_LEFT, "M": KEY_RIGHT,
        }.get(ch2, "unknown")
    if ch in ("\r", "\n"):
        return KEY_ENTER
    if ch == " ":
        return KEY_SPACE
    if ch == "\x1b":
        return KEY_ESC
    return ch


def _read_key_posix() -> str:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            # Possible CSI escape — peek 2 more chars (with timeout
            # via select would be nicer, but blocking-read is OK
            # because real arrow keys arrive together).
            try:
                seq = ch + sys.stdin.read(2)
            except Exception:  # noqa: BLE001
                return KEY_ESC
            return {
                "\x1b[A": KEY_UP, "\x1b[B": KEY_DOWN,
                "\x1b[C": KEY_RIGHT, "\x1b[D": KEY_LEFT,
            }.get(seq, KEY_ESC)
        if ch in ("\r", "\n"):
            return KEY_ENTER
        if ch == " ":
            return KEY_SPACE
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
