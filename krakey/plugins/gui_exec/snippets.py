"""Pure Python-source builders for ``gui_exec`` actions.

Each public function in this module returns a Python source string
that, when run by the target env's Python interpreter, performs
the named GUI action via ``pyautogui``. The strings are dispatched
as ``[python, "-c", <snippet>]`` argv lists through
``Environment.run``.

The snippets stay one-liners (statements joined by ``;``) so they
fit cleanly inside ``-c`` without a heredoc on either Windows or
POSIX shells. ``repr()`` is used for every embedded string so
quotes/backslashes/unicode round-trip safely on either side.

Pure: no side effects, no I/O, no module-level state. Easy to
unit-test by string equality.
"""
from __future__ import annotations

from pathlib import Path


def _q(s: str) -> str:
    """Python-source-safe string literal. ``repr`` handles every
    embedded quote/backslash/control character correctly."""
    return repr(s)


def click(x: int, y: int, button: str = "left") -> str:
    """Click at (x, y) with the named mouse button.

    ``button`` is one of pyautogui's accepted values: ``"left"``,
    ``"right"``, ``"middle"``. The caller is responsible for passing
    a valid value; pyautogui will raise inside the env on bad input
    and the tool surfaces the non-zero rc as an error Stimulus.
    """
    return (
        "import pyautogui;"
        f"pyautogui.click({int(x)}, {int(y)}, button={_q(button)})"
    )


def double_click(x: int, y: int) -> str:
    return (
        "import pyautogui;"
        f"pyautogui.doubleClick({int(x)}, {int(y)})"
    )


def drag(
    x1: int, y1: int, x2: int, y2: int, duration: float = 0.5,
) -> str:
    """Move to (x1, y1) then drag-with-left-button-held to (x2, y2)
    over ``duration`` seconds. ``moveTo`` first so the drag starts
    from a known position regardless of where the cursor was."""
    return (
        "import pyautogui;"
        f"pyautogui.moveTo({int(x1)}, {int(y1)});"
        f"pyautogui.dragTo({int(x2)}, {int(y2)}, "
        f"duration={float(duration)}, button='left')"
    )


def type_text(text: str, interval: float = 0.0) -> str:
    """Type ``text`` character by character. ``pyautogui.typewrite``
    only handles printable ASCII + a fixed set of named keys; the
    caller should use ``key`` for unprintable chords."""
    return (
        "import pyautogui;"
        f"pyautogui.typewrite({_q(text)}, "
        f"interval={float(interval)})"
    )


def key(combo: str) -> str:
    """Press a single key or hotkey chord.

    Single key (no ``+``) → ``pyautogui.press(<key>)``.
    Multi-key chord (``ctrl+shift+t``) → ``pyautogui.hotkey(*parts)``,
    which presses each key in order then releases in reverse —
    pyautogui's standard hotkey semantics.

    Empty / whitespace-only ``combo`` is a caller bug; the tool
    rejects it before reaching this builder.
    """
    parts = [p.strip() for p in combo.split("+") if p.strip()]
    if len(parts) == 1:
        return (
            "import pyautogui;"
            f"pyautogui.press({_q(parts[0])})"
        )
    args = ", ".join(_q(p) for p in parts)
    return (
        "import pyautogui;"
        f"pyautogui.hotkey({args})"
    )


def screenshot(out_path: Path | str) -> str:
    """Save a screenshot to ``out_path``. ``os.makedirs`` of the
    parent directory is included so the very first call (when
    ``workspace/data/screenshots/`` may not exist yet inside the env)
    succeeds. ``exist_ok=True`` so subsequent calls are no-ops."""
    p = str(out_path)
    return (
        "import os, pyautogui;"
        f"os.makedirs(os.path.dirname({_q(p)}) or '.', exist_ok=True);"
        f"pyautogui.screenshot({_q(p)})"
    )
