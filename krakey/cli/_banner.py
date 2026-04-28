"""Cool ASCII banner printed by `krakey`, `krakey --help`, `krakey run`,
`krakey onboard` — anything user-facing where flair beats noise.
"""
from __future__ import annotations

import sys
from typing import IO

from . import _meta

_BAR = "░▒▓" + ("█" * 51) + "▓▒░"   # 57-col gradient bar
_BAR_WIDTH = len(_BAR)               # 57

_LOGO = """\
   ██╗  ██╗ ██████╗   █████╗  ██╗  ██╗ ███████╗ ██╗   ██╗
   ██║ ██╔╝ ██╔══██╗ ██╔══██╗ ██║ ██╔╝ ██╔════╝ ╚██╗ ██╔╝
   █████╔╝  ██████╔╝ ███████║ █████╔╝  █████╗    ╚████╔╝
   ██╔═██╗  ██╔══██╗ ██╔══██║ ██╔═██╗  ██╔══╝     ╚██╔╝
   ██║  ██╗ ██║  ██║ ██║  ██║ ██║  ██╗ ███████╗    ██║
   ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚══════╝    ╚═╝   \
"""

_TAGLINE = "▓░  u l t i m a t e   a u t o n o m o u s   a g e n t  ░▓"


def _ensure_utf8(stream) -> None:
    """Best-effort: switch a text stream to UTF-8 so the box-drawing /
    shaded chars render on Windows consoles that default to GBK / cp1252.
    """
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass


def print_banner(file: IO[str] | None = None) -> None:
    """Render the KRAKEY banner to `file` (defaults to stdout)."""
    out = file if file is not None else sys.stdout
    _ensure_utf8(out)
    version_line = f"[ v {_meta.version()} ]".center(_BAR_WIDTH)
    print(_BAR, file=out)
    print(file=out)
    print(_LOGO, file=out)
    print(file=out)
    print(_TAGLINE, file=out)
    print(version_line, file=out)
    print(file=out)
    print(_BAR, file=out)
