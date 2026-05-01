"""KRAKEY banner printed by `krakey`, `krakey --help`, `krakey run`,
`krakey onboard` — anything user-facing where flair beats noise.
"""
from __future__ import annotations

import sys
from typing import IO

from . import _meta


_LOGO = r"""

    d8b                           d8b
    ?88                           ?88
     88b                           88b
     888  d88'  88bd88b d888b8b    888  d88' d8888b?88   d8P
     888bd8P'   88P'  `d8P' ?88    888bd8P' d8b_,dPd88   88
    d88888b    d88     88b  ,88b  d88888b   88b    ?8(  d88
    d88' `?88b,d88'     `?88P'`88bd88' `?88b,`?888P'`?88P'?8b
                                                          )88
                                                          ,d8P
                                                      `?888P'

        u l t i m a t e   a u t o n o m o u s   a g e n t

"""

# Width of the widest logo line — used to center the version line.
_LOGO_WIDTH = max(len(line) for line in _LOGO.splitlines() if line)


def _ensure_utf8(stream) -> None:
    """Best-effort: switch a text stream to UTF-8 so the figlet
    output (which is pure ASCII so this rarely matters) renders
    consistently on Windows consoles that default to GBK / cp1252."""
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass


def print_banner(file: IO[str] | None = None) -> None:
    """Render the KRAKEY banner to `file` (defaults to stdout)."""
    out = file if file is not None else sys.stdout
    _ensure_utf8(out)
    version_line = f"v {_meta.version()}".center(_LOGO_WIDTH)
    print(_LOGO, file=out, end="")
    print(version_line, file=out)
    print(file=out)
