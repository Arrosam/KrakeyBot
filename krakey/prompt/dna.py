"""Layer-0 DNA instructions — loaded from sibling ``dna.txt``.

Fixed system prompt. Not user-configurable at runtime BUT the
file is mtime-checked on every prompt build (see ``get_dna()``)
so iterating on prompt wording during development doesn't need
a runtime restart. Edit the file, save, next heartbeat picks it
up.

DNA = mechanics only. Input/output format, perception, action, memory,
sleep, inner voice. No identity. No behavioral norms. Identity lives
in GENESIS (Bootstrap once) and persists via self_model.

Import surface (unchanged):

    from krakey.prompt.dna import DNA, get_dna

``DNA`` is the module-level constant captured at import time —
preserved for callers (and tests) that care about a stable
snapshot. ``get_dna()`` reads from disk if mtime changed, returns
the cached value otherwise. Production prompt builders use
``get_dna()`` so live edits propagate.
"""
from __future__ import annotations

from pathlib import Path


_DNA_PATH: Path = Path(__file__).parent / "dna.txt"
_cached_text: str = _DNA_PATH.read_text(encoding="utf-8")
_cached_mtime: float = _DNA_PATH.stat().st_mtime

# Module-level constant — captured once at import. Tests + any
# code that wants the snapshot at process-start time use this.
DNA: str = _cached_text


def get_dna() -> str:
    """Return the DNA text, re-reading from disk if dna.txt's
    mtime changed since the last call.

    Cheap path: the only filesystem op when the file is unchanged
    is a single ``stat()`` per call. Any failure (file deleted /
    permission revoked / I/O error) falls back to the cached
    text so the heartbeat never crashes on a transient disk
    issue."""
    global _cached_text, _cached_mtime
    try:
        mtime = _DNA_PATH.stat().st_mtime
    except OSError:
        return _cached_text
    if mtime != _cached_mtime:
        try:
            _cached_text = _DNA_PATH.read_text(encoding="utf-8")
            _cached_mtime = mtime
        except OSError:
            # Don't update cache; next call retries.
            pass
    return _cached_text
