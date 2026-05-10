"""Tests for ``krakey.prompt.dna.get_dna()`` — mtime-based
hot-reload of the DNA prompt template.

Pinned behaviors:
  * Initial call returns the file's content (matches the
    module-level DNA constant).
  * Editing dna.txt + bumping its mtime → next call returns the
    NEW text.
  * stat() failure → falls back to cached text, doesn't crash.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest


def _bump_mtime(path):
    future = time.time() + 1.0
    os.utime(path, (future, future))


def test_get_dna_returns_current_file_content_on_first_call():
    from krakey.prompt import dna as dna_mod

    out = dna_mod.get_dna()
    assert out == dna_mod.DNA
    assert "DNA" in out


def test_get_dna_re_reads_when_mtime_changes(monkeypatch):
    """Touch dna.txt → next get_dna() returns new content."""
    from krakey.prompt import dna as dna_mod

    original = dna_mod.get_dna()
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False, suffix=".txt",
    )
    try:
        tmp.write("REPLACED CONTENT\n")
        tmp.close()
        _bump_mtime(tmp.name)
        monkeypatch.setattr(dna_mod, "_DNA_PATH", Path(tmp.name))
        monkeypatch.setattr(dna_mod, "_cached_mtime", 0.0)
        out = dna_mod.get_dna()
        assert out == "REPLACED CONTENT\n"
        assert out != original
    finally:
        os.unlink(tmp.name)


def test_get_dna_falls_back_on_stat_failure(monkeypatch):
    """If stat() raises OSError, get_dna returns the cached text
    rather than crashing the prompt builder."""
    from krakey.prompt import dna as dna_mod

    cached = dna_mod.get_dna()

    class _BadPath:
        def stat(self):
            raise OSError("simulated")

        def read_text(self, encoding="utf-8"):
            raise OSError("simulated")

    monkeypatch.setattr(dna_mod, "_DNA_PATH", _BadPath())
    out = dna_mod.get_dna()
    assert out == cached


def test_get_dna_skips_repeated_read_when_unchanged():
    """If mtime hasn't changed, get_dna doesn't re-read.
    Verify with a monkeypatched read_text counter."""
    from krakey.prompt import dna as dna_mod

    dna_mod.get_dna()  # prime cache

    read_count = [0]
    original_read = dna_mod._DNA_PATH.read_text

    def counting_read(*a, **kw):
        read_count[0] += 1
        return original_read(*a, **kw)

    real_path = dna_mod._DNA_PATH

    class _SpyPath:
        def stat(self_inner):
            return real_path.stat()

        def read_text(self_inner, *a, **kw):
            return counting_read(*a, **kw)

    saved = dna_mod._DNA_PATH
    dna_mod._DNA_PATH = _SpyPath()
    try:
        for _ in range(5):
            dna_mod.get_dna()
    finally:
        dna_mod._DNA_PATH = saved
    assert read_count[0] == 0  # mtime stable → no read
