"""GENESIS.md is only read when Bootstrap is active.

Steady-state Runtimes (bootstrap complete, GM has nodes) must never
touch the file. This guards against two failure modes:
  1. wasted I/O on cold-start for the common steady-state case
  2. stale GENESIS bytes sitting on Runtime that could leak into
     prompts later via a misrouted code path
"""
from pathlib import Path

import pytest

from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes


def test_genesis_not_read_in_steady_state(tmp_path, monkeypatch):
    """Fresh Runtime() in non-bootstrap mode must not open GENESIS."""
    opened = {"n": 0}
    real_read_text = Path.read_text

    def _spy_read(self, *args, **kwargs):
        if self.name.lower() == "genesis.md":
            opened["n"] += 1
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _spy_read)

    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
        skip_bootstrap=True,  # pins is_bootstrap=False
    )
    # Sanity: nothing asked for GENESIS during Runtime construction
    assert opened["n"] == 0
    assert runtime._genesis_text is None  # lazy slot still empty


def test_genesis_read_on_first_bootstrap_prompt(tmp_path, monkeypatch):
    """When Bootstrap is active and we build a Self prompt, the
    lazy accessor loads GENESIS. Subsequent calls hit the cache
    (no re-read)."""
    reads = {"n": 0}
    real_read_text = Path.read_text

    def _spy_read(self, *args, **kwargs):
        if self.name.lower() == "genesis.md":
            reads["n"] += 1
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _spy_read)

    # Write a real GENESIS file into the runtime workspace
    (tmp_path / "GENESIS.md").write_text("hello young agent",
                                            encoding="utf-8")
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    # Point the genesis path at our fake file + flip bootstrap on.
    runtime._genesis_path = str(tmp_path / "GENESIS.md")
    runtime.is_bootstrap = True

    # First access reads the file
    txt1 = runtime._get_genesis_text()
    assert reads["n"] == 1
    assert "hello young agent" in txt1
    # Second access is cached — no new read
    txt2 = runtime._get_genesis_text()
    assert reads["n"] == 1
    assert txt1 is txt2


def test_missing_genesis_returns_placeholder_lazily(tmp_path):
    """No file at genesis_path → loader returns the placeholder
    string (not an exception) so the bot still boots."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    runtime._genesis_path = str(tmp_path / "does_not_exist.md")
    text = runtime._get_genesis_text()
    assert text  # non-empty placeholder
    assert "GENESIS" in text or "\u767d\u677f" in text
