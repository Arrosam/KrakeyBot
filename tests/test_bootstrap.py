"""Bootstrap parsers + GENESIS loader."""
import json
from pathlib import Path

import pytest

from krakey.bootstrap import (
    BOOTSTRAP_PROMPT, detect_bootstrap_complete, load_genesis,
    load_self_model_or_default, parse_self_model_update,
)


# ---------------- self-model update parser ----------------

def test_parse_self_model_extracts_json_block():
    """Parser is schema-agnostic — it just extracts the JSON. The
    loader is responsible for filtering against the slim schema."""
    note = """Some thoughts...
<self-model>
{"identity": {"name": "Krakey", "persona": "curious"}}
</self-model>
more notes."""
    out = parse_self_model_update(note)
    assert out == {"identity": {"name": "Krakey", "persona": "curious"}}


def test_parse_self_model_no_block_returns_none():
    assert parse_self_model_update("just a note") is None


def test_parse_self_model_handles_self_model_alias():
    """Accept both <self-model> and <selfmodel>."""
    note = '<selfmodel>{"a": 1}</selfmodel>'
    assert parse_self_model_update(note) == {"a": 1}


def test_parse_self_model_invalid_json_returns_none():
    note = "<self-model>{not json}</self-model>"
    assert parse_self_model_update(note) is None


# ---------------- bootstrap-complete detection ----------------

def test_detect_bootstrap_complete_string():
    assert detect_bootstrap_complete("All done. bootstrap complete") is True
    assert detect_bootstrap_complete("BOOTSTRAP COMPLETE here") is True
    assert detect_bootstrap_complete("bootstrap   complete") is True


def test_detect_bootstrap_complete_negative():
    assert detect_bootstrap_complete("still figuring out") is False
    assert detect_bootstrap_complete("") is False
    assert detect_bootstrap_complete(None) is False


# ---------------- GENESIS loader ----------------

def test_load_genesis_existing_file(tmp_path):
    p = tmp_path / "GENESIS.md"
    p.write_text("# my genesis\nhello", encoding="utf-8")
    text = load_genesis(p)
    assert "my genesis" in text


def test_load_genesis_missing_file_returns_placeholder(tmp_path):
    text = load_genesis(tmp_path / "missing.md")
    assert text  # non-empty placeholder
    assert "blank" in text.lower() or "GENESIS" in text


# ---------------- self-model loader ----------------

def test_load_self_model_missing_file_starts_bootstrap(tmp_path):
    p = tmp_path / "self_model.yaml"
    sm, is_bootstrap = load_self_model_or_default(p)
    assert is_bootstrap is True
    assert sm["state"]["bootstrap_complete"] is False


def test_load_self_model_existing_with_complete_flag_skips_bootstrap(tmp_path):
    """Legacy YAMLs with all the pre-2026-04-25 fields must still
    load: keep what's in the slim schema (identity, state.bootstrap_complete),
    silently drop the rest, and rewrite the file to match.
    """
    p = tmp_path / "self_model.yaml"
    p.write_text(
        "identity: {name: Krakey, persona: ''}\n"
        "state: {bootstrap_complete: true, mood_baseline: neutral, "
        "       energy_level: 1.0, focus_topic: '', is_sleeping: false}\n"
        "goals: {active: [], completed: []}\n"
        "relationships: {users: []}\n"
        "statistics: {total_heartbeats: 0, total_sleep_cycles: 0, "
        "             uptime_hours: 0.0, first_boot: '', last_heartbeat: '',\n"
        "             last_sleep: ''}\n",
        encoding="utf-8",
    )
    sm, is_bootstrap = load_self_model_or_default(p)
    assert is_bootstrap is False
    assert sm["identity"]["name"] == "Krakey"
    # Slim schema enforced — legacy keys gone from the in-memory dict
    assert "goals" not in sm
    assert "relationships" not in sm
    assert "statistics" not in sm
    for legacy in ("mood_baseline", "energy_level", "focus_topic",
                    "is_sleeping"):
        assert legacy not in sm["state"]
    # Migration was persisted: file no longer contains legacy keys
    rewritten = p.read_text(encoding="utf-8")
    for token in ("mood_baseline", "statistics", "relationships",
                   "total_heartbeats", "is_sleeping", "goals",
                   "focus_topic", "energy_level"):
        assert token not in rewritten, (
            f"{token!r} survived migration write-back"
        )


def test_load_self_model_no_rewrite_when_already_slim(tmp_path):
    """If the YAML is already in the slim schema, the loader should
    NOT rewrite it (avoids needless mtime churn + log spam)."""
    p = tmp_path / "self_model.yaml"
    p.write_text(
        "identity:\n  name: Krakey\n  persona: curious\n"
        "state:\n  bootstrap_complete: true\n",
        encoding="utf-8",
    )
    mtime_before = p.stat().st_mtime_ns
    load_self_model_or_default(p)
    mtime_after = p.stat().st_mtime_ns
    assert mtime_after == mtime_before, "loader rewrote a clean file"


def test_load_self_model_incomplete_flag_starts_bootstrap(tmp_path):
    p = tmp_path / "self_model.yaml"
    p.write_text(
        "state: {bootstrap_complete: false}\n", encoding="utf-8",
    )
    _sm, is_bootstrap = load_self_model_or_default(p)
    assert is_bootstrap is True


# ---------------- prompt template ----------------

def test_bootstrap_prompt_contains_genesis_placeholder():
    assert "{genesis_text}" in BOOTSTRAP_PROMPT
    # Must instruct Self how to update self_model and signal completion
    assert "<self-model>" in BOOTSTRAP_PROMPT
    assert "bootstrap complete" in BOOTSTRAP_PROMPT.lower()
