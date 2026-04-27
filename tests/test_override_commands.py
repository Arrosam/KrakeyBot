"""Override CLI commands: /status /memory_stats /sleep /kill."""
import pytest

from src.runtime.overrides.override_commands import (
    KNOWN_COMMANDS, OverrideAction, parse_override,
)


# ---------------- parser ----------------

def test_known_command_recognized():
    assert parse_override("/status") == "status"
    assert parse_override("/memory_stats") == "memory_stats"
    assert parse_override("/sleep") == "sleep"
    assert parse_override("/kill") == "kill"


def test_command_with_trailing_args_returns_just_command():
    assert parse_override("/status now please") == "status"
    assert parse_override("/sleep   ") == "sleep"


def test_case_insensitive():
    assert parse_override("/STATUS") == "status"
    assert parse_override("/Sleep") == "sleep"


def test_normal_text_returns_none():
    assert parse_override("hello") is None
    assert parse_override("") is None
    assert parse_override(None) is None


def test_unknown_slash_command_returns_none():
    assert parse_override("/foobar") is None
    assert parse_override("/nonsense") is None


def test_known_commands_set_complete():
    assert KNOWN_COMMANDS == {"status", "memory_stats", "sleep", "kill"}


def test_overrideaction_enum_members():
    # The enum exposes the meaningful side-effects callers need to react to.
    assert OverrideAction.NONE.name == "NONE"
    assert OverrideAction.SLEEP.name == "SLEEP"
    assert OverrideAction.KILL.name == "KILL"
