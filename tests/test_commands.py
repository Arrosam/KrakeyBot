"""Slash-commands: /status /memory_stats /sleep /kill."""
import pytest

from src.runtime.commands.commands import (
    KNOWN_COMMANDS, CommandAction, parse_command,
)


# ---------------- parser ----------------

def test_known_command_recognized():
    assert parse_command("/status") == "status"
    assert parse_command("/memory_stats") == "memory_stats"
    assert parse_command("/sleep") == "sleep"
    assert parse_command("/kill") == "kill"


def test_command_with_trailing_args_returns_just_command():
    assert parse_command("/status now please") == "status"
    assert parse_command("/sleep   ") == "sleep"


def test_case_insensitive():
    assert parse_command("/STATUS") == "status"
    assert parse_command("/Sleep") == "sleep"


def test_normal_text_returns_none():
    assert parse_command("hello") is None
    assert parse_command("") is None
    assert parse_command(None) is None


def test_unknown_slash_command_returns_none():
    assert parse_command("/foobar") is None
    assert parse_command("/nonsense") is None


def test_known_commands_set_complete():
    assert KNOWN_COMMANDS == {"status", "memory_stats", "sleep", "kill"}


def test_command_action_enum_members():
    # The enum exposes the meaningful side-effects callers need to react to.
    assert CommandAction.NONE.name == "NONE"
    assert CommandAction.SLEEP.name == "SLEEP"
    assert CommandAction.KILL.name == "KILL"
