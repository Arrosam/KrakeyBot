import pytest

from krakey.self_agent import ParsedSelfOutput, parse_self_output


def test_parses_all_four_sections():
    raw = """[THINKING]
internal monologue text.

[DECISION]
Reply user with hello.

[NOTE]
remember to be polite.

[IDLE]
30
"""
    p = parse_self_output(raw)
    assert "internal monologue" in p.thinking
    assert "Reply user with hello" in p.decision
    assert "polite" in p.note
    assert p.idle_seconds == 30


def test_sections_are_trimmed():
    raw = "[THINKING]\n\nA\n\n[DECISION]\n  B  \n"
    p = parse_self_output(raw)
    assert p.thinking == "A"
    assert p.decision == "B"


def test_fallback_no_markers_treats_whole_as_thinking_and_decision():
    raw = "Just some free text without markers."
    p = parse_self_output(raw)
    assert p.thinking == "Just some free text without markers."
    assert p.decision == "Just some free text without markers."
    assert p.note == ""
    assert p.idle_seconds is None


def test_idle_accepts_suffix():
    raw = "[DECISION]\nx\n[IDLE]\n15 seconds"
    p = parse_self_output(raw)
    assert p.idle_seconds == 15


def test_missing_optional_sections_default_empty():
    raw = "[DECISION]\nAct now."
    p = parse_self_output(raw)
    assert p.decision == "Act now."
    assert p.note == ""
    assert p.thinking == ""
    assert p.idle_seconds is None


def test_sections_out_of_order_still_parses():
    raw = "[DECISION]\nact.\n[THINKING]\nwhy.\n[NOTE]\nremember.\n"
    p = parse_self_output(raw)
    assert p.decision == "act."
    assert p.thinking == "why."
    assert p.note == "remember."


def test_fallback_requires_missing_decision():
    # If DECISION is present but empty, it's still structured (no fallback)
    raw = "[THINKING]\nonly thinking.\n[DECISION]\n\n"
    p = parse_self_output(raw)
    assert p.thinking == "only thinking."
    assert p.decision == ""


def test_parsed_self_output_dataclass_shape():
    p = ParsedSelfOutput(thinking="a", decision="b", note="c", idle_seconds=10)
    assert p.thinking == "a" and p.decision == "b" and p.note == "c"
    assert p.idle_seconds == 10
