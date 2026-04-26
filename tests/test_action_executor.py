"""Action executor — parses [ACTION] JSONL into TentacleCalls.

This is the default tentacle dispatch path when no hypothalamus
Reflect is registered (Reflect #1 default OFF design).
"""
from src.runtime.action_executor import parse_action_block


def test_empty_input_returns_empty():
    assert parse_action_block("") == []
    assert parse_action_block(None) == []  # type: ignore[arg-type]


def test_no_action_block_returns_empty():
    """Self can produce thinking/decision/note without invoking any
    tentacle — the executor should accept that as a valid no-op."""
    text = """[THINKING]
Just reflecting today.
[DECISION]
No action needed.
[NOTE]
Nothing to do.
"""
    assert parse_action_block(text) == []


def test_single_call_parsed():
    text = """[DECISION]
Greet the user.
[ACTION]
{"name": "web_chat_reply", "arguments": {"text": "Hi!"}}
[/ACTION]
"""
    calls = parse_action_block(text)
    assert len(calls) == 1
    c = calls[0]
    assert c.tentacle == "web_chat_reply"
    assert c.params == {"text": "Hi!"}
    assert c.adrenalin is False


def test_multiple_calls_in_one_block():
    text = """[ACTION]
{"name": "web_chat_reply", "arguments": {"text": "ok"}}
{"name": "search", "arguments": {"query": "weather"}, "adrenalin": true}
[/ACTION]
"""
    calls = parse_action_block(text)
    assert len(calls) == 2
    assert calls[0].tentacle == "web_chat_reply"
    assert calls[1].tentacle == "search"
    assert calls[1].adrenalin is True


def test_multiple_blocks_concatenated():
    """Edge case: Self might emit two [ACTION] blocks (e.g. one
    after [DECISION], one after [NOTE]). Both should parse."""
    text = """[ACTION]
{"name": "first", "arguments": {}}
[/ACTION]
some text in between
[ACTION]
{"name": "second", "arguments": {}}
[/ACTION]
"""
    calls = parse_action_block(text)
    assert [c.tentacle for c in calls] == ["first", "second"]


def test_one_bad_line_skipped_others_parse():
    """Single-line failure must not poison adjacent good calls."""
    text = """[ACTION]
{"name": "good_one", "arguments": {}}
this is not json
{"name": "good_two", "arguments": {"x": 1}}
[/ACTION]
"""
    calls = parse_action_block(text)
    assert [c.tentacle for c in calls] == ["good_one", "good_two"]


def test_missing_name_skipped():
    text = """[ACTION]
{"name": "ok", "arguments": {}}
{"arguments": {"x": 1}}
{"name": "", "arguments": {}}
{"name": "also_ok"}
[/ACTION]
"""
    calls = parse_action_block(text)
    assert [c.tentacle for c in calls] == ["ok", "also_ok"]


def test_arguments_default_to_empty_dict():
    text = """[ACTION]
{"name": "minimal"}
[/ACTION]
"""
    calls = parse_action_block(text)
    assert calls[0].params == {}


def test_arguments_must_be_object_else_empty():
    """Defensive: if `arguments` is a non-object (string/list/null),
    treat as empty dict so the tentacle doesn't blow up on bad type."""
    text = """[ACTION]
{"name": "weird", "arguments": "not a dict"}
[/ACTION]
"""
    calls = parse_action_block(text)
    assert calls[0].params == {}


def test_unicode_arguments():
    text = """[ACTION]
{"name": "say", "arguments": {"text": "你好世界"}}
[/ACTION]
"""
    calls = parse_action_block(text)
    assert calls[0].params == {"text": "你好世界"}


def test_intent_synthesized_from_arg_keys():
    """The intent string drives /dispatch event display. Should be a
    compact label, not the full arg blob (some tentacles take huge
    payloads)."""
    big_source = "x" * 5000
    text = (
        '[ACTION]\n'
        '{"name": "code_run", "arguments": '
        '{"language": "python", "source": "' + big_source + '"}}\n'
        '[/ACTION]\n'
    )
    calls = parse_action_block(text)
    # Intent ≈ "code_run(language, source)" — short, no full source dump
    assert "code_run" in calls[0].intent
    assert "language" in calls[0].intent
    # Must not include the giant source blob
    assert len(calls[0].intent) < 200


def test_action_block_inside_decision_section():
    """Realistic scenario: Self writes [ACTION]...[/ACTION] nested
    inside [DECISION]. Parser shouldn't care about which tag section
    it lives in — only about the [ACTION]...[/ACTION] sentinels."""
    text = """[THINKING]
weighing options.
[DECISION]
I should reply and then search.
[ACTION]
{"name": "web_chat_reply", "arguments": {"text": "checking"}}
{"name": "search", "arguments": {"query": "X"}}
[/ACTION]
[NOTE]
will check in next beat
[HIBERNATE]
30
"""
    calls = parse_action_block(text)
    assert [c.tentacle for c in calls] == ["web_chat_reply", "search"]
