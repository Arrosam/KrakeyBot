"""Tool-call executor — parses <tool_call>...</tool_call> blocks
into TentacleCalls.

This is the default tentacle dispatch path when no decision-translator
Reflect (e.g. the hypothalamus plugin) is registered. Format chosen
for breadth of training coverage in modern open-source LLMs.
"""
from src.runtime.heartbeat.action_executor import parse_tool_calls


def test_empty_input_returns_empty():
    assert parse_tool_calls("") == []
    assert parse_tool_calls(None) == []  # type: ignore[arg-type]


def test_no_tool_call_returns_empty():
    """Self can produce thinking/decision/note without invoking any
    tentacle — the parser should accept that as a valid no-op."""
    text = """[THINKING]
Just reflecting today.
[DECISION]
No action needed.
[NOTE]
Nothing to do.
"""
    assert parse_tool_calls(text) == []


def test_single_call_parsed():
    text = """[DECISION]
Greet the user.
<tool_call>
{"name": "web_chat_reply", "arguments": {"text": "Hi!"}}
</tool_call>
"""
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    c = calls[0]
    assert c.tentacle == "web_chat_reply"
    assert c.params == {"text": "Hi!"}
    assert c.adrenalin is False


def test_multiple_calls_in_separate_blocks():
    """Parallel calls = repeat the tag (one JSON object per tag, not
    one-call-per-line within a single tag)."""
    text = """<tool_call>
{"name": "web_chat_reply", "arguments": {"text": "ok"}}
</tool_call>
<tool_call>
{"name": "search", "arguments": {"query": "weather"}, "adrenalin": true}
</tool_call>
"""
    calls = parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0].tentacle == "web_chat_reply"
    assert calls[1].tentacle == "search"
    assert calls[1].adrenalin is True


def test_blocks_separated_by_prose_still_parse():
    """Self might explain in between two tool calls. Both should parse."""
    text = """<tool_call>
{"name": "first", "arguments": {}}
</tool_call>
some text in between explaining the next call
<tool_call>
{"name": "second", "arguments": {}}
</tool_call>
"""
    calls = parse_tool_calls(text)
    assert [c.tentacle for c in calls] == ["first", "second"]


def test_one_bad_block_skipped_others_parse():
    """A single malformed block must not poison adjacent good ones."""
    text = """<tool_call>
{"name": "good_one", "arguments": {}}
</tool_call>
<tool_call>
this is not json
</tool_call>
<tool_call>
{"name": "good_two", "arguments": {"x": 1}}
</tool_call>
"""
    calls = parse_tool_calls(text)
    assert [c.tentacle for c in calls] == ["good_one", "good_two"]


def test_missing_name_skipped():
    text = """<tool_call>
{"name": "ok", "arguments": {}}
</tool_call>
<tool_call>
{"arguments": {"x": 1}}
</tool_call>
<tool_call>
{"name": "", "arguments": {}}
</tool_call>
<tool_call>
{"name": "also_ok"}
</tool_call>
"""
    calls = parse_tool_calls(text)
    assert [c.tentacle for c in calls] == ["ok", "also_ok"]


def test_arguments_default_to_empty_dict():
    text = """<tool_call>
{"name": "minimal"}
</tool_call>
"""
    calls = parse_tool_calls(text)
    assert calls[0].params == {}


def test_arguments_must_be_object_else_empty():
    """Defensive: if `arguments` is a non-object (string/list/null),
    treat as empty dict so the tentacle doesn't blow up on bad type."""
    text = """<tool_call>
{"name": "weird", "arguments": "not a dict"}
</tool_call>
"""
    calls = parse_tool_calls(text)
    assert calls[0].params == {}


def test_unicode_arguments():
    text = """<tool_call>
{"name": "say", "arguments": {"text": "你好世界"}}
</tool_call>
"""
    calls = parse_tool_calls(text)
    assert calls[0].params == {"text": "你好世界"}


def test_intent_synthesized_from_arg_keys():
    """The intent string drives the dispatch event display. Should be a
    compact label, not the full arg blob (some tentacles take huge
    payloads)."""
    big_source = "x" * 5000
    text = (
        '<tool_call>\n'
        '{"name": "code_run", "arguments": '
        '{"language": "python", "source": "' + big_source + '"}}\n'
        '</tool_call>\n'
    )
    calls = parse_tool_calls(text)
    assert "code_run" in calls[0].intent
    assert "language" in calls[0].intent
    assert len(calls[0].intent) < 200


def test_tag_inside_decision_section():
    """Realistic scenario: Self writes <tool_call> nested inside
    [DECISION]. Parser only cares about the <tool_call> tags."""
    text = """[THINKING]
weighing options.
[DECISION]
I should reply and then search.
<tool_call>
{"name": "web_chat_reply", "arguments": {"text": "checking"}}
</tool_call>
<tool_call>
{"name": "search", "arguments": {"query": "X"}}
</tool_call>
[NOTE]
will check in next beat
[HIBERNATE]
30
"""
    calls = parse_tool_calls(text)
    assert [c.tentacle for c in calls] == ["web_chat_reply", "search"]


def test_back_compat_alias_still_works():
    """parse_action_block was the old name; kept as alias since two
    other modules import it."""
    from src.runtime.heartbeat.action_executor import parse_action_block
    text = '<tool_call>\n{"name": "x"}\n</tool_call>'
    assert parse_action_block(text) == parse_tool_calls(text)
