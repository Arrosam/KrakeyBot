"""Tool-call executor — parses <tool_call>...</tool_call> blocks
into ToolCalls.

This is the default tool dispatch path when no decision-translator
Modifier (e.g. the hypothalamus plugin) is registered. Format chosen
for breadth of training coverage in modern open-source LLMs.
"""
from krakey.engines.decision._internal.action_executor import (
    parse_tool_calls_with_failures,
)


def parse_tool_calls(text):
    """Test helper — wraps the engine's
    ``parse_tool_calls_with_failures`` and discards the failure list.
    Most action-executor tests only care about the successfully-parsed
    calls; the salvage-failure tests below call
    ``parse_tool_calls_with_failures`` directly."""
    calls, _ = parse_tool_calls_with_failures(text)
    return calls


def test_empty_input_returns_empty():
    assert parse_tool_calls("") == []
    assert parse_tool_calls(None) == []  # type: ignore[arg-type]


def test_no_tool_call_returns_empty():
    """Self can produce thinking/decision/note without invoking any
    tool — the parser should accept that as a valid no-op."""
    text = """[THINKING]
Just thinking today.
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
    assert c.tool == "web_chat_reply"
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
    assert calls[0].tool == "web_chat_reply"
    assert calls[1].tool == "search"
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
    assert [c.tool for c in calls] == ["first", "second"]


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
    assert [c.tool for c in calls] == ["good_one", "good_two"]


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
    assert [c.tool for c in calls] == ["ok", "also_ok"]


def test_arguments_default_to_empty_dict():
    text = """<tool_call>
{"name": "minimal"}
</tool_call>
"""
    calls = parse_tool_calls(text)
    assert calls[0].params == {}


def test_arguments_must_be_object_else_empty():
    """Defensive: if `arguments` is a non-object (string/list/null),
    treat as empty dict so the tool doesn't blow up on bad type."""
    text = """<tool_call>
{"name": "weird", "arguments": "not a dict"}
</tool_call>
"""
    calls = parse_tool_calls(text)
    assert calls[0].params == {}


def test_unicode_arguments():
    text = """<tool_call>
{"name": "say", "arguments": {"text": "héllo wörld"}}
</tool_call>
"""
    calls = parse_tool_calls(text)
    assert calls[0].params == {"text": "héllo wörld"}


def test_intent_synthesized_from_arg_keys():
    """The intent string drives the dispatch event display. Should be a
    compact label, not the full arg blob (some tools take huge
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
[IDLE]
30
"""
    calls = parse_tool_calls(text)
    assert [c.tool for c in calls] == ["web_chat_reply", "search"]


# ---------------- parse_tool_calls_with_failures ----------------


def test_with_failures_empty_input_returns_two_empty_lists():
    from krakey.engines.decision._internal.action_executor import (
        parse_tool_calls_with_failures,
    )
    calls, failures = parse_tool_calls_with_failures("")
    assert calls == [] and failures == []


def test_with_failures_arg_value_tail_salvaged_with_diagnostic():
    """Real-world Self drift: each <tool_call> ends with a stray
    </arg_value> closing tag bleeding from a different format the
    model was fine-tuned on. JSON parse hits 'Extra data', the
    salvage path truncates and recovers, and a ParseFailure
    surfaces alongside the dispatched ToolCall so Self gets
    corrective feedback without losing forward progress."""
    from krakey.engines.decision._internal.action_executor import (
        parse_tool_calls_with_failures,
    )
    raw = (
        '<tool_call>{"name": "search", "arguments": {"query": "X"}}'
        '</arg_value></tool_call>'
    )
    calls, failures = parse_tool_calls_with_failures(raw)
    # Salvaged: ToolCall dispatches AND failure surfaces.
    assert [c.tool for c in calls] == ["search"]
    assert calls[0].params == {"query": "X"}
    assert len(failures) == 1
    f = failures[0]
    assert f.block_index == 0
    assert f.salvaged is True
    assert "</arg_value>" in f.error  # trailing bytes echoed in error
    assert "trailing data" in f.error.lower()


def test_with_failures_partial_success_still_dispatches_clean_blocks():
    """One </arg_value>-tail block + one clean block in the same
    response → 2 ToolCalls returned (one salvaged, one clean),
    1 ParseFailure recorded for the salvaged block."""
    from krakey.engines.decision._internal.action_executor import (
        parse_tool_calls_with_failures,
    )
    raw = (
        '<tool_call>{"name": "bad", "arguments": {}}</arg_value></tool_call>\n'
        '<tool_call>{"name": "good", "arguments": {"q": "y"}}</tool_call>'
    )
    calls, failures = parse_tool_calls_with_failures(raw)
    assert [c.tool for c in calls] == ["bad", "good"]
    assert len(failures) == 1
    assert failures[0].block_index == 0
    assert failures[0].salvaged is True


def test_with_failures_records_block_index_for_each():
    """When several blocks fail in a row, block_index lets the
    corrective stimulus show Self exactly which positions had
    drift. With salvage, all three calls dispatch AND all three
    failures surface."""
    from krakey.engines.decision._internal.action_executor import (
        parse_tool_calls_with_failures,
    )
    raw = (
        '<tool_call>{"name": "a"}</arg_value></tool_call>\n'
        '<tool_call>{"name": "b"}</arg_value></tool_call>\n'
        '<tool_call>{"name": "c"}</arg_value></tool_call>'
    )
    calls, failures = parse_tool_calls_with_failures(raw)
    assert [c.tool for c in calls] == ["a", "b", "c"]
    assert [f.block_index for f in failures] == [0, 1, 2]
    assert all(f.salvaged for f in failures)


def test_with_failures_invisible_chars_in_repr_payload():
    """Zero-width space and other invisible characters after the
    JSON object trigger 'Extra data'. The payload field of the
    salvage ParseFailure uses repr() so Self can SEE the
    invisible char as an escape sequence — fixing the
    "I can't tell what's wrong" 120+-beat loop."""
    from krakey.engines.decision._internal.action_executor import (
        parse_tool_calls_with_failures,
    )
    raw = '<tool_call>{"name": "X", "arguments": {}}​</tool_call>'
    calls, failures = parse_tool_calls_with_failures(raw)
    assert [c.tool for c in calls] == ["X"]
    assert len(failures) == 1
    f = failures[0]
    assert f.salvaged is True
    # The payload is repr()'d — invisible char now visible.
    assert "\\u200b" in f.payload or "\\u200b" in f.error


def test_with_failures_non_json_object_payload_recorded():
    """A JSON array (or other valid JSON that isn't an object) is
    a different failure shape from JSON-decode error — surface the
    diagnosis rather than masking it as 'unparseable'."""
    from krakey.engines.decision._internal.action_executor import (
        parse_tool_calls_with_failures,
    )
    raw = '<tool_call>[1, 2, 3]</tool_call>'
    calls, failures = parse_tool_calls_with_failures(raw)
    assert calls == []
    assert len(failures) == 1
    assert "not an object" in failures[0].error.lower()


def test_with_failures_missing_name_recorded():
    from krakey.engines.decision._internal.action_executor import (
        parse_tool_calls_with_failures,
    )
    raw = '<tool_call>{"arguments": {"x": 1}}</tool_call>'
    calls, failures = parse_tool_calls_with_failures(raw)
    assert calls == []
    assert len(failures) == 1
    assert "name" in failures[0].error.lower()


def test_with_failures_empty_block_does_not_count_as_failure():
    """An empty <tool_call></tool_call> tag is noise, not a parse
    failure — don't burn a corrective stimulus on it."""
    from krakey.engines.decision._internal.action_executor import (
        parse_tool_calls_with_failures,
    )
    calls, failures = parse_tool_calls_with_failures(
        '<tool_call></tool_call>'
    )
    assert calls == [] and failures == []


# =====================================================================
# Synthesized intent — value previews so back-to-back calls of the
# same tool are distinguishable in the dashboard's tool-usage panel
# =====================================================================


def test_synth_intent_shows_string_value_previews():
    """Strings render verbatim (truncated to a per-arg cap), so a
    cli_exec invocation reads ``cli_exec(env='local', cmd=...)``
    instead of the previous useless ``cli_exec(env, cmd, ...)``."""
    text = (
        '<tool_call>{"name":"cli_exec","arguments":'
        '{"env":"local","cmd":["python","--version"]}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    intent = calls[0].intent
    assert "env=" in intent
    assert "'local'" in intent
    assert "cmd=" in intent
    # The cmd list previewed verbatim because it's all primitives.
    assert "python" in intent
    assert "--version" in intent


def test_synth_intent_truncates_giant_values():
    """Per-arg + total length caps so a tool taking a 5KB blob
    doesn't spam the dashboard."""
    big = "x" * 5000
    text = (
        '<tool_call>{"name":"code_run","arguments":'
        '{"language":"python","source":"' + big + '"}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    intent = calls[0].intent
    # Total intent stays under the documented total cap (currently
    # 120 + tool name + parens, but the cap pins it at 120 inner +
    # bookkeeping; just verify it stays well under 200).
    assert len(intent) < 200
    # Truncation marker present somewhere in the rendered intent.
    assert "..." in intent


def test_synth_intent_renders_primitive_lists():
    text = (
        '<tool_call>{"name":"foo","arguments":'
        '{"items":["a","b","c"]}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    intent = calls[0].intent
    assert "items=" in intent
    assert "['a', 'b', 'c']" in intent


def test_synth_intent_collapses_nested_objects():
    """Lists of dicts (or other non-primitive structures) render
    as ``key=...`` so the line stays compact. Full structure is
    available on DispatchEvent.params for callers that need it."""
    text = (
        '<tool_call>{"name":"foo","arguments":'
        '{"big":{"nested":{"deep":1}},"small":42}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    intent = calls[0].intent
    assert "big=..." in intent
    assert "small=42" in intent


def test_synth_intent_with_no_args_returns_just_name():
    text = '<tool_call>{"name":"sleep"}</tool_call>'
    calls = parse_tool_calls(text)
    assert calls[0].intent == "sleep"
