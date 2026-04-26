"""Static prose layers injected by ``PromptBuilder`` (non-DNA).

These are constant text the builder splices into specific positions
of the assembled prompt:

  * ``ACTION_FORMAT_LAYER`` — teaches Self the ``[ACTION]...[/ACTION]``
    JSONL syntax. Suppressed when a hypothalamus Reflect is registered
    (the translator owns dispatch and the structured-tag teaching
    would create an interpretive conflict).
  * ``HEARTBEAT_QUESTION``  — the trailing prompt at the end of every
    beat: "What do you notice? What matters? What do you do?"

DNA itself lives in ``dna.py`` because it's an order-of-magnitude
larger than these and gets edited far more often (Self's operating
manual). The two short layers here are stable scaffolding; the
builder's call sites read clearer when they're imported by name
than when inlined as multi-line literals inside ``builder.py``.
"""
from __future__ import annotations


ACTION_FORMAT_LAYER = """# [ACTION FORMAT]
当你想调用 tentacles 时, 在你的回复里写一段 [ACTION]...[/ACTION] 块,
块内每行写一个 JSON 对象 (OpenAI tool_calls 风格):

[ACTION]
{"name": "<tentacle_name>", "arguments": {...}}
{"name": "<another>", "arguments": {...}, "adrenalin": true}
[/ACTION]

字段:
- name (str, 必需): 从 [CAPABILITIES] 里挑一个 tentacle 名字
- arguments (object, 可选): 该 tentacle 的参数, 不传 = 空对象 {}
- adrenalin (bool, 可选): 紧急标志, 不传 = false; 只在你想要这次动作的
  反馈打断后续 hibernate 时设 true

不需要调用 tentacle 的心跳 (例: 只是思考 / 写 [NOTE]) 直接省略 [ACTION] 块。
[ACTION] 块可以出现在 [DECISION] / [THINKING] 里, 也可以出现在 [DECISION]
之后, 都会被解析。一行解析失败不影响其它行。"""


HEARTBEAT_QUESTION = (
    "# [HEARTBEAT]\n"
    "What do you notice? What matters? What do you do?\n"
    "Respond using [THINKING] / [DECISION] / [NOTE] / [HIBERNATE]."
)
