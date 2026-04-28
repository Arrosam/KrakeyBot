"""Static prose layers injected by ``PromptBuilder`` (non-DNA).

  * ``ACTION_FORMAT_LAYER`` — teaches Self the
    ``<tool_call>...</tool_call>`` syntax (Hermes / Qwen format —
    natively trained-on by most modern open-source models). Removed
    from the prompt by the hypothalamus plugin's ``modify_prompt``
    when that translator is registered (it owns dispatch and would
    otherwise compete with this teaching layer).
  * ``HEARTBEAT_QUESTION``  — the trailing prompt at the end of every
    beat.
"""
from __future__ import annotations


ACTION_FORMAT_LAYER = """# [ACTION FORMAT]
当你想调用 tools 时, 用 <tool_call>...</tool_call> 标签包住一段 JSON
(每个 tag 包一个 call; 多个并发调用就重复 tag):

<tool_call>
{"name": "<tool_name>", "arguments": {...}}
</tool_call>
<tool_call>
{"name": "<another>", "arguments": {...}, "adrenalin": true}
</tool_call>

字段:
- name (str, 必需): 从 [CAPABILITIES] 里挑一个 tool 名字
- arguments (object, 可选): 该 tool 的参数, 不传 = 空对象 {}
- adrenalin (bool, 可选): 紧急标志, 不传 = false; 只在你想要这次动作的
  反馈打断后续 hibernate 时设 true

不需要调用 tool 的心跳 (例: 只是思考 / 写 [NOTE]) 直接省略 tool_call 块。
tool_call 块可以出现在 [DECISION] / [THINKING] 里, 也可以出现在 [DECISION]
之后, 都会被解析。一个 tag 解析失败不影响其它 tag。"""


HEARTBEAT_QUESTION = (
    "# [HEARTBEAT]\n"
    "What do you notice? What matters? What do you do?\n"
    "Respond using [THINKING] / [DECISION] / [NOTE] / [HIBERNATE]."
)
