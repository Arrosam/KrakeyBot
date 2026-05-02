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
When you want to call tools, wrap a JSON payload inside
<tool_call>...</tool_call> tags (one tag per call; repeat the tag for
multiple concurrent calls):

<tool_call>
{"name": "<tool_name>", "arguments": {...}}
</tool_call>
<tool_call>
{"name": "<another>", "arguments": {...}, "adrenalin": true}
</tool_call>

Fields:
- name (str, required): pick a tool name from [CAPABILITIES]
- arguments (object, optional): the tool's parameters; omit = {} (empty)
- adrenalin (bool, optional): urgency flag; omit = false. Set true only
  when you want this action's feedback to interrupt the next idle.

Heartbeats with no tool to call (e.g. pure thinking or writing [NOTE])
just omit the tool_call block. tool_call blocks can appear inside
[DECISION] or [THINKING], or after [DECISION]; all of them get parsed.
A parse failure in one tag does not affect the others."""


HEARTBEAT_QUESTION = (
    "# [HEARTBEAT]\n"
    "What do you notice? What matters? What do you do?\n"
    "Respond using [THINKING] / [DECISION] / [NOTE] / [IDLE]."
)
