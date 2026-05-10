"""Static prose layers injected by ``PromptBuilder`` (non-DNA).

Two action-format teaching prompts live here, one per DecisionEngine
mode. Both target the same ``action_format`` element so PromptBuilder
+ engines never have to reason about which slot to write to:

  * ``ACTION_FORMAT_LAYER_TOOL_CALL`` (default) — teaches Self the
    ``<tool_call>...</tool_call>`` JSON syntax used by
    ``ToolCallParserDecisionEngine``. Includes worked beat examples.
  * ``ACTION_FORMAT_LAYER_HYPOTHALAMUS`` — natural-language guidance
    used when ``HypothalamusDecisionEngine`` is wired in. Self writes
    free-form ``[DECISION]`` text; the hypothalamus LLM translates.
    Includes worked beat examples in NL flavor.

The default layer (TOOL_CALL) is what PromptBuilder injects unsolicited.
``HypothalamusDecisionEngine.modify_prompt`` overwrites the slot with
the HYPOTHALAMUS layer, so each engine ships with its own teaching
prose + examples and the two never compete.

Also exports ``HEARTBEAT_QUESTION`` — the trailing closer for every
beat.
"""
from __future__ import annotations


ACTION_FORMAT_LAYER_TOOL_CALL = """# [ACTION FORMAT]
This runtime parses tool calls **directly from your `[DECISION]`** —
no translator LLM in between. Wrap a JSON payload inside
`<tool_call>...</tool_call>` tags (one tag per call; repeat the tag
for parallel calls):

<tool_call>
{"name": "<tool_name>", "arguments": {...}}
</tool_call>
<tool_call>
{"name": "<another>", "arguments": {...}, "adrenalin": true}
</tool_call>

Fields:
- `name` (str, required): pick a tool name from `[CAPABILITIES]`
- `arguments` (object, optional): the tool's parameters; omit = `{}`
- `adrenalin` (bool, optional): urgency flag; omit = false. Set true
  only when this action's feedback should interrupt the next idle.

Heartbeats with no tool to call (pure thinking, just leaving a `[NOTE]`,
sleeping) just omit the `<tool_call>` block. `<tool_call>` blocks can
appear inside `[DECISION]` or `[THINKING]`, or after `[DECISION]`; all
parse. A parse failure in one tag does not affect the others.

## Worked beat examples

**1. Reply to user, then idle 60s**

```
[THINKING]
user asked time. answer.

[DECISION]
<tool_call>
{"name": "web_chat_reply", "arguments": {"text": "It's 14:32."}}
</tool_call>

[IDLE] 60
```

**2. Parallel actions, one urgent**

```
[THINKING]
need news + weather. fire both. weather urgent.

[DECISION]
<tool_call>
{"name": "web_search", "arguments": {"query": "krakey ai news today"}}
</tool_call>
<tool_call>
{"name": "weather", "arguments": {"city": "Beijing"}, "adrenalin": true}
</tool_call>

[IDLE] 5
```

**3. Quiet beat — observe, leave a note, sleep long**

```
[THINKING]
nothing in stimulus. low fatigue. wait.

[DECISION]
(no tool call)

[NOTE]
User mentioned a deadline Friday. Watch for follow-up.

[IDLE] 600
```

**4. Enter sleep mode**

Sleep is NOT a tool call in this mode. Output the literal phrase
"enter sleep mode" inside `[DECISION]`; the runtime detects it and
triggers the 7-phase Sleep transition:

```
[THINKING]
fatigue 95. gm full. time to sleep.

[DECISION]
enter sleep mode
```"""


ACTION_FORMAT_LAYER_HYPOTHALAMUS = """# [ACTION FORMAT]
This runtime has a **hypothalamus translator LLM** between you and the
tool dispatcher. Write `[DECISION]` in **natural language** — name the
tool from `[CAPABILITIES]` plus what you want it to do. The
hypothalamus extracts the call, fills params, and sets the adrenalin
flag.

Triggers the hypothalamus reads:
- "remember …" / "record …" / "important: …" → memory write.
- "goal achieved" / "task done" / "completed" → memory update.
- "quick" / "urgent" / "someone is waiting" → adrenalin = true.
- "no action" / silence inside `[DECISION]` → empty dispatch.
- "enter sleep mode" (exact phrasing) → triggers full Sleep. Softer
  words like "rest" / "pause" / "wait" do NOT enter Sleep — use
  `[IDLE] N` for those.

## Worked beat examples

**1. Reply to user, then idle 60s**

```
[THINKING]
user asked time. answer.

[DECISION]
Use web_chat_reply to tell the user it's 14:32.

[IDLE] 60
```

**2. Parallel actions, one urgent**

```
[THINKING]
need news + weather. fire both. weather urgent.

[DECISION]
Search the web for "krakey ai news today" AND quickly check the
weather in Beijing — surface the weather next beat.

[IDLE] 5
```

**3. Quiet beat — observe, leave a note, sleep long**

```
[THINKING]
nothing in stimulus. low fatigue. wait.

[DECISION]
No action.

[NOTE]
User mentioned a deadline Friday. Watch for follow-up.

[IDLE] 600
```

**4. Enter sleep mode**

```
[THINKING]
fatigue 95. gm full. time to sleep.

[DECISION]
enter sleep mode
```"""


HEARTBEAT_QUESTION = (
    "# [HEARTBEAT]\n"
    "What do you notice? What matters? What do you do?\n"
    "Respond using [THINKING] / [DECISION] / [NOTE] / [IDLE]."
)
