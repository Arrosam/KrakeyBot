"""``ToolCallParserDecisionEngine`` — scripted ``<tool_call>`` parser.

Default DecisionEngine impl. Wraps the existing
``parse_tool_calls_with_failures`` util (in
``krakey.engines.decision.action_executor``) so the same per-block
salvage + ParseFailure surfacing the heartbeat already relies on
keeps working — only the call-site pattern changes (Engine slot
instead of an inline module-fn import).

Scope: turns parsed-decision text into a ``DecisionResult`` containing
``tool_calls`` + ``parse_failures``. ``memory_writes`` / ``memory_updates``
/ ``sleep`` are always empty in this impl — they're the responsibility
of an LLM-based translator like ``HypothalamusDecisionEngine`` that can
extract those signals from Self's free-form text. The default Engine
parses tags only; richer extraction is opt-in.

Ownership of Self's [ACTION FORMAT] block lives here too. The engine's
``modify_prompt`` hook injects ``ACTION_FORMAT_LAYER`` (the prose +
worked beat examples that teach the ``<tool_call>`` JSON syntax) into
the ``action_format`` element. The runtime's prompt builder leaves
that slot empty; whichever decision engine is wired in fills it. This
keeps prompt-builder code engine-agnostic — it never imports any
specific decision engine's prose.

Note on input scoping (preserved from the original orchestrator
behavior, see commit bb752c8): the parser scans ONLY the
``decision_text`` (i.e. the ``[DECISION]`` section), NOT the full raw
response. Self's ``[NOTE]`` and ``[THINKING]`` sections sometimes
contain quoted format examples for self-correction; scanning the full
raw response would re-parse those examples as real tool calls and
re-fail on the same drift signature. The Engine receives ``raw`` as a
parameter so future impls can scan it if they want, but the default
deliberately doesn't.
"""
from __future__ import annotations

from typing import Any

from krakey.interfaces.engines.decision import DecisionResult
from krakey.engines.decision.action_executor import (
    parse_tool_calls_with_failures,
)


ACTION_FORMAT_LAYER = """# [ACTION FORMAT]
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

`sleep` is a built-in tool — call it like any other. The runtime
intercepts the dispatch and runs the full 7-phase Sleep cycle at
the end of this beat:

```
[THINKING]
fatigue 95. gm full. time to sleep.

[DECISION]
<tool_call>
{"name": "sleep"}
</tool_call>
```"""


class ToolCallParserDecisionEngine:
    """Default DecisionEngine — scripts tool_call tag parsing.

    Stateless. Accepts ``cfg`` + ``factory`` kwargs for signature
    uniformity with other DecisionEngine impls
    (HypothalamusDecisionEngine needs them); the script parser
    ignores both.
    """

    def __init__(self, *, cfg=None, factory=None):
        del cfg, factory

    def modify_prompt(self, elements) -> None:
        """Inject the ``<tool_call>`` syntax + worked beat examples
        into the pre-allocated ``action_format`` element. The runtime
        invokes this hook before any plugin Modifier's modify_prompt
        runs (see ``orchestrator.assemble_prompt``), so the prose is
        in place by the time anyone else looks at the prompt."""
        elements["action_format"] = ACTION_FORMAT_LAYER

    async def translate(
        self,
        decision: str,
        raw: str,
        tools: list[dict[str, Any]],
    ) -> DecisionResult:
        # ``tools`` is unused by the script parser (it doesn't
        # validate names against the registry — that's the
        # dispatcher's job, which produces an "Unknown tool" stimulus
        # on a name miss). Argument kept for Protocol compatibility
        # so swap-in alternative impls (e.g. an LLM translator) can
        # use the live tool list without a separate Protocol.
        del raw, tools

        tool_calls, parse_failures = parse_tool_calls_with_failures(
            decision,
        )
        return DecisionResult(
            tool_calls=tool_calls,
            memory_writes=[],
            memory_updates=[],
            sleep=False,
            parse_failures=parse_failures,
        )
