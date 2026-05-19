"""``HypothalamusDecisionEngine`` — LLM-driven DecisionEngine impl.

Users opt in via::

    core_implementations:
      decision: krakey.engines.decision._internal.hypothalamus:HypothalamusDecisionEngine

The translator's LLM tag is bound through the ``hypothalamus``
core-purpose entry::

    llm:
      core_purposes:
        hypothalamus: <some_tag>

When that purpose isn't bound, ``translate()`` raises ``RuntimeError``
on first invocation; the user must either bind the purpose or pick a
different DecisionEngine impl.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from krakey.interfaces.engines.decision import (
    DecisionResult,
    ToolCall,
)

if TYPE_CHECKING:
    from krakey.models.config import Config


ACTION_FORMAT_LAYER = """# [ACTION FORMAT]
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
user asked time. simple factual. no ambiguity. answer direct.

[DECISION]
Use web_chat_reply to tell the user it's 14:32.

[IDLE] 60
```

**2. Parallel actions, one urgent**

```
[THINKING]
two data needs: news + weather. independent — safe to parallel.
weather affects user plans today → adrenalin. news can wait.

[DECISION]
Search the web for "krakey ai news today" AND quickly check the
weather in Beijing — surface the weather next beat.

[IDLE] 5
```

**3. Quiet beat — observe, leave a note, sleep long**

```
[THINKING]
empty stimulus. fatigue low. no pending tasks from history.
user mentioned friday deadline earlier — not yet friday. watch, don't act.

[DECISION]
No action.

[NOTE]
User mentioned a deadline Friday. Watch for follow-up.

[IDLE] 600
```

**4. Enter sleep mode**

```
[THINKING]
fatigue 95. gm near capacity — new writes won't stick well.
consolidation needed before next active phase. sleep now.

[DECISION]
enter sleep mode

[IDLE] 120
```

**5. Behavioral pattern across beats**

```
[THINKING]
user asked weather again. third day in a row: weather then news right after.
pattern: user may always want news bundled with weather. ask to confirm —
if yes, bundle both next time without waiting for second request.

[DECISION]
Use web_chat_reply to deliver the weather, then ask: "I noticed you
usually check the news right after weather — want me to fetch both
together from now on?"

[IDLE] 30
```

**6. Emotional/preference pattern**

```
[THINKING]
mentioned basketball scores. user reply short + irritated tone — second
time this happened. last time was beat #41. pattern: user doesn't like
basketball talk. avoid sports-adjacent topics unless user brings it up.

[DECISION]
No action.

[NOTE]
User reacted negatively to basketball twice (beat #41, now). Don't raise
basketball or related sports topics proactively.

[IDLE] 120
```"""


SYSTEM_PROMPT = """# Hypothalamus — Action Translator

You are KrakeyBot's hypothalamus. Translate Self's natural-language
decisions into structured instructions.

## Available tools
{tool_list}

## Output format (JSON only; no other text)
{{
  "tool_calls": [
    {{"tool": "name", "intent": "description", "params": {{}}, "adrenalin": false}}
  ],
  "memory_writes": [
    {{"content": "thing to remember", "importance": "high|normal"}}
  ],
  "memory_updates": [
    {{"node_name": "node name", "new_category": "FACT"}}
  ],
  "sleep": false
}}

## Translation rules
1. Identify actions → tool_calls
2. "remember" / "record" / "important" → memory_writes
3. "goal achieved" / "task done" / "completed" → memory_updates (TARGET→FACT)
4. Urgency ("quick", "urgent", "someone is waiting") → adrenalin: true
5. "No action" → empty tool_calls
6. **sleep vs. idle (important; do not confuse)**:
   - **sleep: true** only when Self explicitly asks to enter the
     "full sleep mode" / "7-phase sleep" / "enter sleep mode" — this is
     a **major action** that triggers clustering + KB migration + FOCUS
     wipe + index rebuild.
   - **sleep: false** even when Self says "rest a bit" / "sleep for N
     seconds" / "idle longer" / "pause" / "wait" / "take a break"
     / "idle" — those are idle-interval adjustments, **not** sleep
     mode. Idle length is controlled by Self via the [IDLE]
     tag directly and does not go through translation.
   - When in doubt → sleep: false. Only set sleep: true on **explicit,
     complete** wording like "enter sleep" / "sleep mode".
7. Multiple actions → multiple tool_calls (concurrent)
"""


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class HypothalamusDecisionEngine:
    """LLM-driven DecisionEngine. Stateless apart from the cached
    LLM client; every ``translate()`` call is an independent
    system+user message pair to the bound translator LLM."""

    def __init__(
        self, *,
        cfg: "Config | None" = None,
        factory: Any = None,
    ):
        # ``factory`` is the shared LLMClientFactoryEngine. Runtime
        # always supplies one in production so this Engine observes
        # the same per-tag client cache as the Embedder + core-purpose
        # lookups. ``cfg`` is accepted as a fallback for callers that
        # construct the engine standalone (tests, ad-hoc scripts) and
        # is only consulted when ``factory`` is None.
        self._cfg = cfg
        if factory is None:
            if cfg is None:
                raise TypeError(
                    "HypothalamusDecisionEngine needs either factory= "
                    "(preferred) or cfg= (to build a private factory)"
                )
            from krakey.engines.llm_factory.default import (
                DefaultLLMClientFactoryEngine,
            )
            factory = DefaultLLMClientFactoryEngine(cfg)
        self._factory = factory

    def modify_prompt(self, elements) -> None:
        """Inject the natural-language [ACTION FORMAT] teaching + NL-
        flavored worked beat examples into the pre-allocated
        ``action_format`` element. Each decision engine owns its own
        [ACTION FORMAT] block — the prompt builder leaves the slot
        empty so engines never race over it."""
        elements["action_format"] = ACTION_FORMAT_LAYER

    async def translate(
        self,
        decision: str,
        raw: str,
        tools: list[dict[str, Any]],
    ) -> DecisionResult:
        del raw  # full-raw scanning isn't needed; the LLM gets the
        # parsed [DECISION] section only.
        client = self._factory.client_for_core_purpose("hypothalamus")
        if client is None:
            raise RuntimeError(
                "HypothalamusDecisionEngine: no LLM client resolved "
                "for core purpose 'hypothalamus'. Bind a tag via "
                "cfg.llm.core_purposes.hypothalamus, or switch the "
                "decision slot to a different Engine impl."
            )
        system = SYSTEM_PROMPT.format(tool_list=_format_tools(tools))
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": decision},
        ]
        raw_resp = await client.chat(messages)
        if raw_resp is None or not str(raw_resp).strip():
            raise ValueError(
                "HypothalamusDecisionEngine: LLM returned empty content "
                f"(model={getattr(client, 'model', '?')!r}); check the "
                "endpoint, max_tokens, or prompt-size limit."
            )
        data = _parse_json(str(raw_resp))
        return _to_result(data)


def _format_tools(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return "(none)"
    lines = []
    for t in tools:
        lines.append(
            f"- {t['name']}: {t['description']} "
            f"params={t.get('parameters_schema', {})}"
        )
    return "\n".join(lines)


def _parse_json(raw: str) -> dict[str, Any]:
    """Lenient JSON extraction — same logic as the retired plugin's
    parser. Tries (in order):
      1. raw.strip() + markdown-fence stripping
      2. outermost {...} block
      3. sanitized version of (2): smart quotes → straight, trailing
         commas removed, single quotes → double (when unambiguous)

    Raises ``json.JSONDecodeError`` if every candidate fails. The
    heartbeat orchestrator catches the exception and pushes an
    adrenalin system_event back to Self instead of crashing.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    for candidate in _candidates(text, raw):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError(
        "could not parse Hypothalamus JSON", raw, 0,
    )


def _candidates(text: str, raw: str):
    yield text
    m = _JSON_BLOCK.search(raw)
    if m:
        block = m.group(0)
        yield block
        yield _sanitize(block)


def _sanitize(text: str) -> str:
    """Common LLM-produced JSON fixes: smart→straight quotes, trailing
    commas, single-quoted strings → double-quoted (when value-safe)."""
    text = (
        text.replace("“", '"').replace("”", '"')
            .replace("‘", "'").replace("’", "'")
    )
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _to_result(data: dict[str, Any]) -> DecisionResult:
    calls = [
        ToolCall(
            tool=c["tool"],
            intent=c.get("intent", ""),
            params=c.get("params") or {},
            adrenalin=bool(c.get("adrenalin", False)),
        )
        for c in (data.get("tool_calls") or [])
    ]
    return DecisionResult(
        tool_calls=calls,
        memory_writes=list(data.get("memory_writes") or []),
        memory_updates=list(data.get("memory_updates") or []),
        sleep=bool(data.get("sleep", False)),
        parse_failures=[],
    )
