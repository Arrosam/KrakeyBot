"""``HypothalamusDecisionEngine`` — LLM-driven DecisionEngine impl.

Lifted from the in-tree ``hypothalamus`` plugin (Engine refactor 2026-05).
The LLM-translator behavior — turning Self's natural-language [DECISION]
into structured ``ToolCall``s + memory writes/updates + sleep flag —
is now an Engine slot impl rather than a Modifier-role plugin.

Users opt in via::

    core_implementations:
      decision: krakey.engines.decision.hypothalamus:HypothalamusDecisionEngine

The translator's LLM tag is bound through the ``hypothalamus``
core-purpose entry::

    llm:
      core_purposes:
        hypothalamus: <some_tag>

When that purpose isn't bound, ``translate()`` raises ``RuntimeError``
on first invocation. This is the equivalent of the plugin's old
"factory returned None and the loader skipped the modifier" path
— previously the runtime fell back to the script parser; now,
under Engine-only dispatch, the user must either bind the
purpose or pick a different DecisionEngine impl.

Migration note: during the transitional period (steps 7b → 12)
this Engine constructs its OWN ``DefaultLLMClientFactoryEngine``
from ``cfg`` rather than receiving the runtime's central factory.
Consequence: Hypothalamus's LLM client is cached separately from
the embedder / reranker / core-purpose clients held by the runtime
factory, so two clients for the same tag can coexist. The unified
factory injection lands in step 12 once every Engine takes its
shared dependencies through the EngineRegistry constructor's
kwargs uniformly.
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

    def __init__(self, *, cfg: "Config | None" = None):
        # ``cfg`` is the only constructor input. The Engine builds
        # its own LLMClientFactoryEngine internally during the
        # migration window — see module docstring for the duplicate-
        # cache caveat that step 12 resolves.
        if cfg is None:
            raise TypeError(
                "HypothalamusDecisionEngine requires cfg= kwarg "
                "(the runtime passes it during EngineRegistry resolution)"
            )
        from krakey.engines.llm_factory.default import (
            DefaultLLMClientFactoryEngine,
        )
        self._cfg = cfg
        self._factory = DefaultLLMClientFactoryEngine(cfg)

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
