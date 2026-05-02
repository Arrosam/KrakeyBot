"""Hypothalamus Modifier — LLM-driven [DECISION] → tool-call translator.

Stateless: every call is an independent LLM invocation. Converts
Self's natural-language [DECISION] into structured ``ToolCall``
objects, memory writes/updates, and the sleep flag.

Only loaded by ``src.plugin_system.load_component`` when
``hypothalamus`` is listed in ``config.yaml``'s ``plugins:``.
The contract types (``ToolCall``, ``DecisionResult``) live
in ``src.interfaces.modifier`` so the runtime can consume them without
importing this plugin.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from krakey.interfaces.modifier import DecisionResult, ToolCall
from krakey.llm.resolve import ChatLike

if TYPE_CHECKING:
    from krakey.interfaces.plugin_context import PluginContext


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


class HypothalamusModifierImpl:
    """LLM-driven [DECISION] → tool-call translator.

    Stateless: every ``translate()`` call is an independent
    system+user message pair. The translator LLM is bound at
    construction (resolved from ``ctx.get_llm_for_tag`` in the
    factory) and never mutated.
    """

    name = "hypothalamus"
    role = "hypothalamus"

    def __init__(self, llm: ChatLike):
        self._llm = llm

    def modify_prompt(self, elements) -> None:
        """The hypothalamus owns the [DECISION] → tool-call
        translation, so teaching Self the structured-tag syntax in
        [ACTION FORMAT] would create competing dispatch paths. Drop
        the action_format element."""
        if "action_format" in elements:
            del elements["action_format"]

    async def translate(
        self, decision: str, tools: list[dict[str, Any]],
    ) -> DecisionResult:
        system = SYSTEM_PROMPT.format(tool_list=_format_tools(tools))
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": decision},
        ]
        raw = await self._llm.chat(messages)
        if raw is None or not str(raw).strip():
            raise ValueError(
                "Hypothalamus LLM returned empty content "
                f"(model={getattr(self._llm, 'model', '?')!r}); check the "
                "endpoint, max_tokens, or prompt-size limit."
            )
        data = _parse_json(str(raw))
        return _to_result(data)


def build_modifier(ctx: "PluginContext") -> HypothalamusModifierImpl | None:
    """Factory invoked by ``load_component``.

    Reads its own ``llm_purposes.translator`` binding from
    ``workspace/plugins/hypothalamus/config.yaml`` (surfaced via
    ``ctx.config``), then asks the runtime to resolve that tag to
    a concrete ``LLMClient``. When the binding is missing or the tag
    is unknown, returns ``None`` — the loader skips this Modifier
    rather than crashing the runtime (additive plugin model).
    """
    import logging
    purposes = ctx.config.get("llm_purposes") or {}
    tag_name = (
        purposes.get("translator") if isinstance(purposes, dict) else None
    )
    llm = ctx.get_llm_for_tag(tag_name)
    if llm is None:
        logging.getLogger(__name__).warning(
            "hypothalamus: no LLM resolved for purpose 'translator' "
            "(check workspace/plugins/hypothalamus/config.yaml's "
            "llm_purposes.translator and llm.tags in central config). "
            "Skipping registration."
        )
        return None
    return HypothalamusModifierImpl(llm)


def _format_tools(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return "(none)"
    lines = []
    for t in tools:
        lines.append(f"- {t['name']}: {t['description']} "
                     f"params={t.get('parameters_schema', {})}")
    return "\n".join(lines)


def _parse_json(raw: str) -> dict[str, Any]:
    """Lenient JSON extraction. Tries (in order):
      1. raw.strip() + markdown-fence stripping
      2. outermost {...} block
      3. sanitized version of (2): smart quotes → straight, trailing
         commas removed, single quotes → double (when unambiguous)

    Raises ``json.JSONDecodeError`` if every candidate fails. The
    heartbeat orchestrator catches the exception and pushes an
    adrenalin system_event back to Self instead of crashing — that
    surfaces the parse failure to the agent so it can re-state the
    decision more concretely on the next beat (a silent empty-result
    fallback would hide the problem).
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
    raise json.JSONDecodeError("could not parse Hypothalamus JSON", raw, 0)


def _candidates(text: str, raw: str):
    yield text
    m = _JSON_BLOCK.search(raw)
    if m:
        block = m.group(0)
        yield block
        yield _sanitize(block)


def _sanitize(text: str) -> str:
    """Best-effort fixes for common LLM-produced JSON quirks:
      - smart/curly quotes → straight
      - trailing commas before } or ]
      - single-quoted strings → double-quoted (only when value-safe)
    """
    text = (text.replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'"))
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
    )
