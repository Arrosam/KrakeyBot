"""Default Hypothalamus Reflect — LLM-driven [DECISION] → tentacle-call
translator.

Stateless: every call is an independent LLM invocation. Converts
Self's natural-language [DECISION] into structured ``TentacleCall``
objects, memory writes/updates, and the sleep flag.

Only loaded by ``src.plugin_system.load_component`` when
``default_hypothalamus`` is listed in ``config.yaml``'s ``plugins:``.
The contract types (``TentacleCall``, ``DecisionResult``) live
in ``src.interfaces.reflect`` so the runtime can consume them without
importing this plugin.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Protocol

from src.interfaces.reflect import DecisionResult, TentacleCall

if TYPE_CHECKING:
    from src.interfaces.plugin_context import PluginContext


SYSTEM_PROMPT = """# Hypothalamus — 行动翻译器

你是 KrakeyBot 的下丘脑。将 Self 的自然语言决策翻译为结构化指令。

## 可用 Tentacles
{tentacle_list}

## 输出格式 (JSON, 只输出 JSON, 无其他文字)
{{
  "tentacle_calls": [
    {{"tentacle": "name", "intent": "描述", "params": {{}}, "adrenalin": false}}
  ],
  "memory_writes": [
    {{"content": "要记住的内容", "importance": "high|normal"}}
  ],
  "memory_updates": [
    {{"node_name": "节点名", "new_category": "FACT"}}
  ],
  "sleep": false
}}

## 翻译规则
1. 识别行动 → tentacle_calls
2. "记住"/"记录"/"重要" → memory_writes
3. "目标完成"/"任务结束"/"已完成" → memory_updates (TARGET→FACT)
4. 紧迫感 ("快"/"急"/"有人在等") → adrenalin: true
5. "无行动"/"No action" → 空 tentacle_calls
6. **sleep 与 hibernate 的区分 (重要, 不要混淆)**:
   - **sleep: true** 仅当 Self 明确表达要进入"完整睡眠模式" /
     "7-phase sleep" / "enter sleep mode" 这种**重大动作**时
     (会触发: 聚类 + KB 迁移 + FOCUS 清理 + Index 重建).
   - **sleep: false** 即使 Self 说 "rest a bit" / "休息片刻" / "睡 N 秒" /
     "hibernate longer" / "pause" / "wait" / "take a break" / "idle" —
     这些都是 hibernate 间隔调节, **不是** sleep 模式.
     Hibernate 长度由 Self 用 [HIBERNATE] tag 直接控制, 不经过你翻译.
   - 如有疑问 → sleep: false. 只有看到"进入睡眠 / 睡眠模式 / sleep mode" 这类
     **明确、完整**的措辞才 sleep: true.
7. 多个行动 → 多个 tentacle_calls (并发)
"""


class ChatLike(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class DefaultHypothalamusReflect:
    """LLM-driven decision → tentacle-call translator.

    Was previously a thin wrapper around an inner ``Hypothalamus``
    class living in ``src/hypothalamus.py``; folded together when
    that core leftover was removed (the Reflect IS the translator —
    no third party uses the inner class).
    """

    name = "default_hypothalamus"
    role = "hypothalamus"

    def __init__(self, llm: ChatLike):
        self._llm = llm

    def modify_prompt(self, elements) -> None:
        """The hypothalamus owns the [DECISION] → tentacle-call
        translation, so teaching Self the structured-tag syntax in
        [ACTION FORMAT] would create competing dispatch paths. Drop
        the action_format element."""
        if "action_format" in elements:
            del elements["action_format"]

    async def translate(
        self, decision: str, tentacles: list[dict[str, Any]],
    ) -> DecisionResult:
        system = SYSTEM_PROMPT.format(tentacle_list=_format_tentacles(tentacles))
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


def build_reflect(ctx: "PluginContext") -> DefaultHypothalamusReflect | None:
    """Factory invoked by ``load_component``.

    Reads its own ``llm_purposes.translator`` binding from
    ``workspace/plugins/default_hypothalamus/config.yaml`` (surfaced
    via ``ctx.config``), then asks the runtime to resolve that tag to
    a concrete ``LLMClient``. When the binding is missing or the tag
    is unknown, returns ``None`` — the loader skips this Reflect
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
            "default_hypothalamus: no LLM resolved for purpose 'translator' "
            "(check workspace/plugins/default_hypothalamus/config.yaml's "
            "llm_purposes.translator and llm.tags in central config). "
            "Skipping registration."
        )
        return None
    return DefaultHypothalamusReflect(llm)


def _format_tentacles(tentacles: list[dict[str, Any]]) -> str:
    if not tentacles:
        return "(none)"
    lines = []
    for t in tentacles:
        lines.append(f"- {t['name']}: {t['description']} "
                     f"params={t.get('parameters_schema', {})}")
    return "\n".join(lines)


def _parse_json(raw: str) -> dict[str, Any]:
    """Lenient JSON extraction. Tries (in order):
      1. raw.strip() + markdown-fence stripping
      2. outermost {...} block
      3. sanitized version of (2): smart quotes → straight, trailing
         commas removed, single quotes → double (when unambiguous)

    Falls through to a safe empty-result dict on total failure rather
    than raising — Hypothalamus errors should never crash the heartbeat.
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
        TentacleCall(
            tentacle=c["tentacle"],
            intent=c.get("intent", ""),
            params=c.get("params") or {},
            adrenalin=bool(c.get("adrenalin", False)),
        )
        for c in (data.get("tentacle_calls") or [])
    ]
    return DecisionResult(
        tentacle_calls=calls,
        memory_writes=list(data.get("memory_writes") or []),
        memory_updates=list(data.get("memory_updates") or []),
        sleep=bool(data.get("sleep", False)),
    )
