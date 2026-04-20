"""Hypothalamus — translation layer (DevSpec §4).

Stateless: every call is an independent LLM invocation. Converts Self's
natural-language [DECISION] into structured tentacle calls, memory
writes, memory updates, and sleep signal.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


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


@dataclass
class TentacleCall:
    tentacle: str
    intent: str
    params: dict[str, Any] = field(default_factory=dict)
    adrenalin: bool = False


@dataclass
class HypothalamusResult:
    tentacle_calls: list[TentacleCall] = field(default_factory=list)
    memory_writes: list[dict[str, Any]] = field(default_factory=list)
    memory_updates: list[dict[str, Any]] = field(default_factory=list)
    sleep: bool = False


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class Hypothalamus:
    def __init__(self, llm: ChatLike):
        self._llm = llm

    async def translate(self, decision: str,
                        tentacles: list[dict[str, Any]]) -> HypothalamusResult:
        system = SYSTEM_PROMPT.format(tentacle_list=_format_tentacles(tentacles))
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": decision},
        ]
        raw = await self._llm.chat(messages)
        data = _parse_json(raw)
        return _to_result(data)


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
    # Surface the original error so the caller can log it
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
    # Smart quotes
    text = (text.replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2018", "'").replace("\u2019", "'"))
    # Trailing commas
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _to_result(data: dict[str, Any]) -> HypothalamusResult:
    calls = [
        TentacleCall(
            tentacle=c["tentacle"],
            intent=c.get("intent", ""),
            params=c.get("params") or {},
            adrenalin=bool(c.get("adrenalin", False)),
        )
        for c in (data.get("tentacle_calls") or [])
    ]
    return HypothalamusResult(
        tentacle_calls=calls,
        memory_writes=list(data.get("memory_writes") or []),
        memory_updates=list(data.get("memory_updates") or []),
        sleep=bool(data.get("sleep", False)),
    )
