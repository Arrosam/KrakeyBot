"""Compact — evict oldest rounds from sliding window into GM (DevSpec §10).

Phase 1.4: blocking loop + single-round oversize splitting. Compact LLM
is stateless per call; recall_fn surfaces existing GM nodes so the LLM
can prefer edges over duplicates.
"""
from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable, Protocol

from src.memory.graph_memory import GraphMemory
from src.runtime.sliding_window import SlidingWindow, SlidingWindowRound


COMPACT_PROMPT = """整理一段过去的对话记忆，提取需要记住的信息。

## 内容
Stimulus: {stimulus}
Decision: {decision}
Note: {note}

## 已有节点 (参考；尽量复用以避免重复)
{existing_nodes}

## 输出 (strict JSON, 只输出 JSON)
{{
  "nodes": [
    {{"name": "短标签",
      "category": "FACT|RELATION|KNOWLEDGE|TARGET|FOCUS",
      "description": "简要描述"}}
  ],
  "edges": [
    {{"source_name": "name", "target_name": "name",
      "predicate": "RELATED_TO|SERVES|DEPENDS_ON|INDUCES|SUMMARIZES|"
                   "SUPPORTS|CONTRADICTS|FOLLOWS|CAUSES"}}
  ]
}}

## 规则
1. 只提取值得记住的。日常寒暄不提取。
2. 与已有节点重叠 → 不建新节点，用 edges 连接。
3. 遵循边类型约束。
4. 图必须无环。
"""


class ChatLike(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...


RecallFn = Callable[[str], Awaitable[list[dict[str, Any]]]]


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_compact_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(raw)
        if not m:
            return {"nodes": [], "edges": []}
        return json.loads(m.group(0))


def _format_existing(nodes: list[dict[str, Any]]) -> str:
    if not nodes:
        return "(none)"
    return "\n".join(
        f"- [{n.get('name','')}] ({n.get('category','?')}) — "
        f"{n.get('description','')}"
        for n in nodes
    )


async def _apply_extraction(gm: GraphMemory, parsed: dict[str, Any]) -> None:
    name_to_id: dict[str, int] = {}
    for n in parsed.get("nodes", []):
        try:
            nid = await gm.upsert_node({
                "name": n["name"],
                "category": n["category"],
                "description": n.get("description", ""),
                "source_type": "compact",
            })
            name_to_id[n["name"]] = nid
        except Exception:  # noqa: BLE001
            continue  # malformed node — skip

    for e in parsed.get("edges", []):
        src = name_to_id.get(e.get("source_name"))
        if src is None:
            src = await gm.find_by_name(e.get("source_name", ""))
        tgt = name_to_id.get(e.get("target_name"))
        if tgt is None:
            tgt = await gm.find_by_name(e.get("target_name", ""))
        if src is None or tgt is None or src == tgt:
            continue
        try:
            await gm.insert_edge_with_cycle_check(src, tgt, e["predicate"])
        except Exception:  # noqa: BLE001
            continue  # malformed edge — skip


async def _compact_round(round_: SlidingWindowRound, gm: GraphMemory,
                          llm: ChatLike, recall_fn: RecallFn) -> None:
    query = round_.stimulus_summary or round_.decision_text or round_.note_text
    existing = await recall_fn(query) if query else []
    prompt = COMPACT_PROMPT.format(
        stimulus=round_.stimulus_summary,
        decision=round_.decision_text,
        note=round_.note_text,
        existing_nodes=_format_existing(existing),
    )
    raw = await llm.chat([{"role": "user", "content": prompt}])
    parsed = _parse_compact_json(raw)
    await _apply_extraction(gm, parsed)


def _chunks_by_char_budget(text: str, chars: int) -> list[str]:
    if not text:
        return []
    out = []
    for i in range(0, len(text), chars):
        out.append(text[i : i + chars])
    return out


async def _split_and_compact_single_round(
    window: SlidingWindow, gm: GraphMemory, llm: ChatLike,
    recall_fn: RecallFn, split_chunk_tokens: int,
) -> None:
    """Last resort: single round too big to remove in one shot.
    Pop it and compact each chunk independently."""
    if not window.rounds:
        return
    oldest = window.pop_oldest()
    assert oldest is not None
    full = " ".join(
        t for t in (oldest.stimulus_summary, oldest.decision_text,
                     oldest.note_text) if t
    )
    char_budget = max(80, split_chunk_tokens * 4)  # 4 chars ≈ 1 token
    for chunk in _chunks_by_char_budget(full, char_budget):
        synthetic = SlidingWindowRound(
            heartbeat_id=oldest.heartbeat_id,
            stimulus_summary=chunk,
            decision_text="",
            note_text="",
        )
        await _compact_round(synthetic, gm, llm, recall_fn)


async def compact_if_needed(
    window: SlidingWindow, gm: GraphMemory, llm: ChatLike,
    *, recall_fn: RecallFn, split_chunk_tokens: int = 1000,
) -> None:
    """Blocking compact loop. Evict oldest rounds via LLM summarization until
    the window fits, or a single oversized round remains (then split it).
    """
    while window.needs_compact() and len(window.rounds) > 1:
        oldest = window.pop_oldest()
        assert oldest is not None
        await _compact_round(oldest, gm, llm, recall_fn)

    if window.needs_compact() and len(window.rounds) == 1:
        await _split_and_compact_single_round(
            window, gm, llm, recall_fn, split_chunk_tokens,
        )


# Public alias for callers that need to compact a single
# already-popped round. The overall-input-budget enforcer in
# main.Runtime uses this when it prunes oldest history to make
# room for the current prompt (separate trigger from
# `compact_if_needed`, same per-round mechanics).
compact_round = _compact_round
