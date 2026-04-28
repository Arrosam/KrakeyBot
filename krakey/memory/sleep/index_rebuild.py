"""Phase 2.3c: rebuild GM Index Graph after Sleep migration (DevSpec §11.4).

For each KB in kb_registry:
  - upsert a KNOWLEDGE node in GM with metadata {is_kb_index, kb_id,
    entry_count} so memory_recall can follow it back into the KB.
  - update kb_registry.entry_count to the live count.

When more than one KB exists, ask the LLM to identify cross-KB relations
and insert edges between the index nodes (cycle-safe via the GM helper).
"""
from __future__ import annotations

import json
import re
from typing import Any, Protocol

from krakey.memory.graph_memory import GraphMemory
from krakey.memory.knowledge_base import KBRegistry


KB_RELATION_PROMPT = """以下是当前所有 Knowledge Base 的元数据 (Sleep 后的索引层):

{kbs}

请判断哪些 KB 之间存在概念上的关联, 输出关系列表 (JSON):

{{
  "edges": [
    {{"source_kb_id": "...", "target_kb_id": "...",
      "predicate": "RELATED_TO|CAUSES|FOLLOWS|CONTRADICTS|SUPPORTS"}}
  ]
}}

规则:
- 只输出 JSON, 不要前缀。
- 关系应明确而非牵强。
- source 和 target 都必须是上面列表中的 kb_id。
"""


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class AsyncChatLLM(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...


class AsyncEmbedder(Protocol):
    async def __call__(self, text: str) -> list[float]: ...


async def rebuild_index_graph(gm: GraphMemory, reg: KBRegistry, *,
                                 llm: AsyncChatLLM,
                                 embedder: AsyncEmbedder
                                 ) -> dict[str, int]:
    """Rebuild KB index nodes + cross-KB edges. Returns counters."""
    kbs = await reg.list_kbs()
    if not kbs:
        return {"index_nodes": 0, "edges_added": 0}

    # Refresh entry counts (may be stale after migration)
    enriched: list[dict[str, Any]] = []
    db = gm._require()  # noqa: SLF001
    for meta in kbs:
        kb = await reg.open_kb(meta["kb_id"])
        live_count = await kb.count_entries()
        await db.execute(
            "UPDATE kb_registry SET entry_count = ?, "
            "updated_at = CURRENT_TIMESTAMP WHERE kb_id = ?",
            (live_count, meta["kb_id"]),
        )
        meta = dict(meta)
        meta["entry_count"] = live_count
        enriched.append(meta)
    await db.commit()

    # Upsert one KNOWLEDGE index node per KB
    name_to_node_id: dict[str, int] = {}  # kb_id → node_id
    for meta in enriched:
        index_name = f"KB:{meta['kb_id']}"
        node_id = await gm.upsert_node({
            "name": index_name,
            "category": "KNOWLEDGE",
            "description": (
                f"知识库索引: {meta.get('description') or meta['name']}. "
                f"{meta['entry_count']} 条目。"
            ),
            "embedding": await embedder(
                f"{meta['name']} {meta.get('description') or ''}"
            ),
            "source_type": "sleep",
            "metadata": {
                "is_kb_index": True,
                "kb_id": meta["kb_id"],
                "entry_count": meta["entry_count"],
            },
        })
        # Force-merge metadata (upsert_node only writes metadata for fresh
        # inserts; for idempotent re-runs we need updates too).
        await gm.set_metadata(node_id, {
            "is_kb_index": True,
            "kb_id": meta["kb_id"],
            "entry_count": meta["entry_count"],
        })
        name_to_node_id[meta["kb_id"]] = node_id

    edges_added = 0
    if len(enriched) > 1:
        edges_added = await _llm_link_kbs(
            gm, llm, enriched, name_to_node_id,
        )

    return {"index_nodes": len(enriched), "edges_added": edges_added}


async def _llm_link_kbs(gm: GraphMemory, llm: AsyncChatLLM,
                          metas: list[dict[str, Any]],
                          name_to_node_id: dict[str, int]) -> int:
    body = "\n".join(
        f"- kb_id={m['kb_id']!r}  name={m['name']!r}  "
        f"description={m.get('description') or '(none)'!r}  "
        f"entries={m['entry_count']}"
        for m in metas
    )
    prompt = KB_RELATION_PROMPT.format(kbs=body)
    raw = await llm.chat([{"role": "user", "content": prompt}])
    parsed = _parse_json_block(raw)
    if not parsed:
        return 0

    added = 0
    for e in parsed.get("edges", []):
        src_kb = e.get("source_kb_id")
        tgt_kb = e.get("target_kb_id")
        predicate = e.get("predicate")
        if not (src_kb and tgt_kb and predicate):
            continue
        src_node = name_to_node_id.get(src_kb)
        tgt_node = name_to_node_id.get(tgt_kb)
        if src_node is None or tgt_node is None or src_node == tgt_node:
            continue
        info = await gm.insert_edge_with_cycle_check(
            src_node, tgt_node, predicate,
        )
        if not info.get("skipped"):
            added += 1
    return added


def _parse_json_block(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(raw or "")
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
