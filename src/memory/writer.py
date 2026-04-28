"""LLM-driven Graph Memory writers (DevSpec §7.5–§7.6).

Pulled out of GraphMemory to keep that class focused on storage +
graph algorithms. The writers compose GM's CRUD + edge primitives;
they do NOT depend on GM's internal connection.

Three strategies:
  * ``auto_ingest`` — zero-LLM: embed → cosine match against existing
    nodes, bump importance on hit, insert as FACT on miss.
  * ``explicit_write`` — extractor LLM parses content into nodes +
    edges, then upserts and inserts edges with cycle checking.
  * ``classify_and_link_pending`` — background classifier LLM picks
    a category for unclassified ``source_type='auto'`` nodes and
    optionally adds edges to existing classified ones.

GraphMemory keeps thin wrapper methods (``gm.auto_ingest`` etc.) that
forward to these functions, so callers don't change.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from src.memory.graph_memory import GraphMemory


class AsyncChatLLM(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...


# --------------------------------------------------------------------
# LLM prompts
# --------------------------------------------------------------------


EXPLICIT_WRITE_PROMPT = """Extract nodes and edges from the content below.

## Content
{content}

## Existing nodes (reuse these names when relevant; do not duplicate)
{existing}

## Output (strict JSON, no commentary)
{{
  "nodes": [
    {{"name": "short entity label",
      "category": "FACT|RELATION|KNOWLEDGE|TARGET|FOCUS",
      "description": "full description"}}
  ],
  "edges": [
    {{"source_name": "name", "target_name": "name",
      "predicate": "RELATED_TO|SERVES|DEPENDS_ON|INDUCES|SUMMARIZES|"
                   "SUPPORTS|CONTRADICTS|FOLLOWS|CAUSES"}}
  ]
}}

Rules:
1. Extract only what is worth remembering. Skip small talk.
2. Prefer reusing existing node names over creating duplicates.
3. Respect predicate-type constraints (e.g. SERVES: FOCUS→TARGET).
4. Graph must remain acyclic.
"""


CLASSIFY_PROMPT = """Classify each pending node into a category and optionally
add relational edges to existing classified nodes.

## Pending (choose one category per node)
{pending}

## Existing classified nodes (reference only — may be edge targets)
{existing}

## Output (strict JSON, no commentary)
{{
  "classifications": [
    {{"node_id": <int>, "category": "FACT|RELATION|KNOWLEDGE|TARGET|FOCUS"}}
  ],
  "edges": [
    {{"source_id": <int>, "target_id": <int>,
      "predicate": "RELATED_TO|SERVES|DEPENDS_ON|INDUCES|SUMMARIZES|"
                   "SUPPORTS|CONTRADICTS|FOLLOWS|CAUSES"}}
  ]
}}

Rules:
1. Classify every pending node.
2. Edges optional; include only if clearly justified.
3. Honor predicate/category constraints.
4. Resulting graph must remain acyclic.
"""


# --------------------------------------------------------------------
# Prompt-formatting + JSON-parsing helpers
# --------------------------------------------------------------------


def _format_pending(pending: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- id={p['id']}  name={p['name']!r}  desc={p['description']!r}"
        for p in pending
    ) or "(none)"


def _format_existing(existing: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- id={e['id']}  ({e['category']}) name={e['name']!r}"
        for e in existing
    ) or "(none)"


def _format_recall_for_prompt(recall: list[dict[str, Any]]) -> str:
    if not recall:
        return "(no prior nodes)"
    lines = []
    for r in recall:
        lines.append(f"- [{r.get('name', '')}] ({r.get('category', '?')}) "
                     f"— {r.get('description', '')}")
    return "\n".join(lines)


_JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _parse_extraction_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_PATTERN.search(raw)
        if not m:
            raise
        return json.loads(m.group(0))


_MIN_INGEST_CHARS = 4
"""Below this length, auto_ingest treats content as noise and skips it."""


def _is_meaningful(content: str) -> bool:
    """Reject pure-symbol / emoji / whitespace blobs ("✓", "...", "❤️").

    Need at least one alphanumeric char (Unicode letters + digits count, so
    Chinese/Cyrillic/etc. work) AND total length above the floor.
    """
    if content is None:
        return False
    stripped = content.strip()
    if len(stripped) < _MIN_INGEST_CHARS:
        return False
    return any(ch.isalnum() for ch in stripped)


def _short_name(text: str, max_len: int = 80) -> str:
    stripped = " ".join(text.split())
    if len(stripped) <= max_len:
        return stripped
    return stripped[: max_len - 1].rstrip() + "…"


# --------------------------------------------------------------------
# Public writer functions — first arg is the GraphMemory to write into
# --------------------------------------------------------------------


async def auto_ingest(
    gm: "GraphMemory",
    content: str,
    *,
    source_heartbeat: int | None = None,
) -> dict[str, Any]:
    """Zero-LLM write (DevSpec §7.5). Embed content; if a similar node
    exists (cosine ≥ threshold), bump its importance; otherwise insert
    a new FACT node. Returns ``{"created": bool, "node_id": int|None,
    "skipped": bool}``.
    """
    if not _is_meaningful(content):
        return {"created": False, "node_id": None, "skipped": True}
    db = gm._require()
    embedding = await gm._embedder(content)
    matches = await gm.vec_search(
        embedding, top_k=1, min_similarity=gm._auto_ingest_threshold,
    )
    if matches:
        node, _sim = matches[0]
        new_imp = float(node["importance"]) + 0.5
        await db.execute(
            "UPDATE gm_nodes SET importance=?, updated_at=CURRENT_TIMESTAMP "
            "WHERE id=?",
            (new_imp, node["id"]),
        )
        await db.commit()
        return {"created": False, "node_id": node["id"]}

    node_id = await gm.insert_node(
        name=_short_name(content),
        category="FACT",
        description=content,
        embedding=embedding,
        source_type="auto",
        source_heartbeat=source_heartbeat,
    )
    return {"created": True, "node_id": node_id}


async def explicit_write(
    gm: "GraphMemory",
    content: str,
    *,
    extractor_llm: AsyncChatLLM,
    importance: str = "normal",
    recall_context: list[dict[str, Any]] | None = None,
    source_heartbeat: int | None = None,
) -> dict[str, Any]:
    """LLM-assisted write (DevSpec §7.5). Returns ``{"node_ids": [...]}``."""
    existing_text = _format_recall_for_prompt(recall_context or [])
    prompt = EXPLICIT_WRITE_PROMPT.format(content=content,
                                            existing=existing_text)
    raw = await extractor_llm.chat([{"role": "user", "content": prompt}])
    parsed = _parse_extraction_json(raw)

    imp_value = 2.0 if importance == "high" else 1.0
    node_ids_by_name: dict[str, int] = {}
    for n in parsed.get("nodes", []):
        nid = await gm.upsert_node({
            "name": n["name"],
            "category": n["category"],
            "description": n.get("description", ""),
            "source_type": "explicit",
            "importance": imp_value,
            "source_heartbeat": source_heartbeat,
            "metadata": {"classified": True},
        })
        # upsert_node only writes metadata for fresh inserts; force-merge
        # so an existing node also picks up the flag.
        await gm.set_metadata(nid, {"classified": True})
        node_ids_by_name[n["name"]] = nid

    for e in parsed.get("edges", []):
        src = node_ids_by_name.get(e.get("source_name"))
        if src is None:
            src = await gm.find_by_name(e.get("source_name", ""))
        tgt = node_ids_by_name.get(e.get("target_name"))
        if tgt is None:
            tgt = await gm.find_by_name(e.get("target_name", ""))
        if src is None or tgt is None or src == tgt:
            continue
        await gm.insert_edge_with_cycle_check(src, tgt, e["predicate"])

    return {"node_ids": list(node_ids_by_name.values())}


async def classify_and_link_pending(
    gm: "GraphMemory",
    *,
    classifier_llm: AsyncChatLLM,
    batch_size: int,
    existing_context: int,
) -> dict[str, int]:
    """Background job (DevSpec §7.6). Returns ``{"classified": n, "edges": m}``."""
    db = gm._require()
    async with db.execute(
        """
        SELECT id, name, description FROM gm_nodes
        WHERE source_type='auto'
          AND (metadata IS NULL
               OR json_extract(metadata, '$.classified') IS NULL
               OR json_extract(metadata, '$.classified') = 0)
        ORDER BY created_at ASC
        LIMIT ?
        """,
        (batch_size,),
    ) as cur:
        pending_rows = await cur.fetchall()

    if not pending_rows:
        return {"classified": 0, "edges": 0}

    async with db.execute(
        """
        SELECT id, name, category, description FROM gm_nodes
        WHERE json_extract(metadata, '$.classified') = 1
        ORDER BY last_accessed DESC
        LIMIT ?
        """,
        (existing_context,),
    ) as cur:
        existing_rows = await cur.fetchall()

    pending = [{"id": r["id"], "name": r["name"],
                 "description": r["description"]} for r in pending_rows]
    existing = [{"id": r["id"], "name": r["name"],
                  "category": r["category"],
                  "description": r["description"]} for r in existing_rows]

    prompt = CLASSIFY_PROMPT.format(
        pending=_format_pending(pending),
        existing=_format_existing(existing),
    )
    raw = await classifier_llm.chat([{"role": "user", "content": prompt}])
    parsed = _parse_extraction_json(raw)

    classified_count = 0
    for c in parsed.get("classifications", []):
        node_id = c.get("node_id")
        category = c.get("category")
        if node_id is None or category is None:
            continue
        await db.execute(
            """
            UPDATE gm_nodes
            SET category = ?,
                metadata = json_set(COALESCE(metadata, '{}'),
                                     '$.classified', json('true')),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (category, node_id),
        )
        classified_count += 1
    await db.commit()

    edge_count = 0
    for e in parsed.get("edges", []):
        src = e.get("source_id")
        tgt = e.get("target_id")
        predicate = e.get("predicate")
        if src is None or tgt is None or predicate is None or src == tgt:
            continue
        try:
            await gm.insert_edge_with_cycle_check(int(src), int(tgt), predicate)
            edge_count += 1
        except Exception:  # noqa: BLE001
            # malformed ids or FK violation — skip silently (background job)
            continue

    return {"classified": classified_count, "edges": edge_count}
