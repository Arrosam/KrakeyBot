"""Graph Memory — middle-tier working memory (DevSpec §7).

Phase 1.2a: init + basic node CRUD.
Phase 1.2b: upsert + cycle-checked edges with RELATION bridge.
Phase 1.2c: auto_ingest (embed + similarity dedup).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Protocol

import aiosqlite
import sqlite_vec


class AsyncChatLLM(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...


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


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors.

    Returns 0.0 when either operand is a zero vector (avoids NaN).
    """
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _format_recall_for_prompt(recall: list[dict[str, Any]]) -> str:
    if not recall:
        return "(no prior nodes)"
    lines = []
    for r in recall:
        lines.append(f"- [{r.get('name', '')}] ({r.get('category', '?')}) "
                     f"— {r.get('description', '')}")
    return "\n".join(lines)


def _short_name(text: str, max_len: int = 80) -> str:
    stripped = " ".join(text.split())
    if len(stripped) <= max_len:
        return stripped
    return stripped[: max_len - 1].rstrip() + "…"


SCHEMA_PATH = Path(__file__).parent / "schemas.sql"


class AsyncEmbedder(Protocol):
    async def __call__(self, text: str) -> list[float]: ...


def _encode_embedding(vec: list[float] | None) -> bytes | None:
    if vec is None:
        return None
    return json.dumps(list(vec)).encode("utf-8")


def _decode_embedding(blob: bytes | None) -> list[float] | None:
    if blob is None:
        return None
    return json.loads(blob.decode("utf-8"))


def _row_to_node(row: aiosqlite.Row) -> dict[str, Any]:
    meta_raw = row["metadata"]
    metadata = json.loads(meta_raw) if meta_raw else {}
    return {
        "id": row["id"],
        "name": row["name"],
        "category": row["category"],
        "description": row["description"],
        "importance": row["importance"],
        "metadata": metadata,
        "embedding": _decode_embedding(row["embedding"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "last_accessed": row["last_accessed"],
        "access_count": row["access_count"],
        "source_heartbeat": row["source_heartbeat"],
        "source_type": row["source_type"],
    }


class GraphMemory:
    def __init__(self, db_path: str | Path, embedder: AsyncEmbedder,
                  *, auto_ingest_threshold: float = 0.92,
                  extractor_llm: AsyncChatLLM | None = None,
                  classifier_llm: AsyncChatLLM | None = None,
                  classify_batch_size: int = 10,
                  classify_existing_context: int = 30):
        self.db_path = str(db_path)
        self._embedder = embedder
        self._auto_ingest_threshold = auto_ingest_threshold
        self._extractor_llm = extractor_llm
        self._classifier_llm = classifier_llm
        self._classify_batch_size = classify_batch_size
        self._classify_existing_context = classify_existing_context
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        if self._db is not None:
            return
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        await db.enable_load_extension(True)
        await db.load_extension(sqlite_vec.loadable_path())
        await db.enable_load_extension(False)
        await db.execute("PRAGMA foreign_keys = ON")
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        await db.executescript(schema)
        await db.commit()
        self._db = db

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def _require(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("GraphMemory not initialized; call initialize() first")
        return self._db

    # ---------- node CRUD ----------

    async def insert_node(self, *, name: str, category: str, description: str,
                           embedding: list[float] | None = None,
                           importance: float = 1.0,
                           source_type: str = "auto",
                           source_heartbeat: int | None = None,
                           metadata: dict | None = None) -> int:
        db = self._require()
        cur = await db.execute(
            "INSERT INTO gm_nodes(name, category, description, embedding, "
            "importance, source_type, source_heartbeat, metadata) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (name, category, description, _encode_embedding(embedding),
             importance, source_type, source_heartbeat,
             json.dumps(metadata) if metadata else None),
        )
        await db.commit()
        return cur.lastrowid

    async def get_node(self, node_id: int) -> dict[str, Any] | None:
        db = self._require()
        async with db.execute(
            "SELECT * FROM gm_nodes WHERE id=?", (node_id,)
        ) as cur:
            row = await cur.fetchone()
            return _row_to_node(row) if row else None

    async def list_nodes(self, *, category: str | None = None,
                          limit: int | None = None) -> list[dict[str, Any]]:
        db = self._require()
        sql = "SELECT * FROM gm_nodes"
        params: list[Any] = []
        if category is not None:
            sql += " WHERE category = ?"
            params.append(category)
        sql += " ORDER BY id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
            return [_row_to_node(r) for r in rows]

    async def delete_node(self, node_id: int) -> None:
        db = self._require()
        await db.execute("DELETE FROM gm_nodes WHERE id=?", (node_id,))
        await db.commit()

    async def count_nodes(self) -> int:
        db = self._require()
        async with db.execute("SELECT COUNT(*) FROM gm_nodes") as cur:
            row = await cur.fetchone()
            return int(row[0])

    async def count_edges(self) -> int:
        db = self._require()
        async with db.execute("SELECT COUNT(*) FROM gm_edges") as cur:
            row = await cur.fetchone()
            return int(row[0])

    async def set_metadata(self, node_id: int, delta: dict) -> None:
        """Merge `delta` into the existing metadata JSON."""
        db = self._require()
        current = await self.get_node(node_id)
        if current is None:
            raise KeyError(f"no node id={node_id}")
        merged = {**current["metadata"], **delta}
        await db.execute(
            "UPDATE gm_nodes SET metadata=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (json.dumps(merged), node_id),
        )
        await db.commit()

    # ---------- upsert ----------

    async def upsert_node(self, node: dict[str, Any]) -> int:
        """Same (name, category) → update description/embedding + bump importance.
        Otherwise → new insert. Returns the node id.
        """
        db = self._require()
        name = node["name"]
        category = node["category"]
        async with db.execute(
            "SELECT id, importance FROM gm_nodes WHERE name=? AND category=?",
            (name, category),
        ) as cur:
            row = await cur.fetchone()

        if row is None:
            return await self.insert_node(
                name=name,
                category=category,
                description=node.get("description", ""),
                embedding=node.get("embedding"),
                importance=node.get("importance", 1.0),
                source_type=node.get("source_type", "auto"),
                source_heartbeat=node.get("source_heartbeat"),
                metadata=node.get("metadata"),
            )

        node_id = row["id"]
        new_importance = float(row["importance"]) + 0.5
        desc = node.get("description")
        emb = node.get("embedding")
        sets = ["importance = ?", "updated_at = CURRENT_TIMESTAMP"]
        params: list[Any] = [new_importance]
        if desc is not None:
            sets.append("description = ?")
            params.append(desc)
        if emb is not None:
            sets.append("embedding = ?")
            params.append(_encode_embedding(emb))
        params.append(node_id)
        await db.execute(
            f"UPDATE gm_nodes SET {', '.join(sets)} WHERE id = ?", params,
        )
        await db.commit()
        return node_id

    # ---------- cycle-safe edges ----------

    async def would_create_cycle(self, a: int, b: int) -> bool:
        """Undirected connectivity check (DevSpec §7.7)."""
        if a == b:
            return True
        db = self._require()
        async with db.execute(
            """
            WITH RECURSIVE walk(nid, visited) AS (
                SELECT ?, CAST(? AS TEXT)
                UNION ALL
                SELECT CASE WHEN e.node_a = w.nid THEN e.node_b ELSE e.node_a END,
                       w.visited || ',' ||
                       CAST(CASE WHEN e.node_a = w.nid THEN e.node_b
                                 ELSE e.node_a END AS TEXT)
                FROM walk w
                JOIN gm_edges e ON (e.node_a = w.nid OR e.node_b = w.nid)
                WHERE INSTR(
                    ',' || w.visited || ',',
                    ',' || CAST(CASE WHEN e.node_a = w.nid THEN e.node_b
                                     ELSE e.node_a END AS TEXT) || ','
                ) = 0
            )
            SELECT 1 FROM walk WHERE nid = ? LIMIT 1
            """,
            (a, str(a), b),
        ) as cur:
            row = await cur.fetchone()
            return row is not None

    async def insert_edge_with_cycle_check(self, src: int, tgt: int,
                                             predicate: str) -> dict[str, Any]:
        """Normalize (a<b), check cycle; if cycle, insert a RELATION bridge
        node and link src→bridge→tgt instead. Returns info dict.
        """
        if src == tgt:
            raise ValueError("self-loop edges not allowed")
        a, b = (src, tgt) if src < tgt else (tgt, src)
        db = self._require()

        if await self.would_create_cycle(a, b):
            bridge_id = await self.insert_node(
                name=f"bridge_{a}_{b}_{predicate}",
                category="RELATION",
                description=f"Bridge between nodes {a} and {b} ({predicate}).",
                source_type="auto",
            )
            a1, b1 = (min(a, bridge_id), max(a, bridge_id))
            a2, b2 = (min(b, bridge_id), max(b, bridge_id))
            await db.execute(
                "INSERT INTO gm_edges(node_a, node_b, predicate) VALUES(?, ?, ?)",
                (a1, b1, predicate),
            )
            await db.execute(
                "INSERT INTO gm_edges(node_a, node_b, predicate) VALUES(?, ?, ?)",
                (a2, b2, predicate),
            )
            await db.commit()
            return {"bridged": True, "bridge_node_id": bridge_id}

        await db.execute(
            "INSERT INTO gm_edges(node_a, node_b, predicate) VALUES(?, ?, ?)",
            (a, b, predicate),
        )
        await db.commit()
        return {"bridged": False, "bridge_node_id": None}

    # ---------- vector search ----------

    async def _top_similar(self, query_vec: list[float], *,
                             top_k: int = 1,
                             min_similarity: float = 0.0
                             ) -> list[tuple[dict[str, Any], float]]:
        """Brute-force python-side cosine over rows with embedding != NULL.

        Adequate for Phase 1 scale (≤ soft_limit nodes).
        """
        db = self._require()
        async with db.execute(
            "SELECT * FROM gm_nodes WHERE embedding IS NOT NULL"
        ) as cur:
            rows = await cur.fetchall()
        scored: list[tuple[dict[str, Any], float]] = []
        for row in rows:
            node = _row_to_node(row)
            sim = cosine_similarity(query_vec, node["embedding"])
            if sim >= min_similarity:
                scored.append((node, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ---------- auto_ingest ----------

    async def auto_ingest(self, content: str,
                            *, source_heartbeat: int | None = None
                            ) -> dict[str, Any]:
        """Zero-LLM write (DevSpec §7.5). Embed content; if a similar node
        exists (cosine ≥ threshold), bump its importance; otherwise insert
        a new FACT node. Returns {"created": bool, "node_id": int}.
        """
        db = self._require()
        embedding = await self._embedder(content)
        matches = await self._top_similar(
            embedding, top_k=1, min_similarity=self._auto_ingest_threshold,
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

        node_id = await self.insert_node(
            name=_short_name(content),
            category="FACT",
            description=content,
            embedding=embedding,
            source_type="auto",
            source_heartbeat=source_heartbeat,
        )
        return {"created": True, "node_id": node_id}

    # ---------- explicit write ----------

    async def _find_by_name(self, name: str) -> int | None:
        db = self._require()
        async with db.execute(
            "SELECT id FROM gm_nodes WHERE name = ? LIMIT 1", (name,)
        ) as cur:
            row = await cur.fetchone()
            return int(row["id"]) if row else None

    async def explicit_write(self, content: str, *,
                               importance: str = "normal",
                               recall_context: list[dict[str, Any]] | None = None,
                               source_heartbeat: int | None = None
                               ) -> dict[str, Any]:
        """LLM-assisted write (DevSpec §7.5): ask the extractor LLM to parse
        `content` into nodes + edges, upsert them, and insert edges with
        cycle checking. Returns {"node_ids": [...]}.
        """
        if self._extractor_llm is None:
            raise RuntimeError("explicit_write requires an extractor_llm")

        existing_text = _format_recall_for_prompt(recall_context or [])
        prompt = EXPLICIT_WRITE_PROMPT.format(content=content,
                                                existing=existing_text)
        raw = await self._extractor_llm.chat(
            [{"role": "user", "content": prompt}]
        )
        parsed = _parse_extraction_json(raw)

        imp_value = 2.0 if importance == "high" else 1.0
        node_ids_by_name: dict[str, int] = {}
        for n in parsed.get("nodes", []):
            nid = await self.upsert_node({
                "name": n["name"],
                "category": n["category"],
                "description": n.get("description", ""),
                "source_type": "explicit",
                "importance": imp_value,
                "source_heartbeat": source_heartbeat,
            })
            node_ids_by_name[n["name"]] = nid

        for e in parsed.get("edges", []):
            src = node_ids_by_name.get(e.get("source_name"))
            if src is None:
                src = await self._find_by_name(e.get("source_name", ""))
            tgt = node_ids_by_name.get(e.get("target_name"))
            if tgt is None:
                tgt = await self._find_by_name(e.get("target_name", ""))
            if src is None or tgt is None or src == tgt:
                continue
            await self.insert_edge_with_cycle_check(src, tgt, e["predicate"])

        return {"node_ids": list(node_ids_by_name.values())}

    # ---------- category update + async classify ----------

    async def update_node_category(self, node_name: str,
                                      new_category: str) -> bool:
        """Hypothalamus path: change category by name (e.g. TARGET → FACT).
        Returns True if a row was updated, False if name not found.
        """
        db = self._require()
        cur = await db.execute(
            "UPDATE gm_nodes SET category=?, updated_at=CURRENT_TIMESTAMP "
            "WHERE name=?", (new_category, node_name),
        )
        await db.commit()
        return cur.rowcount > 0

    async def classify_and_link_pending(self) -> dict[str, int]:
        """Background job (DevSpec §7.6): LLM-classify up to N auto-source
        nodes that aren't yet classified, and optionally add edges between
        them and existing classified nodes.
        Returns counters: {"classified": n, "edges": m}.
        """
        if self._classifier_llm is None:
            return {"classified": 0, "edges": 0}

        db = self._require()
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
            (self._classify_batch_size,),
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
            (self._classify_existing_context,),
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
        raw = await self._classifier_llm.chat(
            [{"role": "user", "content": prompt}]
        )
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
                await self.insert_edge_with_cycle_check(int(src), int(tgt),
                                                          predicate)
                edge_count += 1
            except Exception:  # noqa: BLE001
                # malformed ids or FK violation — skip silently (background job)
                continue

        return {"classified": classified_count, "edges": edge_count}

    # ---------- low-level escape hatch for tests ----------

    async def raw_fetchone(self, sql: str, params: tuple = ()):
        db = self._require()
        async with db.execute(sql, params) as cur:
            return await cur.fetchone()
