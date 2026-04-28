"""Sleep KB lifecycle: consolidate similar KBs, archive low-importance KBs,
and revive archived KBs when a new community matches one of them.

Index vector for a KB = mean of its entries' embeddings (active only).
Stored on `kb_registry.index_embedding` so revive matches don't need to
re-embed everything every sleep.

Importance for a KB = entry_count * mean(entry.importance) over active
entries. Archive sorts ascending by importance.
"""
from __future__ import annotations

from typing import Any

from krakey.memory._db import cosine_similarity
from krakey.memory.graph_memory import GraphMemory
from krakey.memory.knowledge_base import KBRegistry, KnowledgeBase


# ---------------- index vector + importance ----------------


async def compute_kb_index_embedding(kb: KnowledgeBase) -> list[float] | None:
    """Mean of all active entries' embeddings. None when no embeddings exist."""
    db = kb._require()  # noqa: SLF001
    async with db.execute(
        "SELECT embedding FROM kb_entries "
        "WHERE is_active = 1 AND embedding IS NOT NULL"
    ) as cur:
        rows = await cur.fetchall()
    from krakey.memory._db import decode_embedding
    vecs = [decode_embedding(r[0]) for r in rows]
    vecs = [v for v in vecs if v]
    if not vecs:
        return None
    dim = len(vecs[0])
    out = [0.0] * dim
    for v in vecs:
        if len(v) != dim:
            continue  # skip dimension mismatches
        for i, x in enumerate(v):
            out[i] += x
    return [x / len(vecs) for x in out]


async def compute_kb_importance(kb: KnowledgeBase) -> float:
    """entry_count * mean(importance) over active entries. 0 when empty."""
    db = kb._require()  # noqa: SLF001
    async with db.execute(
        "SELECT COUNT(*), COALESCE(AVG(importance), 0) "
        "FROM kb_entries WHERE is_active = 1"
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        return 0.0
    n, mean_imp = int(row[0]), float(row[1])
    return n * mean_imp


# ---------------- consolidation ----------------


async def consolidate_kbs(reg: KBRegistry, *, threshold: float
                            ) -> dict[str, int]:
    """Greedy merge: walk active KBs in importance-desc order; for each one,
    if it cosine-matches an already-kept KB above `threshold`, move its
    entries into the kept KB and delete it.

    Returns counters {merged, remaining, with_embedding}.
    """
    metas = await reg.list_kbs(include_archived=False)
    info: list[dict[str, Any]] = []
    for m in metas:
        kb = await reg.open_kb(m["kb_id"])
        emb = await compute_kb_index_embedding(kb)
        if emb is None:
            continue
        await reg.set_index_embedding(m["kb_id"], emb)
        info.append({
            "kb_id": m["kb_id"],
            "kb": kb,
            "emb": emb,
            "importance": await compute_kb_importance(kb),
        })

    info.sort(key=lambda x: x["importance"], reverse=True)
    keep: list[dict[str, Any]] = []
    merged = 0
    for cand in info:
        target = None
        for kept in keep:
            if cosine_similarity(cand["emb"], kept["emb"]) >= threshold:
                target = kept
                break
        if target is None:
            keep.append(cand)
            continue
        await _move_entries(cand["kb"], target["kb"])
        new_emb = await compute_kb_index_embedding(target["kb"])
        if new_emb is not None:
            target["emb"] = new_emb
            await reg.set_index_embedding(target["kb_id"], new_emb)
        await reg.delete_kb(cand["kb_id"])
        merged += 1

    return {"merged": merged, "remaining": len(keep),
            "with_embedding": len(info)}


async def _move_entries(src: KnowledgeBase, dst: KnowledgeBase) -> int:
    """Copy active entries from src to dst (preserving content + embedding +
    tags + importance). Edges are dropped. Returns number copied."""
    src_db = src._require()  # noqa: SLF001
    async with src_db.execute(
        "SELECT content, source, tags, embedding, importance "
        "FROM kb_entries WHERE is_active = 1"
    ) as cur:
        rows = await cur.fetchall()
    import json as _json
    from krakey.memory._db import decode_embedding
    n = 0
    for r in rows:
        await dst.write_entry(
            r["content"],
            tags=_json.loads(r["tags"]) if r["tags"] else None,
            embedding=decode_embedding(r["embedding"]),
            source=r["source"],
            importance=r["importance"],
        )
        n += 1
    return n


# ---------------- archive ----------------


async def archive_excess_kbs(reg: KBRegistry, gm: GraphMemory, *,
                                max_count: int, archive_pct: int
                                ) -> dict[str, int]:
    """When active KB count > max_count, archive the bottom `archive_pct`%
    by importance. Archive = mark `is_archived=1`, persist current index
    vector (so revive can match later), drop GM index node so recall
    forgets it. KB file + entries are kept on disk.
    """
    metas = await reg.list_kbs(include_archived=False)
    if len(metas) <= max_count:
        return {"archived": 0, "active_after": len(metas)}

    n_archive = max(1, (len(metas) * archive_pct) // 100)
    # Score each + sort ascending
    scored: list[tuple[float, dict[str, Any]]] = []
    for m in metas:
        kb = await reg.open_kb(m["kb_id"])
        importance = await compute_kb_importance(kb)
        # Refresh index embedding in case migration just added entries
        emb = await compute_kb_index_embedding(kb)
        if emb is not None:
            await reg.set_index_embedding(m["kb_id"], emb)
        scored.append((importance, m))
    scored.sort(key=lambda x: x[0])

    archived = 0
    for importance, m in scored[:n_archive]:
        await reg.set_archived(m["kb_id"], True)
        await _drop_gm_index_node(gm, m["kb_id"])
        archived += 1

    return {"archived": archived, "active_after": len(metas) - archived}


async def _drop_gm_index_node(gm: GraphMemory, kb_id: str) -> None:
    """Find and delete the GM KNOWLEDGE node whose metadata.kb_id matches.

    rebuild_index_graph will not recreate it on next sleep because that
    pass iterates only active KBs (list_kbs default).
    """
    db = gm._require()  # noqa: SLF001
    async with db.execute(
        "SELECT id, metadata FROM gm_nodes WHERE category = 'KNOWLEDGE' "
        "AND metadata IS NOT NULL"
    ) as cur:
        rows = await cur.fetchall()
    import json as _json
    for r in rows:
        try:
            md = _json.loads(r["metadata"])
        except (TypeError, ValueError):
            continue
        if md.get("is_kb_index") and md.get("kb_id") == kb_id:
            await db.execute("DELETE FROM gm_nodes WHERE id = ?", (r["id"],))
    await db.commit()


# ---------------- revive ----------------


async def find_revive_target(reg: KBRegistry, summary_embedding: list[float],
                                *, threshold: float) -> str | None:
    """Among archived KBs, return the kb_id whose index_embedding is the
    closest cosine match to the new community summary AND beats the
    threshold. None if nothing qualifies."""
    if not summary_embedding:
        return None
    archived = [k for k in await reg.list_kbs(include_archived=True)
                if k["is_archived"] and k["index_embedding"]]
    best_id = None
    best_sim = threshold  # only return matches that beat threshold
    for k in archived:
        sim = cosine_similarity(summary_embedding, k["index_embedding"])
        if sim >= best_sim:
            best_sim = sim
            best_id = k["kb_id"]
    return best_id


async def revive_kb(reg: KBRegistry, kb_id: str) -> KnowledgeBase:
    """Mark archived KB as active again and return it. The next
    rebuild_index_graph pass will recreate its GM index node."""
    await reg.set_archived(kb_id, False)
    return await reg.open_kb(kb_id)
