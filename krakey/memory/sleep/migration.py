"""Phase 2.3b: GM → KB migration (DevSpec §11.3 Phase 3).

For each FACT/RELATION/KNOWLEDGE node in GM:
  - locate its community (run after clustering)
  - find or create a KB for that community (or revive an archived one
    whose stored index vector is cosine-close to the new summary)
  - migrate the node into the KB as an entry
  - migrate intra-community edges (both endpoints in same community)
  - delete the node from GM (cascade removes other GM edges)

Cross-community edges are intentionally dropped here; the KB index graph
(Phase 2.3c) re-expresses inter-KB relations on a higher level.

Communities below `min_community_size` (config) are left untouched in
GM — small clusters tend to be noise that may grow into something next
sleep, so we don't materialize a tiny KB for them.
"""
from __future__ import annotations

import json
from typing import Any

from typing import Protocol

from krakey.memory._db import decode_embedding
from krakey.memory.graph_memory import GraphMemory
from krakey.memory.knowledge_base import KBRegistry, KnowledgeBase
from krakey.memory.recall import Reranker, rerank
from krakey.memory.sleep.kb_lifecycle import find_revive_target, revive_kb


class AsyncChatLLM(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...


_MIGRATABLE = ("FACT", "RELATION", "KNOWLEDGE")


async def migrate_gm_to_kb(gm: GraphMemory, reg: KBRegistry, *,
                              llm: AsyncChatLLM,
                              reranker: Reranker | None = None,
                              dedup_top_k: int = 5,
                              min_community_size: int = 1,
                              revive_threshold: float = 0.80,
                              ) -> dict[str, int]:
    """Run the migration pass. Returns counters."""
    db = gm._require()  # noqa: SLF001
    counters = {
        "migrated_nodes": 0, "merged_entries": 0,
        "migrated_edges": 0,
        "skipped_no_community": 0, "skipped_small_community": 0,
        "kbs_created": 0, "kbs_revived": 0,
    }

    nodes = await _fetch_migratable(db)
    if not nodes:
        return counters

    node_to_community = await _node_to_community_map(db)

    # Honor min_community_size: skip nodes whose community is below the
    # threshold. Tiny clusters (often singletons) aren't worth a KB.
    community_sizes = await _community_sizes(db, node_to_community)
    too_small = {cid for cid, sz in community_sizes.items()
                 if sz < min_community_size}

    community_kbs: dict[int, KnowledgeBase] = {}
    gm_to_kb_entry: dict[int, tuple[int, int]] = {}  # gm_node_id → (kb_id, entry_id)

    # First pass: write entries
    for node in nodes:
        community_id = node_to_community.get(node["id"])
        if community_id is None:
            counters["skipped_no_community"] += 1
            continue
        if community_id in too_small:
            counters["skipped_small_community"] += 1
            continue
        kb = community_kbs.get(community_id)
        if kb is None:
            kb, status = await _find_or_create_kb_for_community(
                db, reg, community_id, revive_threshold=revive_threshold,
            )
            community_kbs[community_id] = kb
            if status == "created":
                counters["kbs_created"] += 1
            elif status == "revived":
                counters["kbs_revived"] += 1

        entry_id, merged = await _dedup_or_write(
            kb, node,
            judge_llm=llm, reranker=reranker,
            top_k=dedup_top_k,
        )
        gm_to_kb_entry[node["id"]] = (community_id, entry_id)
        counters["migrated_nodes"] += 1
        if merged:
            counters["merged_entries"] += 1

    # Second pass: migrate intra-community edges
    counters["migrated_edges"] = await _migrate_edges(
        db, gm_to_kb_entry, community_kbs,
    )

    # Third pass: delete migrated GM nodes (cascade-deletes their edges)
    for gm_node_id in gm_to_kb_entry:
        await db.execute("DELETE FROM gm_nodes WHERE id=?", (gm_node_id,))
    await db.commit()

    return counters


async def _community_sizes(db, node_to_community: dict[int, int]
                              ) -> dict[int, int]:
    sizes: dict[int, int] = {}
    for cid in node_to_community.values():
        sizes[cid] = sizes.get(cid, 0) + 1
    return sizes


async def _fetch_migratable(db) -> list[dict[str, Any]]:
    placeholders = ",".join("?" * len(_MIGRATABLE))
    async with db.execute(
        f"SELECT * FROM gm_nodes WHERE category IN ({placeholders})",
        list(_MIGRATABLE),
    ) as cur:
        rows = await cur.fetchall()
    # Avoid circular import at module load
    from krakey.memory.graph_memory import _row_to_node
    return [_row_to_node(r) for r in rows]


async def _node_to_community_map(db) -> dict[int, int]:
    """Return node_id → community_id (first community wins for nodes
    that ended up in multiple — Leiden usually gives one)."""
    out: dict[int, int] = {}
    async with db.execute(
        "SELECT node_id, community_id FROM gm_node_communities "
        "ORDER BY community_id ASC"
    ) as cur:
        async for row in cur:
            nid = int(row[0])
            cid = int(row[1])
            out.setdefault(nid, cid)
    return out


async def _find_or_create_kb_for_community(
    db, reg: KBRegistry, community_id: int, *, revive_threshold: float,
) -> tuple[KnowledgeBase, str]:
    """Returns (kb, status) where status is one of:
        "revived"  — matched an archived KB; reactivated
        "created"  — fresh new KB
        "reused"   — kb_id already present (re-run of same sleep)
    """
    async with db.execute(
        "SELECT name, summary, summary_embedding FROM gm_communities "
        "WHERE community_id=?",
        (community_id,),
    ) as cur:
        row = await cur.fetchone()
    name = (row["name"] if row else None) or f"community {community_id}"
    summary = (row["summary"] if row else "") or name
    summary_emb = decode_embedding(row["summary_embedding"]) if row else None

    # Try to revive an archived KB whose stored index vector is cosine-close.
    # Skips silently if no embedding or no archived candidates.
    if summary_emb:
        revived_id = await find_revive_target(
            reg, summary_emb, threshold=revive_threshold,
        )
        if revived_id is not None:
            return await revive_kb(reg, revived_id), "revived"

    kb_id = f"community_{community_id}"
    try:
        kb = await reg.create_kb(kb_id, name=name, description=summary)
        return kb, "created"
    except ValueError:
        # Already exists from a prior sleep; reopen
        return await reg.open_kb(kb_id), "reused"


async def _migrate_edges(db, gm_to_kb_entry: dict[int, tuple[int, int]],
                            community_kbs: dict[int, "KnowledgeBase"]) -> int:
    if not gm_to_kb_entry:
        return 0
    gm_ids = list(gm_to_kb_entry.keys())
    placeholders = ",".join("?" * len(gm_ids))
    async with db.execute(
        f"""
        SELECT node_a, node_b, predicate FROM gm_edges
        WHERE node_a IN ({placeholders}) AND node_b IN ({placeholders})
        """,
        gm_ids + gm_ids,
    ) as cur:
        rows = await cur.fetchall()

    migrated = 0
    for row in rows:
        a, b, pred = int(row[0]), int(row[1]), row[2]
        ca, ea = gm_to_kb_entry[a]
        cb, eb = gm_to_kb_entry[b]
        if ca != cb:
            continue  # cross-community → leave for index_rebuild
        if ea == eb:
            # Both endpoints collapsed into the same KB entry by the
            # dedup pass. The relationship is now self-referential — KB
            # rejects self-loops, and there's nothing useful to record
            # anyway (an entry can't relate to itself in the KB graph).
            continue
        kb = community_kbs[ca]
        info = await kb.write_edge(ea, eb, pred)
        if info["written"]:
            migrated += 1
    return migrated


# --------------------------------------------------------------------
# Semantic dedup: before writing a fresh entry, ask the LLM whether the
# incoming GM node is the same thing as an existing nearby KB entry.
# On match → merge (sum importance, union tags); otherwise → fresh write.
# Every external call (reranker, LLM judge) is fail-soft so a flaky
# service can never crash sleep — the worst case is fresh-write, which
# matches the pre-feature behavior.
# --------------------------------------------------------------------


async def _dedup_or_write(
    kb: KnowledgeBase, node: dict[str, Any], *,
    judge_llm: AsyncChatLLM,
    reranker: Reranker | None,
    top_k: int = 5,
) -> tuple[int, bool]:
    """Returns ``(entry_id, merged)``. ``merged=True`` iff the GM node
    was folded into an existing KB entry instead of inserted fresh."""
    new_content = node.get("description") or node["name"]
    new_imp = node.get("importance", 1.0)
    new_tags = [node["category"], node["name"]]
    embedding = node.get("embedding")

    candidates: list[tuple[dict[str, Any], float]] = []
    if embedding is not None:
        candidates = await kb.vec_search(embedding, top_k=top_k)

    # Reranker first; cosine fallback when rerank() reports None.
    ranked = await rerank(candidates, query=new_content, reranker=reranker)
    ordered: list[dict[str, Any]]
    if ranked is None:
        ordered = [n for (n, _sim) in candidates]
    else:
        ordered = [n for (n, _score) in ranked]

    if ordered:
        match_idx = await _llm_pick_same(judge_llm, new_content, ordered)
        if match_idx is not None:
            cand = ordered[match_idx]
            await kb.merge_entry(
                cand["id"],
                new_content=new_content,
                new_embedding=embedding,
                new_importance=new_imp,
                new_tags=new_tags,
            )
            return cand["id"], True

    entry_id = await kb.write_entry(
        new_content,
        tags=new_tags,
        embedding=embedding,
        source=f"gm_node:{node['id']}",
        importance=new_imp,
    )
    return entry_id, False


async def _llm_pick_same(llm: AsyncChatLLM, new_content: str,
                            candidates: list[dict[str, Any]]) -> int | None:
    """One LLM call. Returns 0-based index of the matching candidate
    or ``None`` on no-match / parse failure / exception."""
    listing = "\n\n".join(
        f"[{i + 1}] {c['content']}" for i, c in enumerate(candidates)
    )
    prompt = (
        "Decide whether the NEW text describes the same fact, "
        "concept, or event as any of the EXISTING entries.\n\n"
        "Reply with strictly one of: a single integer (1-based) "
        "naming the matching entry, or the word NONE. First line only.\n\n"
        f"NEW:\n{new_content}\n\nEXISTING:\n{listing}"
    )
    try:
        raw = await llm.chat([{"role": "user", "content": prompt}])
    except Exception:  # noqa: BLE001
        return None
    first = raw.strip().split("\n", 1)[0].strip().upper()
    if first.startswith("NONE"):
        return None
    try:
        idx = int(first.split()[0]) - 1
    except (ValueError, IndexError):
        return None
    if 0 <= idx < len(candidates):
        return idx
    return None
