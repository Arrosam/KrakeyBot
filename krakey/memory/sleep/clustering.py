"""Phase 2.3a: Leiden clustering + per-community LLM summary (DevSpec §11.3).

Builds an undirected igraph from gm_nodes + gm_edges, runs the Leiden
algorithm, asks the LLM for a 1-2 sentence summary of each community,
embeds the summary, and persists to gm_communities + gm_node_communities.
"""
from __future__ import annotations

import json
from typing import Any, Protocol

import igraph as ig
import leidenalg as la

from krakey.memory._db import encode_embedding
from krakey.memory.graph_memory import GraphMemory, _row_to_node


COMMUNITY_SUMMARY_PROMPT = """以下是一组紧密相关的记忆节点 (Sleep 聚类发现):

{members}

用一两句话概括它们共同的主题 (用作长期 KB 的目录摘要)。
**只输出概括, 不要前缀, 不要列表。**
"""


class AsyncChatLLM(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...


class AsyncEmbedder(Protocol):
    async def __call__(self, text: str) -> list[float]: ...


async def run_leiden_clustering(
    gm: GraphMemory, *, llm: AsyncChatLLM, embedder: AsyncEmbedder,
    min_size: int = 1,
) -> list[dict[str, Any]]:
    """Cluster GM, persist communities, return list of dicts:
        [{community_id, member_ids, size, summary, embedding}]
    """
    nodes, edges = await _fetch_graph(gm)
    if not nodes:
        return []

    partition = _partition_nodes(nodes, edges)
    communities: list[dict[str, Any]] = []

    for member_ids in partition:
        if len(member_ids) < min_size:
            continue
        members = [n for n in nodes if n["id"] in set(member_ids)]
        summary = await _summarize(members, llm)
        embedding = await embedder(summary)
        cid = await _persist_community(gm, summary, embedding,
                                          member_ids)
        communities.append({
            "community_id": cid,
            "member_ids": member_ids,
            "size": len(member_ids),
            "summary": summary,
            "embedding": embedding,
        })

    return communities


# ---------------- internals ----------------


async def _fetch_graph(gm: GraphMemory):
    db = gm._require()  # noqa: SLF001
    async with db.execute(
        "SELECT * FROM gm_nodes ORDER BY id ASC"
    ) as cur:
        node_rows = await cur.fetchall()
    nodes = [_row_to_node(r) for r in node_rows]

    async with db.execute(
        "SELECT node_a, node_b FROM gm_edges"
    ) as cur:
        edge_rows = await cur.fetchall()
    edges = [(int(r[0]), int(r[1])) for r in edge_rows]
    return nodes, edges


def _partition_nodes(nodes: list[dict[str, Any]],
                       edges: list[tuple[int, int]]) -> list[list[int]]:
    id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes))
    valid_edges = [(id_to_idx[a], id_to_idx[b]) for (a, b) in edges
                    if a in id_to_idx and b in id_to_idx]
    g.add_edges(valid_edges)
    partition = la.find_partition(g, la.ModularityVertexPartition)
    out = []
    for cluster in partition:
        out.append([nodes[idx]["id"] for idx in cluster])
    return out


async def _summarize(members: list[dict[str, Any]],
                       llm: AsyncChatLLM) -> str:
    body = "\n".join(
        f"- [{m['name']}] ({m['category']}) {(m.get('description') or '').strip()}"
        for m in members
    )
    prompt = COMMUNITY_SUMMARY_PROMPT.format(members=body)
    raw = await llm.chat([{"role": "user", "content": prompt}])
    return (raw or "").strip()


async def _persist_community(gm: GraphMemory, summary: str,
                                embedding: list[float],
                                member_ids: list[int]) -> int:
    db = gm._require()  # noqa: SLF001
    cur = await db.execute(
        "INSERT INTO gm_communities(name, summary, summary_embedding, "
        "member_count) VALUES(?, ?, ?, ?)",
        (summary[:80], summary, encode_embedding(embedding),
         len(member_ids)),
    )
    cid = cur.lastrowid
    for nid in member_ids:
        await db.execute(
            "INSERT OR IGNORE INTO gm_node_communities(node_id, community_id) "
            "VALUES(?, ?)", (nid, cid),
        )
    await db.commit()
    return int(cid)
