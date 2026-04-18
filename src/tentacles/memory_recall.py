"""Memory Recall Tentacle — Self-initiated memory exploration.

Read-side counterpart to explicit_write. Self uses [DECISION] like
"recall what I know about X" or "回忆一下 Y" → Hypothalamus translates
to a memory_recall tentacle call → this tentacle queries GM
(vec → FTS fallback → neighbor expansion) and returns the result as a
tentacle_feedback stimulus, surfaced under [STIMULUS] / YOUR RECENT
ACTIONS on the next heartbeat.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from src.interfaces.tentacle import Tentacle
from src.memory.graph_memory import GraphMemory
from src.models.stimulus import Stimulus


class AsyncEmbedder(Protocol):
    async def __call__(self, text: str) -> list[float]: ...


class MemoryRecallTentacle(Tentacle):
    def __init__(self, gm: GraphMemory | None,
                  embedder: AsyncEmbedder | None,
                  *, default_top_k: int = 8):
        self._gm = gm
        self._embedder = embedder
        self._default_top_k = default_top_k

    @property
    def name(self) -> str:
        return "memory_recall"

    @property
    def description(self) -> str:
        return ("Self-initiated memory recall. Use when you want to remember, "
                "reflect, or browse what you already know about a topic. "
                "Returns matching nodes + their neighbors + edges between them.")

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "query": "free-text topic / question to recall",
            "top_k": "max nodes to return (default 8)",
        }

    @property
    def sandboxed(self) -> bool:
        return True

    async def execute(self, intent: str,
                        params: dict[str, Any]) -> Stimulus:
        assert self._gm is not None and self._embedder is not None
        query = (params.get("query") or intent or "").strip()
        top_k = int(params.get("top_k") or self._default_top_k)

        candidates: list[tuple[dict[str, Any], float]] = []
        try:
            vec = await self._embedder(query)
            candidates = await self._gm.vec_search(
                vec, top_k=top_k, min_similarity=0.3,
            )
        except Exception:  # noqa: BLE001
            candidates = []

        if not candidates:
            fts_hits = await self._gm.fts_search(query, top_k=top_k)
            candidates = [(n, 0.0) for n in fts_hits]

        # de-dup + cap top_k
        seen: set[int] = set()
        nodes: list[dict[str, Any]] = []
        for node, _sim in candidates:
            if node["id"] in seen:
                continue
            seen.add(node["id"])
            nodes.append(node)
            if len(nodes) >= top_k:
                break

        if not nodes:
            return self._stim(
                f"No matching memories for query: {query!r}.",
            )

        node_ids = [n["id"] for n in nodes]
        neighbor_map = await self._gm.get_neighbor_keywords(node_ids)
        edges = await self._gm.get_edges_among(node_ids)

        return self._stim(_format(nodes, neighbor_map, edges, query))

    def _stim(self, content: str) -> Stimulus:
        return Stimulus(
            type="tentacle_feedback",
            source=f"tentacle:{self.name}",
            content=content,
            timestamp=datetime.now(),
            adrenalin=False,
        )


def _format(nodes: list[dict[str, Any]],
             neighbors: dict[int, list[str]],
             edges: list[dict[str, Any]],
             query: str) -> str:
    lines = [f"Recall result for {query!r} — {len(nodes)} node(s):"]
    for n in nodes:
        nb = ", ".join(neighbors.get(n["id"], [])) or "(no neighbors)"
        desc = (n.get("description") or "").strip()
        if len(desc) > 200:
            desc = desc[:197] + "..."
        lines.append(
            f"- [{n['name']}] ({n['category']}, src={n.get('source_type')}, "
            f"imp={n.get('importance', 1.0):.1f}) {desc}"
        )
        lines.append(f"    neighbors: {nb}")
    if edges:
        lines.append("Edges:")
        for e in edges:
            lines.append(f"  [{e['source']}] --{e['predicate']}--> "
                          f"[{e['target']}]")
    return "\n".join(lines)
