"""``memory_recall`` tentacle — Self-driven explicit GM/KB exploration.

Read-side counterpart to ``gm.explicit_write`` (the LLM-extraction
write path). Self emits ``[DECISION]`` like "recall what I know about
X" → orchestrator dispatches a ``memory_recall`` tentacle call → this
tentacle queries GM (via the shared ``gm_query`` helper) and returns
the result as a ``tentacle_feedback`` Stimulus, surfaced under
``[STIMULUS]`` / ``YOUR RECENT ACTIONS`` on the next heartbeat.

Two query paths, picked by the presence of ``kb_id`` in params:
  * ``kb_id`` given → bypass GM, query that KnowledgeBase directly.
    Used when Self has noticed a KB index node in ``[GRAPH MEMORY]``
    on a prior beat and wants to drill in.
  * default → vec_search GM with FTS fallback, dedup to top-K, fetch
    neighbors + edges. Plus: any recalled node that itself is a
    ``kb_index`` (placed by Sleep when migrating GM content into a
    KB) auto-expands its KB's matching entries into the result. That
    auto-expansion is the very mechanism that *introduces* Self to
    KB existence — surface a KB index node, and on the next active
    recall Self can name the KB and drill in.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from krakey.interfaces.tentacle import Tentacle
from krakey.memory.graph_memory import GraphMemory
from krakey.memory.knowledge_base import KBRegistry
from krakey.memory.recall import AsyncEmbedder
from krakey.models.stimulus import Stimulus
from krakey.plugins.recall.gm_query import query_gm_with_fts_fallback


class MemoryRecallTentacle(Tentacle):
    def __init__(self, gm: GraphMemory | None,
                  embedder: AsyncEmbedder | None,
                  *, default_top_k: int = 8,
                  kb_registry: KBRegistry | None = None):
        self._gm = gm
        self._embedder = embedder
        self._default_top_k = default_top_k
        self._kb_registry = kb_registry

    @property
    def name(self) -> str:
        return "memory_recall"

    @property
    def description(self) -> str:
        return ("Self-initiated memory recall. Use when you want to remember, "
                "reflect, or browse what you already know about a topic. "
                "Returns matching GM nodes + neighbors + edges + (when a KB "
                "index node hits or kb_id is given) entries from that KB.")

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "query": "free-text topic / question to recall",
            "top_k": "max nodes to return (default 8)",
            "kb_id": "optional — query a specific KB directly, skipping GM",
        }

    async def execute(self, intent: str,
                        params: dict[str, Any]) -> Stimulus:
        if self._gm is None or self._embedder is None:
            return self._stim(
                "memory_recall not configured (gm or embedder missing). "
                "This is a setup error — check the runtime services dict."
            )
        query = (params.get("query") or intent or "").strip()
        top_k = int(params.get("top_k") or self._default_top_k)
        explicit_kb = params.get("kb_id")

        # Direct KB browse — bypass GM entirely
        if explicit_kb and self._kb_registry is not None:
            try:
                kb = await self._kb_registry.open_kb(explicit_kb)
            except KeyError:
                return self._stim(f"No KB registered with id {explicit_kb!r}.")
            entries = await kb.search(query, top_k=top_k)
            if not entries:
                return self._stim(
                    f"KB {explicit_kb!r} returned no matches for {query!r}."
                )
            return self._stim(_format_kb(explicit_kb, entries, query))

        # Normal path: GM vec → FTS fallback (via shared helper)
        nodes = await self._search_gm(query, top_k)
        if not nodes:
            return self._stim(
                f"No matching memories for query: {query!r}.",
            )

        node_ids = [n["id"] for n in nodes]
        neighbor_map = await self._gm.get_neighbor_keywords(node_ids)
        edges = await self._gm.get_edges_among(node_ids)

        # KB index expansion: if any recalled node is a KB index, also
        # pull entries from that KB. This is how Self learns a KB
        # exists — see module docstring.
        kb_sections = await self._expand_kb_indexes(nodes, query, top_k)

        return self._stim(_format(nodes, neighbor_map, edges, query)
                           + kb_sections)

    async def _search_gm(self, query: str,
                          top_k: int) -> list[dict[str, Any]]:
        candidates = await query_gm_with_fts_fallback(
            self._gm, self._embedder, query, top_k=top_k,
        )
        # Dedup + cap to top_k. Defensive — vec_search and fts_search
        # individually shouldn't repeat a node, but the cap layer here
        # also limits the nodes Self sees regardless of search quirks.
        seen: set[int] = set()
        out: list[dict[str, Any]] = []
        for node, _sim in candidates:
            if node["id"] in seen:
                continue
            seen.add(node["id"])
            out.append(node)
            if len(out) >= top_k:
                break
        return out

    async def _expand_kb_indexes(self, nodes: list[dict[str, Any]],
                                   query: str, top_k: int) -> str:
        if self._kb_registry is None:
            return ""
        sections: list[str] = []
        for n in nodes:
            meta = n.get("metadata") or {}
            if not meta.get("is_kb_index"):
                continue
            kb_id = meta.get("kb_id")
            if not kb_id:
                continue
            try:
                kb = await self._kb_registry.open_kb(kb_id)
            except KeyError:
                continue
            entries = await kb.search(query, top_k=top_k)
            if not entries:
                continue
            sections.append("\n" + _format_kb(kb_id, entries, query))
        return "".join(sections)

    def _stim(self, content: str) -> Stimulus:
        return Stimulus(
            type="tentacle_feedback",
            source=f"tentacle:{self.name}",
            content=content,
            timestamp=datetime.now(),
            adrenalin=False,
        )


def _format_kb(kb_id: str, entries: list[dict[str, Any]], query: str) -> str:
    lines = [f"KB {kb_id!r} entries for {query!r} — {len(entries)}:"]
    for e in entries:
        content = (e.get("content") or "").strip().replace("\n", " ")
        if len(content) > 200:
            content = content[:197] + "..."
        tags = ", ".join(e.get("tags") or []) or "(no tags)"
        lines.append(f"- [#{e['id']}] (tags: {tags}) {content}")
    return "\n".join(lines)


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


def build_tentacle(ctx) -> Tentacle:
    """Unified-format factory. Pulls gm + embedder + kb_registry from
    ctx.services."""
    return MemoryRecallTentacle(
        gm=ctx.services["gm"],
        embedder=ctx.services["embedder"],
        kb_registry=ctx.services["kb_registry"],
        default_top_k=int(ctx.config.get("default_top_k", 8)),
    )
