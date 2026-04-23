"""Built-in `memory_recall` plugin — Self-driven GM + KB exploration.

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
from src.memory.knowledge_base import KBRegistry
from src.models.stimulus import Stimulus


MANIFEST = {
    "name": "memory_recall",
    "description": "Active recall of GM nodes (and, via KB index nodes, "
                   "their KB entries). Self dispatches this when she "
                   "wants to reflect or dig into past learning.",
    "is_internal": True,
    "config_schema": [
        {"field": "default_top_k", "type": "number", "default": 8,
         "help": "Default number of top-K nodes to return when Self "
                 "doesn't specify."},
    ],
}


class AsyncEmbedder(Protocol):
    async def __call__(self, text: str) -> list[float]: ...


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

    @property
    def sandboxed(self) -> bool:
        return True

    @property
    def is_internal(self) -> bool:
        # Recall results feed Self's next heartbeat; never reach the user.
        return True

    async def execute(self, intent: str,
                        params: dict[str, Any]) -> Stimulus:
        assert self._gm is not None and self._embedder is not None
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

        # Normal path: GM vec → FTS fallback
        nodes = await self._search_gm(query, top_k)
        if not nodes:
            return self._stim(
                f"No matching memories for query: {query!r}.",
            )

        node_ids = [n["id"] for n in nodes]
        neighbor_map = await self._gm.get_neighbor_keywords(node_ids)
        edges = await self._gm.get_edges_among(node_ids)

        # KB index expansion: if any recalled node is a KB index, also
        # pull entries from that KB.
        kb_sections = await self._expand_kb_indexes(nodes, query, top_k)

        return self._stim(_format(nodes, neighbor_map, edges, query)
                           + kb_sections)

    async def _search_gm(self, query: str,
                          top_k: int) -> list[dict[str, Any]]:
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


def create_tentacle(config: dict, deps: dict) -> Tentacle:
    return MemoryRecallTentacle(
        gm=deps["gm"],
        embedder=deps["embedder"],
        kb_registry=deps["kb_registry"],
        default_top_k=int(config.get("default_top_k", 8)),
    )
