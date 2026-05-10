"""Minimal in-memory MemoryService + KBRegistryService — proof of Protocol.

Used by ``test_memory_swap_e2e.py`` to verify the memory slot's
Protocol surface is implementable without a SQLite backing. Real
production backends (Postgres, Redis, Neo4j, ...) would be much
larger but the kind of stubbing here is the same: implement each
Protocol method with whatever your storage primitives provide.

Module-level so importlib can find it via dotted path:
``tests._fake_memory:InMemoryMemoryService``.

Scope:
  * Full GM-side surface (~22 methods) implemented at minimum-viable
    fidelity — vec_search uses brute-force cosine, fts_search does
    substring match, edges are a list of dicts.
  * LLM-driven facades (auto_ingest, explicit_write,
    classify_and_link_pending) are stubbed: they record the call and
    return plausible shapes, no actual LLM calls. A real backend
    would route through the user's classifier/extractor LLMs.
  * KBRegistry-side surface implemented; KB instances use a tiny
    nested dict store.

Not optimized: each upsert scans the full node list. Fine for tests
(few dozen nodes); a real backend would index by name.
"""
from __future__ import annotations

import math
from typing import Any


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class InMemoryMemoryService:
    """Dict-backed MemoryService — enough to satisfy the Protocol and
    serve a runtime smoke test."""

    def __init__(
        self, *,
        db_path: str = ":memory:",
        embedder=None,
        auto_ingest_threshold: float = 0.92,
        extractor_llm=None,
        classifier_llm=None,
    ):
        # All kwargs are accepted (slot contract); only ``embedder``
        # is meaningfully used here.
        self._db_path = db_path
        self._embedder = embedder
        self._threshold = auto_ingest_threshold
        self._nodes: dict[int, dict[str, Any]] = {}
        self._edges: list[dict[str, Any]] = []
        self._next_id = 1
        self.auto_ingest_calls: list[str] = []
        self.explicit_write_calls: list[str] = []

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    # ---- node CRUD -----------------------------------------------

    async def upsert_node(self, node: dict[str, Any]) -> int:
        name = node.get("name", "")
        # Find existing by exact name match.
        for nid, existing in self._nodes.items():
            if existing.get("name") == name:
                existing.update({k: v for k, v in node.items() if k != "id"})
                return nid
        nid = self._next_id
        self._next_id += 1
        self._nodes[nid] = {
            "id": nid,
            "name": name,
            "category": node.get("category", "FACT"),
            "description": node.get("description", ""),
            "source_type": node.get("source_type", "auto"),
            "embedding": node.get("embedding"),
            "importance": node.get("importance", 1.0),
            "metadata": node.get("metadata", {}),
        }
        return nid

    async def find_by_name(self, name: str) -> int | None:
        for nid, n in self._nodes.items():
            if n.get("name") == name:
                return nid
        return None

    async def update_node_category(
        self, node_name: str, new_category: str,
    ) -> bool:
        for n in self._nodes.values():
            if n.get("name") == node_name:
                n["category"] = new_category
                return True
        return False

    async def list_nodes(
        self, *, category: str | None = None, limit: int | None = None,
    ) -> list[dict[str, Any]]:
        out = list(self._nodes.values())
        if category is not None:
            out = [n for n in out if n.get("category") == category]
        if limit is not None:
            out = out[:limit]
        return out

    async def count_nodes(self) -> int:
        return len(self._nodes)

    async def count_by_category(self, category: str) -> int:
        return sum(
            1 for n in self._nodes.values()
            if n.get("category") == category
        )

    async def counts_by_category(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for n in self._nodes.values():
            cat = n.get("category", "?")
            out[cat] = out.get(cat, 0) + 1
        return out

    async def counts_by_source(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for n in self._nodes.values():
            src = n.get("source_type", "?")
            out[src] = out.get(src, 0) + 1
        return out

    async def delete_by_category(self, category: str) -> int:
        removed = [
            nid for nid, n in self._nodes.items()
            if n.get("category") == category
        ]
        for nid in removed:
            del self._nodes[nid]
        return len(removed)

    # ---- edges ----------------------------------------------------

    async def insert_edge_with_cycle_check(
        self, src: int, tgt: int, predicate: str,
    ) -> dict[str, Any]:
        if src == tgt:
            return {"inserted": False, "reason": "self_loop"}
        edge = {"node_a": src, "node_b": tgt, "predicate": predicate}
        self._edges.append(edge)
        return {"inserted": True, **edge}

    async def list_edges_named(
        self, *, limit: int,
    ) -> list[dict[str, str]]:
        out = []
        for e in self._edges[:limit]:
            a = self._nodes.get(e["node_a"], {}).get("name", "?")
            b = self._nodes.get(e["node_b"], {}).get("name", "?")
            out.append({"a": a, "b": b, "predicate": e["predicate"]})
        return out

    async def count_edges(self) -> int:
        return len(self._edges)

    async def get_neighbor_keywords(
        self, node_ids: list[int], *, depth: int = 1,
    ) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {nid: [] for nid in node_ids}
        for nid in node_ids:
            for e in self._edges:
                if e["node_a"] == nid:
                    nb = self._nodes.get(e["node_b"], {}).get("name")
                    if nb:
                        out[nid].append(nb)
                elif e["node_b"] == nid:
                    nb = self._nodes.get(e["node_a"], {}).get("name")
                    if nb:
                        out[nid].append(nb)
        return out

    async def get_edges_among(
        self, node_ids: list[int],
    ) -> list[dict[str, Any]]:
        s = set(node_ids)
        return [
            e for e in self._edges
            if e["node_a"] in s and e["node_b"] in s
        ]

    # ---- search ---------------------------------------------------

    async def vec_search(
        self, query_vec: list[float], *,
        top_k: int = 5, min_similarity: float = 0.0,
    ) -> list[tuple[dict[str, Any], float]]:
        scored = []
        for n in self._nodes.values():
            emb = n.get("embedding")
            if not emb:
                continue
            sim = _cosine(query_vec, emb)
            if sim >= min_similarity:
                scored.append((n, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def fts_search(
        self, query: str, *, top_k: int = 5,
    ) -> list[dict[str, Any]]:
        q = query.lower()
        out = [
            n for n in self._nodes.values()
            if q in n.get("name", "").lower()
            or q in n.get("description", "").lower()
        ]
        return out[:top_k]

    # ---- LLM-driven facades (stubbed) -----------------------------

    async def auto_ingest(
        self, content: str, *, source_heartbeat: int | None = None,
    ) -> dict[str, Any]:
        self.auto_ingest_calls.append(content)
        # Embed (if embedder present) and drop a node — simplest
        # fidelity to the contract.
        emb = None
        if self._embedder is not None:
            try:
                emb = await self._embedder(content)
            except Exception:  # noqa: BLE001
                pass
        nid = await self.upsert_node({
            "name": content[:50],
            "category": "FACT",
            "description": content,
            "source_type": "auto",
            "embedding": emb,
        })
        return {"node_id": nid, "merged": False}

    async def explicit_write(
        self, content: str, *,
        importance: str = "normal",
        recall_context: list[dict[str, Any]] | None = None,
        source_heartbeat: int | None = None,
    ) -> dict[str, Any]:
        self.explicit_write_calls.append(content)
        nid = await self.upsert_node({
            "name": content[:50],
            "category": "FACT",
            "description": content,
            "source_type": "explicit",
        })
        return {"node_id": nid}

    async def classify_and_link_pending(self) -> dict[str, int]:
        # No-op — we have no async classifier; treat every call as
        # "nothing pending."
        return {"classified": 0, "linked": 0}


class _InMemoryKB:
    """KnowledgeBaseLike — minimal append-only store."""

    def __init__(self, kb_id: str, *, embedder=None):
        self.kb_id = kb_id
        self._embedder = embedder
        self._entries: dict[int, dict[str, Any]] = {}
        self._edges: list[dict[str, Any]] = []
        self._next_id = 1

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def write_entry(
        self, content: str, *,
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
        source: str | None = None,
        importance: float = 1.0,
    ) -> int:
        eid = self._next_id
        self._next_id += 1
        self._entries[eid] = {
            "id": eid, "content": content,
            "tags": tags or [], "embedding": embedding,
            "source": source, "importance": importance, "active": True,
        }
        return eid

    async def merge_entry(
        self, entry_id: int, *,
        new_content: str,
        new_embedding: list[float] | None,
        new_importance: float,
        new_tags: list[str] | None,
    ) -> None:
        if entry_id in self._entries:
            self._entries[entry_id].update({
                "content": new_content, "embedding": new_embedding,
                "importance": new_importance, "tags": new_tags or [],
            })

    async def write_edge(
        self, entry_a: int, entry_b: int, predicate: str,
    ) -> dict[str, Any]:
        edge = {"a": entry_a, "b": entry_b, "predicate": predicate}
        self._edges.append(edge)
        return edge

    async def count_entries(self) -> int:
        return len([e for e in self._entries.values() if e.get("active")])

    async def list_active_entries(
        self, *, limit: int,
    ) -> list[dict[str, Any]]:
        return [
            e for e in list(self._entries.values())
            if e.get("active")
        ][:limit]

    async def search(
        self, query: str, *,
        top_k: int = 5, min_similarity: float = 0.3,
    ) -> list[dict[str, Any]]:
        q = query.lower()
        return [
            e for e in self._entries.values()
            if e.get("active") and q in e.get("content", "").lower()
        ][:top_k]

    async def vec_search(
        self, query_vec: list[float], *,
        top_k: int = 5, min_similarity: float = 0.5,
    ) -> list[tuple[dict[str, Any], float]]:
        scored = []
        for e in self._entries.values():
            if not e.get("active"):
                continue
            emb = e.get("embedding")
            if not emb:
                continue
            sim = _cosine(query_vec, emb)
            if sim >= min_similarity:
                scored.append((e, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class InMemoryKBRegistryService:
    """KBRegistryService — dict of kb_id → _InMemoryKB."""

    def __init__(self, *, gm=None, kb_dir: str = "", embedder=None):
        self._gm = gm
        self._embedder = embedder
        self._kbs: dict[str, _InMemoryKB] = {}
        self._meta: dict[str, dict[str, Any]] = {}

    async def create_kb(
        self, kb_id: str, *,
        name: str,
        description: str = "",
        topics: list[str] | None = None,
    ) -> "_InMemoryKB":
        kb = _InMemoryKB(kb_id, embedder=self._embedder)
        await kb.initialize()
        self._kbs[kb_id] = kb
        self._meta[kb_id] = {
            "kb_id": kb_id, "name": name,
            "description": description, "topics": topics or [],
            "archived": False, "index_embedding": None,
        }
        return kb

    async def open_kb(self, kb_id: str) -> "_InMemoryKB":
        if kb_id not in self._kbs:
            raise KeyError(f"KB {kb_id!r} not registered")
        return self._kbs[kb_id]

    async def list_kbs(
        self, *, include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        out = []
        for meta in self._meta.values():
            if not include_archived and meta.get("archived"):
                continue
            out.append(dict(meta))
        return out

    async def set_archived(self, kb_id: str, archived: bool) -> None:
        if kb_id in self._meta:
            self._meta[kb_id]["archived"] = archived

    async def set_index_embedding(
        self, kb_id: str, embedding: list[float] | None,
    ) -> None:
        if kb_id in self._meta:
            self._meta[kb_id]["index_embedding"] = embedding

    async def delete_kb(self, kb_id: str) -> None:
        self._kbs.pop(kb_id, None)
        self._meta.pop(kb_id, None)

    async def close_all(self) -> None:
        for kb in self._kbs.values():
            await kb.close()


# --------------------------------------------------------------------
# InMemoryMemoryEngine — combined class for the new MemoryEngine slot
# --------------------------------------------------------------------


class InMemoryMemoryEngine(InMemoryMemoryService):
    """Combined ``MemoryEngine`` test fake.

    The Engine refactor (2026-05) collapsed the ``memory`` and
    ``kb_registry`` slots into one ``memory`` slot whose Protocol
    surface includes KB management + sleep_cycle. This class extends
    ``InMemoryMemoryService`` (the GM-only fake) with KB delegation
    methods backed by an internal ``InMemoryKBRegistryService`` plus
    a no-op ``sleep_cycle`` stub. Result: ``InMemoryMemoryEngine``
    satisfies ``MemoryEngine`` end-to-end.

    Used by ``test_memory_swap_e2e.py`` to drive the engine slot
    override path.

    The ``sleep_cycle`` stub records its invocation in
    ``self.sleep_cycle_calls`` so tests can assert it ran without
    actually invoking clustering / migration / index-rebuild
    pipelines (those need a real LLM). Returns an empty stats dict.
    """

    def __init__(
        self,
        *,
        db_path: str = ":memory:",
        embedder=None,
        kb_dir: str = "",
        auto_ingest_threshold: float = 0.92,
        extractor_llm=None,
        classifier_llm=None,
    ):
        super().__init__(
            db_path=db_path, embedder=embedder,
            auto_ingest_threshold=auto_ingest_threshold,
            extractor_llm=extractor_llm,
            classifier_llm=classifier_llm,
        )
        self._kb_dir = kb_dir
        self._kb = InMemoryKBRegistryService(
            gm=self, kb_dir=kb_dir, embedder=embedder,
        )
        self.sleep_cycle_calls: list[dict[str, Any]] = []

    # ---- KB management — delegate to the internal registry ----------

    async def create_kb(self, kb_id, *, name, description="", topics=None):
        return await self._kb.create_kb(
            kb_id, name=name, description=description, topics=topics,
        )

    async def open_kb(self, kb_id):
        return await self._kb.open_kb(kb_id)

    async def list_kbs(self, *, include_archived=False):
        return await self._kb.list_kbs(include_archived=include_archived)

    async def set_archived(self, kb_id, archived):
        return await self._kb.set_archived(kb_id, archived)

    async def set_index_embedding(self, kb_id, embedding):
        return await self._kb.set_index_embedding(kb_id, embedding)

    async def delete_kb(self, kb_id):
        return await self._kb.delete_kb(kb_id)

    async def close_all_kbs(self):
        return await self._kb.close_all()

    # ---- sleep — record + return empty stats ------------------------

    async def sleep_cycle(self, *, channels, log_dir, config):
        self.sleep_cycle_calls.append({
            "channels": channels, "log_dir": log_dir, "config": config,
        })
        return {}
