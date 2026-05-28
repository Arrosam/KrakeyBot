"""``MemOSMemoryEngine`` -- optional MemOS (MOS) adapter for the memory slot.

Satisfies ``MemoryEngine`` + ``KnowledgeBaseLike`` Protocols entirely through
the external ``memos`` package. All MemOS calls are synchronous; every async
Protocol method wraps them in ``asyncio.to_thread``.

LAZY IMPORT: This module imports cleanly even when ``memos`` is NOT installed.
All ``memos`` imports are deferred inside ``_new_mos`` and method bodies.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from krakey.interfaces.engines.memory import KnowledgeBaseLike


def _new_mos(*, mos_config_path: str):
    """Lazily import MemOS and return a constructed MOS instance.
    Raises ImportError with an install hint if memos is not installed."""
    try:
        from memos.mem_os.main import MOS
        from memos.configs.mem_os import MOSConfig
    except ImportError as e:
        raise ImportError(
            "MemOS is required for the 'memos' memory engine. "
            "Select core_implementations.memory: memos in config.yaml and "
            "run `krakey install` (or manually: pip install MemoryOS)."
        ) from e
    return MOS(MOSConfig.from_json_file(mos_config_path))


def _item_field(item: Any, name: str, default: Any = None) -> Any:
    """Read a field from an item that may be a dict or an object."""
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _extract_text_memories(result: Any) -> list[Any]:
    """Flatten all memories from every text_mem group in a MOSSearchResult."""
    if result is None:
        return []
    if isinstance(result, dict):
        groups = result.get("text_mem", [])
    else:
        groups = getattr(result, "text_mem", [])
    memories: list[Any] = []
    for group in (groups or []):
        items = _item_field(group, "memories", []) or []
        memories.extend(items)
    return memories


class MemOSKBWrapper:
    """Per-topic MemCube wrapper that satisfies ``KnowledgeBaseLike``."""

    def __init__(
        self,
        *,
        mos: Any,
        kb_id: str,
        user_id: str,
        name: str,
        description: str = "",
        topics: list[str] | None = None,
    ) -> None:
        self._mos = mos
        self.kb_id = kb_id
        self.user_id = user_id
        self.name = name
        self.description = description
        self.topics: list[str] = topics or []
        self._entry_counter: int = 0

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def write_entry(
        self,
        content: str,
        *,
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
        source: str | None = None,
        importance: float = 1.0,
    ) -> int:
        mos = self._mos
        await asyncio.to_thread(
            mos.add,
            memory_content=content,
            mem_cube_id=self.kb_id,
            user_id=self.user_id,
        )
        self._entry_counter += 1
        return self._entry_counter

    async def merge_entry(
        self,
        entry_id: int,
        *,
        new_content: str,
        new_embedding: list[float] | None,
        new_importance: float,
        new_tags: list[str] | None,
    ) -> None:
        mos = self._mos
        try:
            await asyncio.to_thread(
                mos.update,
                self.kb_id,
                str(entry_id),
                new_content,
                user_id=self.user_id,
            )
        except Exception:
            pass

    async def write_edge(
        self, entry_a: int, entry_b: int, predicate: str,
    ) -> dict[str, Any]:
        return {"a": entry_a, "b": entry_b, "predicate": predicate}

    async def count_entries(self) -> int:
        mos = self._mos
        try:
            result = await asyncio.to_thread(
                mos.get_all,
                mem_cube_id=self.kb_id,
                user_id=self.user_id,
            )
            return len(_extract_text_memories(result))
        except Exception:
            return 0

    async def list_active_entries(self, *, limit: int) -> list[dict[str, Any]]:
        mos = self._mos
        try:
            result = await asyncio.to_thread(
                mos.get_all,
                mem_cube_id=self.kb_id,
                user_id=self.user_id,
            )
            items = _extract_text_memories(result)[:limit]
            return [self._to_entry_dict(item) for item in items]
        except Exception:
            return []

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_similarity: float = 0.3,
    ) -> list[dict[str, Any]]:
        mos = self._mos
        result = await asyncio.to_thread(
            mos.search,
            query,
            top_k=top_k,
            install_cube_ids=[self.kb_id],
            user_id=self.user_id,
        )
        return [self._to_entry_dict(item) for item in _extract_text_memories(result)]

    async def vec_search(
        self,
        query_vec: list[float],
        *,
        top_k: int = 5,
        min_similarity: float = 0.5,
    ) -> list[tuple[dict[str, Any], float]]:
        # MemOS retrieves by text only; raw-vector search cannot be honored.
        return []

    def _to_entry_dict(self, item: Any) -> dict[str, Any]:
        text = _item_field(item, "memory", "") or ""
        meta = _item_field(item, "metadata", {}) or {}
        if isinstance(meta, dict):
            source = meta.get("source")
        else:
            source = getattr(meta, "source", None)
        self._entry_counter += 1
        return {
            "id": self._entry_counter,
            "content": text,
            "tags": [],
            "source": source,
            "importance": 1.0,
        }


class MemOSMemoryEngine:
    """``MemoryEngine`` adapter backed by the external MemOS (MOS) library.

    All MemOS calls are synchronous and wrapped in ``asyncio.to_thread``.
    Graph-edge operations, raw-vector search, sleep cycle, and category
    counts are stubbed -- MemOS consolidates memories internally and
    retrieves exclusively by text query.
    """

    def __init__(self, *, config: dict | None = None) -> None:
        cfg = config or {}
        mos_config_path: str = cfg.get("mos_config_path", "")
        if not mos_config_path:
            raise ValueError(
                "MemOSMemoryEngine requires 'mos_config_path' in its config dict. "
                "Set core_implementations.memory.memos.mos_config_path in your "
                "Krakey config."
            )
        self._mos_config_path = mos_config_path
        self._user_id: str = cfg.get("user_id", "krakey")
        self._mem_cube_id: str = cfg.get("mem_cube_id", "krakey_main")
        self._mem_cube_path: str = cfg.get("mem_cube_path", "")
        self._mos: Any = None
        self._id_counter: int = 0
        self._memory_id_map: dict[str, int] = {}
        self._kb_registry: dict[str, MemOSKBWrapper] = {}
        self._kb_meta: dict[str, dict[str, Any]] = {}

    def _local_id(self, mos_id: str) -> int:
        if mos_id in self._memory_id_map:
            return self._memory_id_map[mos_id]
        self._id_counter += 1
        self._memory_id_map[mos_id] = self._id_counter
        return self._id_counter

    def _to_node_dict(self, item: Any) -> dict[str, Any]:
        text = _item_field(item, "memory", "") or ""
        mos_id = str(_item_field(item, "id", "") or "")
        meta = _item_field(item, "metadata", {}) or {}
        if isinstance(meta, dict):
            source = meta.get("source")
            meta_dict = dict(meta)
        else:
            source = getattr(meta, "source", None)
            meta_dict = {}
        name = (text[:40] if text else mos_id) or mos_id
        local_id = self._local_id(mos_id) if mos_id else self._local_id(name)
        return {
            "id": local_id,
            "name": name,
            "description": text,
            "category": "FACT",
            "source_type": source,
            "importance": 1.0,
            "metadata": meta_dict,
        }

    async def initialize(self) -> None:
        self._mos = await asyncio.to_thread(
            _new_mos, mos_config_path=self._mos_config_path
        )
        mos = self._mos
        try:
            await asyncio.to_thread(
                mos.create_user,
                user_id=self._user_id,
                role=None,
                user_name=self._user_id,
            )
        except Exception:
            pass
        cube_name_or_path = self._mem_cube_path or self._mem_cube_id
        try:
            await asyncio.to_thread(
                mos.register_mem_cube,
                cube_name_or_path,
                mem_cube_id=self._mem_cube_id,
                user_id=self._user_id,
            )
        except Exception:
            pass

    async def close(self) -> None:
        return None

    async def upsert_node(self, node: dict[str, Any]) -> int:
        mos = self._mos
        content: str = node.get("description", "") or ""
        if mos is not None and content:
            try:
                await asyncio.to_thread(
                    mos.add,
                    memory_content=content,
                    user_id=self._user_id,
                    mem_cube_id=self._mem_cube_id,
                )
            except Exception:
                pass
        self._id_counter += 1
        return self._id_counter

    async def find_by_name(self, name: str) -> int | None:
        return None

    async def update_node_category(
        self, node_name: str, new_category: str,
    ) -> bool:
        return False

    async def list_nodes(
        self,
        *,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        mos = self._mos
        if mos is None:
            return []
        try:
            result = await asyncio.to_thread(
                mos.get_all,
                mem_cube_id=self._mem_cube_id,
                user_id=self._user_id,
            )
            items = _extract_text_memories(result)
            if limit is not None:
                items = items[:limit]
            return [self._to_node_dict(item) for item in items]
        except Exception:
            return []

    async def count_nodes(self) -> int:
        mos = self._mos
        if mos is None:
            return 0
        try:
            result = await asyncio.to_thread(
                mos.get_all,
                mem_cube_id=self._mem_cube_id,
                user_id=self._user_id,
            )
            return len(_extract_text_memories(result))
        except Exception:
            return 0

    async def count_by_category(self, category: str) -> int:
        return 0

    async def counts_by_category(self) -> dict[str, int]:
        return {}

    async def counts_by_source(self) -> dict[str, int]:
        return {}

    async def delete_by_category(self, category: str) -> int:
        return 0

    async def insert_edge_with_cycle_check(
        self, src: int, tgt: int, predicate: str,
    ) -> dict[str, Any]:
        return {"src": src, "tgt": tgt, "predicate": predicate}

    async def list_edges_named(self, *, limit: int) -> list[dict[str, str]]:
        return []

    async def count_edges(self) -> int:
        return 0

    async def get_neighbor_keywords(
        self, node_ids: list[int], *, depth: int = 1,
    ) -> dict[int, list[str]]:
        return {nid: [] for nid in node_ids}

    async def get_edges_among(
        self, node_ids: list[int],
    ) -> list[dict[str, Any]]:
        return []

    async def vec_search(
        self,
        query_vec: list[float],
        *,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[tuple[dict[str, Any], float]]:
        # MemOS has no raw-vector endpoint; vec_search cannot be honored.
        return []

    async def fts_search(
        self, query: str, *, top_k: int = 5,
    ) -> list[dict[str, Any]]:
        mos = self._mos
        result = await asyncio.to_thread(
            mos.search,
            query,
            top_k=top_k,
            mode="fast",
            user_id=self._user_id,
        )
        return [self._to_node_dict(item) for item in _extract_text_memories(result)]

    async def auto_ingest(
        self, content: str, *, source_heartbeat: int | None = None,
    ) -> dict[str, Any]:
        mos = self._mos
        await asyncio.to_thread(
            mos.add,
            memory_content=content,
            user_id=self._user_id,
            mem_cube_id=self._mem_cube_id,
            session_id=str(source_heartbeat or 0),
        )
        return {"added": 1, "source_heartbeat": source_heartbeat}

    async def explicit_write(
        self,
        content: str,
        *,
        importance: str = "normal",
        recall_context: list[dict[str, Any]] | None = None,
        source_heartbeat: int | None = None,
    ) -> dict[str, Any]:
        mos = self._mos
        await asyncio.to_thread(
            mos.add,
            memory_content=content,
            user_id=self._user_id,
            mem_cube_id=self._mem_cube_id,
        )
        return {"added": 1}

    async def classify_and_link_pending(self) -> dict[str, int]:
        # MemOS consolidates memories internally; no pending queue to classify.
        return {"classified": 0, "edges": 0}

    async def create_kb(
        self,
        kb_id: str,
        *,
        name: str,
        description: str = "",
        topics: list[str] | None = None,
    ) -> "KnowledgeBaseLike":
        mos = self._mos
        try:
            await asyncio.to_thread(
                mos.register_mem_cube,
                kb_id,
                mem_cube_id=kb_id,
                user_id=self._user_id,
            )
        except Exception:
            pass
        wrapper = MemOSKBWrapper(
            mos=mos,
            kb_id=kb_id,
            user_id=self._user_id,
            name=name,
            description=description,
            topics=topics or [],
        )
        self._kb_registry[kb_id] = wrapper
        self._kb_meta[kb_id] = {
            "kb_id": kb_id,
            "name": name,
            "description": description,
            "topics": topics or [],
            "archived": False,
        }
        return wrapper

    async def open_kb(self, kb_id: str) -> "KnowledgeBaseLike":
        if kb_id not in self._kb_registry:
            raise KeyError(f"KB {kb_id!r} not registered")
        return self._kb_registry[kb_id]

    async def list_kbs(
        self, *, include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        result = []
        for meta in self._kb_meta.values():
            if not include_archived and meta.get("archived", False):
                continue
            result.append(dict(meta))
        return result

    async def set_archived(self, kb_id: str, archived: bool) -> None:
        if kb_id in self._kb_meta:
            self._kb_meta[kb_id]["archived"] = archived

    async def set_index_embedding(
        self, kb_id: str, embedding: list[float] | None,
    ) -> None:
        return None

    async def delete_kb(self, kb_id: str) -> None:
        mos = self._mos
        try:
            await asyncio.to_thread(
                mos.delete_all,
                mem_cube_id=kb_id,
                user_id=self._user_id,
            )
        except Exception:
            pass
        self._kb_registry.pop(kb_id, None)
        self._kb_meta.pop(kb_id, None)

    async def close_all_kbs(self) -> None:
        self._kb_registry.clear()

    async def sleep_cycle(
        self,
        *,
        channels: Any,
        log_dir: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        # MemOS consolidates internally; sleep cycle is a no-op here.
        return {}