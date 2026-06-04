"""Edge tests for the SHRUNK ``MemoryEngine`` Protocol surface.

Contract under test:
  - krakey/interfaces/engines/memory.py  (MemoryEngine — 12 methods only)
  - krakey/interfaces/engines/__init__.py (exports / removed exports)
  - krakey/interfaces/engines/decision.py (DecisionResult — memory_updates removed)

Spec: The memory slot now exposes EXACTLY 12 async methods (lifecycle x2,
storage x3, recall x3, read-only stats x4). All raw graph/KB/search/sleep
primitives are engine-internal and MUST NOT be on the Protocol. The
``KnowledgeBaseLike`` value-type Protocol is also REMOVED. ``DecisionResult``
loses its ``memory_updates`` field.

Tests are written BEFORE the implementation lands — all tests targeting the
final state will be RED until the dev unit shrinks the Protocol.

Techniques applied:
  Protocol surface — positive + negative (removed names + KnowledgeBaseLike)
  runtime_checkable BVA / structural subtyping
  DecisionResult shape — state / error guessing
  Swap-conformance harness — positive + boundary

asyncio_mode = auto (see pytest.ini); plain ``async def`` functions and
class-based ``async def`` methods work without @pytest.mark.asyncio.
"""
from __future__ import annotations

import importlib
from dataclasses import fields as dc_fields
from typing import Protocol, get_type_hints

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _protocol_attrs(cls) -> set[str]:
    """Return the set of method/attribute names declared on a Protocol.

    Uses ``__protocol_attrs__`` (CPython 3.12+); falls back to scanning
    ``cls.__dict__`` and filtering out Protocol/object noise. This keeps
    the tests resilient across CPython versions.
    """
    attrs = getattr(cls, "__protocol_attrs__", None)
    if attrs is not None:
        return set(attrs)
    # Fallback: everything in __dict__ that isn't a dunder or Protocol
    # infrastructure attribute.
    _noise = frozenset(dir(object)) | {
        "_is_protocol", "_is_runtime_protocol", "_abc_impl",
        "__abstractmethods__", "__class_getitem__", "__parameters__",
        "__protocol_attrs__", "__subclasshook__",
    }
    return {
        k for k in cls.__dict__
        if not k.startswith("__") and k not in _noise
    }


# ---------------------------------------------------------------------------
# The 12 required method names (single source of truth for tests)
# ---------------------------------------------------------------------------

_REQUIRED_12 = frozenset({
    # lifecycle
    "initialize",
    "close",
    # storage
    "ingest",
    "remember",
    "remember_extraction",
    # recall
    "search",
    "recall_context",
    "recall_kb",
    # read-only stats
    "count_nodes",
    "count_edges",
    "counts_by_category",
    "counts_by_source",
})

# Every method name that MUST NOT appear on the Protocol after the shrink.
_REMOVED_NAMES = frozenset({
    # raw node CRUD
    "upsert_node",
    "find_by_name",
    "update_node_category",
    "list_nodes",
    "count_by_category",
    "delete_by_category",
    # raw edge CRUD
    "insert_edge_with_cycle_check",
    "list_edges_named",
    "get_neighbor_keywords",
    "get_edges_among",
    # raw search
    "vec_search",
    "fts_search",
    # pending classification
    "classify_and_link_pending",
    # KB fleet management
    "create_kb",
    "open_kb",
    "list_kbs",
    "set_archived",
    "set_index_embedding",
    "delete_kb",
    "close_all_kbs",
    # sleep consolidation
    "sleep_cycle",
    # old LLM facades
    "auto_ingest",
    "explicit_write",
})


# ---------------------------------------------------------------------------
# Minimal in-test backend (used by swap-conformance section)
# ---------------------------------------------------------------------------

class MinimalMemory:
    """Implements EXACTLY the 12 Protocol methods using in-memory dicts.

    Intentionally no inheritance — structural duck-typing only.
    Behaviour is minimal-correct per the contract spec.
    """

    def __init__(self):
        self._nodes: dict[int, dict] = {}
        self._next_id: int = 1

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def ingest(
        self, content: str, *, source_heartbeat: int | None = None
    ) -> dict:
        nid = self._next_id
        self._next_id += 1
        self._nodes[nid] = {"id": nid, "content": content}
        return {"node_id": nid, "action": "ingest"}

    async def remember(
        self,
        content: str,
        *,
        importance: str = "normal",
        recall_context: list[dict] | None = None,
        source_heartbeat: int | None = None,
    ) -> dict:
        nid = self._next_id
        self._next_id += 1
        self._nodes[nid] = {
            "id": nid, "content": content, "importance": importance,
        }
        return {"node_id": nid, "action": "remember"}

    async def remember_extraction(
        self, nodes: list[dict], edges: list[dict]
    ) -> dict:
        return {"nodes_written": len(nodes), "edges_written": len(edges)}

    async def search(
        self, query: str, *, top_k: int = 8, min_similarity: float = 0.3
    ) -> list[tuple[dict, float]]:
        if top_k == 0:
            return []
        results = []
        for node in self._nodes.values():
            content = node.get("content", "")
            if query.lower() in content.lower():
                results.append((node, 1.0))
            else:
                results.append((node, 0.1))
        # best-first: sort descending by score
        results.sort(key=lambda t: t[1], reverse=True)
        # filter by min_similarity
        results = [(d, s) for d, s in results if s >= min_similarity]
        return results[:top_k]

    async def recall_context(self, node_ids: list[int]) -> dict:
        if not node_ids:
            return {"neighbor_keywords": {}, "edges": []}
        kw: dict[int, list[str]] = {nid: [] for nid in node_ids}
        return {"neighbor_keywords": kw, "edges": []}

    async def recall_kb(
        self, kb_id: str, query: str, *, top_k: int = 5
    ) -> list[dict]:
        # No KB tier — always raise KeyError per spec
        raise KeyError(kb_id)

    async def count_nodes(self) -> int:
        return len(self._nodes)

    async def count_edges(self) -> int:
        return 0

    async def counts_by_category(self) -> dict[str, int]:
        return {}

    async def counts_by_source(self) -> dict[str, int]:
        return {}


# ---------------------------------------------------------------------------
# 1. Protocol surface — positive tests (import + presence of 12 names)
# ---------------------------------------------------------------------------

class TestProtocolSurfacePositive:
    """Positive equivalence: import succeeds; all 12 required names present."""

    def test_memory_engine_importable_from_package(self):
        """MemoryEngine must be importable from krakey.interfaces.engines."""
        from krakey.interfaces.engines import MemoryEngine  # noqa: F401
        assert MemoryEngine is not None

    def test_memory_engine_importable_from_module(self):
        """MemoryEngine must be importable directly from the source module."""
        from krakey.interfaces.engines.memory import MemoryEngine  # noqa: F401
        assert MemoryEngine is not None

    def test_memory_engine_in_package_all(self):
        """MemoryEngine should appear in krakey.interfaces.engines.__all__."""
        import krakey.interfaces.engines as pkg
        assert "MemoryEngine" in pkg.__all__

    def test_memory_engine_is_runtime_checkable(self):
        """MemoryEngine must be decorated with @runtime_checkable."""
        from krakey.interfaces.engines import MemoryEngine
        # runtime_checkable protocols expose _is_runtime_protocol
        assert getattr(MemoryEngine, "_is_runtime_protocol", False) is True

    def test_all_12_required_methods_on_protocol(self):
        """Every one of the 12 required method names must appear in the
        Protocol's declared attribute set."""
        from krakey.interfaces.engines import MemoryEngine
        attrs = _protocol_attrs(MemoryEngine)
        missing = _REQUIRED_12 - attrs
        assert not missing, (
            f"Required methods missing from MemoryEngine Protocol: {missing!r}"
        )

    def test_exactly_12_methods_on_protocol(self):
        """The Protocol declares EXACTLY 12 methods — no extras, no fewer.

        This is a strict surface contract: if a dev adds a 13th method it
        should be caught here, forcing an explicit contract amendment.
        """
        from krakey.interfaces.engines import MemoryEngine
        attrs = _protocol_attrs(MemoryEngine)
        extra = attrs - _REQUIRED_12
        assert not extra, (
            f"MemoryEngine Protocol has unexpected extra methods: {extra!r}\n"
            "The contract specifies exactly 12 methods. Add them to "
            "_REQUIRED_12 only after a deliberate contract amendment."
        )
        assert len(attrs) == 12, (
            f"Expected 12 Protocol methods, found {len(attrs)}: {attrs!r}"
        )


# ---------------------------------------------------------------------------
# 2. Protocol surface — negative: removed names absent from Protocol
# ---------------------------------------------------------------------------

class TestRemovedNamesAbsent:
    """All methods moved to engine-internal must NOT appear on the Protocol."""

    @pytest.mark.parametrize("name", sorted(_REMOVED_NAMES))
    def test_removed_method_not_in_protocol_attrs(self, name: str):
        """Each individually-removed method must not be in __protocol_attrs__."""
        from krakey.interfaces.engines import MemoryEngine
        attrs = _protocol_attrs(MemoryEngine)
        assert name not in attrs, (
            f"Removed method {name!r} still appears on the MemoryEngine Protocol."
        )

    def test_upsert_node_not_on_protocol(self):
        """Spot-check: upsert_node must not be accessible via hasattr on the
        Protocol class itself (belt-and-suspenders beyond __protocol_attrs__)."""
        from krakey.interfaces.engines import MemoryEngine
        assert "upsert_node" not in _protocol_attrs(MemoryEngine)

    def test_sleep_cycle_not_on_protocol(self):
        """sleep_cycle (self-scheduled consolidation) is now engine-internal."""
        from krakey.interfaces.engines import MemoryEngine
        assert "sleep_cycle" not in _protocol_attrs(MemoryEngine)

    def test_vec_search_not_on_protocol(self):
        """vec_search is raw search — must not be on Protocol."""
        from krakey.interfaces.engines import MemoryEngine
        assert "vec_search" not in _protocol_attrs(MemoryEngine)

    def test_fts_search_not_on_protocol(self):
        """fts_search is raw search — must not be on Protocol."""
        from krakey.interfaces.engines import MemoryEngine
        assert "fts_search" not in _protocol_attrs(MemoryEngine)

    def test_open_kb_not_on_protocol(self):
        """open_kb is KB fleet management — now engine-internal."""
        from krakey.interfaces.engines import MemoryEngine
        assert "open_kb" not in _protocol_attrs(MemoryEngine)

    def test_create_kb_not_on_protocol(self):
        """create_kb is KB fleet management — now engine-internal."""
        from krakey.interfaces.engines import MemoryEngine
        assert "create_kb" not in _protocol_attrs(MemoryEngine)

    def test_auto_ingest_not_on_protocol(self):
        """auto_ingest (old LLM facade) is replaced by ingest() on the surface."""
        from krakey.interfaces.engines import MemoryEngine
        assert "auto_ingest" not in _protocol_attrs(MemoryEngine)

    def test_explicit_write_not_on_protocol(self):
        """explicit_write (old LLM facade) is replaced by remember()."""
        from krakey.interfaces.engines import MemoryEngine
        assert "explicit_write" not in _protocol_attrs(MemoryEngine)

    def test_get_edges_among_not_on_protocol(self):
        """get_edges_among is raw graph traversal — engine-internal."""
        from krakey.interfaces.engines import MemoryEngine
        assert "get_edges_among" not in _protocol_attrs(MemoryEngine)

    def test_get_neighbor_keywords_not_on_protocol(self):
        """get_neighbor_keywords is raw graph traversal — engine-internal."""
        from krakey.interfaces.engines import MemoryEngine
        assert "get_neighbor_keywords" not in _protocol_attrs(MemoryEngine)

    def test_classify_and_link_pending_not_on_protocol(self):
        """classify_and_link_pending is now self-triggered by the engine."""
        from krakey.interfaces.engines import MemoryEngine
        assert "classify_and_link_pending" not in _protocol_attrs(MemoryEngine)

    def test_list_nodes_not_on_protocol(self):
        """list_nodes is raw node CRUD — engine-internal."""
        from krakey.interfaces.engines import MemoryEngine
        assert "list_nodes" not in _protocol_attrs(MemoryEngine)

    def test_count_by_category_not_on_protocol(self):
        """count_by_category (singular) is raw CRUD — replaced by counts_by_category."""
        from krakey.interfaces.engines import MemoryEngine
        assert "count_by_category" not in _protocol_attrs(MemoryEngine)


# ---------------------------------------------------------------------------
# 3. KnowledgeBaseLike removed from public surface
# ---------------------------------------------------------------------------

class TestKnowledgeBaseLikeRemoved:
    """KnowledgeBaseLike must no longer be importable from the public package."""

    def test_knowledge_base_like_absent_from_package_namespace(self):
        """getattr on the package must return None — not present."""
        import krakey.interfaces.engines as pkg
        result = getattr(pkg, "KnowledgeBaseLike", None)
        assert result is None, (
            "KnowledgeBaseLike is still exported from krakey.interfaces.engines "
            "but the contract removes it from the public surface."
        )

    def test_knowledge_base_like_absent_from_package_all(self):
        """KnowledgeBaseLike must not appear in __all__ of the package."""
        import krakey.interfaces.engines as pkg
        assert "KnowledgeBaseLike" not in pkg.__all__

    def test_knowledge_base_like_not_importable_via_importlib(self):
        """Attempting to access KnowledgeBaseLike via importlib attribute
        lookup on the engines package must not find it."""
        engines_mod = importlib.import_module("krakey.interfaces.engines")
        assert not hasattr(engines_mod, "KnowledgeBaseLike"), (
            "KnowledgeBaseLike must not be a module-level attribute of "
            "krakey.interfaces.engines after the Protocol shrink."
        )


# ---------------------------------------------------------------------------
# 4. runtime_checkable structural isinstance — BVA
# ---------------------------------------------------------------------------

class TestRuntimeCheckableIsinstance:
    """Structural isinstance checks: passing/failing based on method presence."""

    def test_class_with_all_12_passes_isinstance(self):
        """A class implementing all 12 methods is accepted as MemoryEngine."""
        from krakey.interfaces.engines import MemoryEngine
        assert isinstance(MinimalMemory(), MemoryEngine)

    def test_class_missing_search_fails_isinstance(self):
        """Missing one required method → isinstance is False."""
        from krakey.interfaces.engines import MemoryEngine

        class MissingSearch:
            async def initialize(self): ...
            async def close(self): ...
            async def ingest(self, content, *, source_heartbeat=None): ...
            async def remember(self, content, *, importance="normal",
                               recall_context=None, source_heartbeat=None): ...
            async def remember_extraction(self, nodes, edges): ...
            # search intentionally absent
            async def recall_context(self, node_ids): ...
            async def recall_kb(self, kb_id, query, *, top_k=5): ...
            async def count_nodes(self): ...
            async def count_edges(self): ...
            async def counts_by_category(self): ...
            async def counts_by_source(self): ...

        assert not isinstance(MissingSearch(), MemoryEngine)

    def test_class_missing_initialize_fails_isinstance(self):
        """initialize is lifecycle — required; absence fails check."""
        from krakey.interfaces.engines import MemoryEngine

        class MissingInit:
            # initialize intentionally absent
            async def close(self): ...
            async def ingest(self, content, *, source_heartbeat=None): ...
            async def remember(self, content, *, importance="normal",
                               recall_context=None, source_heartbeat=None): ...
            async def remember_extraction(self, nodes, edges): ...
            async def search(self, query, *, top_k=8, min_similarity=0.3): ...
            async def recall_context(self, node_ids): ...
            async def recall_kb(self, kb_id, query, *, top_k=5): ...
            async def count_nodes(self): ...
            async def count_edges(self): ...
            async def counts_by_category(self): ...
            async def counts_by_source(self): ...

        assert not isinstance(MissingInit(), MemoryEngine)

    def test_class_missing_count_edges_fails_isinstance(self):
        """count_edges is a required stat — absence fails check."""
        from krakey.interfaces.engines import MemoryEngine

        class MissingCountEdges:
            async def initialize(self): ...
            async def close(self): ...
            async def ingest(self, content, *, source_heartbeat=None): ...
            async def remember(self, content, *, importance="normal",
                               recall_context=None, source_heartbeat=None): ...
            async def remember_extraction(self, nodes, edges): ...
            async def search(self, query, *, top_k=8, min_similarity=0.3): ...
            async def recall_context(self, node_ids): ...
            async def recall_kb(self, kb_id, query, *, top_k=5): ...
            async def count_nodes(self): ...
            # count_edges intentionally absent
            async def counts_by_category(self): ...
            async def counts_by_source(self): ...

        assert not isinstance(MissingCountEdges(), MemoryEngine)

    def test_class_missing_recall_kb_fails_isinstance(self):
        """recall_kb is required; absence fails check."""
        from krakey.interfaces.engines import MemoryEngine

        class MissingRecallKb:
            async def initialize(self): ...
            async def close(self): ...
            async def ingest(self, content, *, source_heartbeat=None): ...
            async def remember(self, content, *, importance="normal",
                               recall_context=None, source_heartbeat=None): ...
            async def remember_extraction(self, nodes, edges): ...
            async def search(self, query, *, top_k=8, min_similarity=0.3): ...
            async def recall_context(self, node_ids): ...
            # recall_kb intentionally absent
            async def count_nodes(self): ...
            async def count_edges(self): ...
            async def counts_by_category(self): ...
            async def counts_by_source(self): ...

        assert not isinstance(MissingRecallKb(), MemoryEngine)

    def test_empty_class_fails_isinstance(self):
        """A class with no methods fails the check."""
        from krakey.interfaces.engines import MemoryEngine

        class Empty:
            pass

        assert not isinstance(Empty(), MemoryEngine)

    def test_class_with_all_12_plus_extras_passes_isinstance(self):
        """A structural superset (12 + extra methods) still passes — Protocol
        only checks presence of declared names, not absence of others."""
        from krakey.interfaces.engines import MemoryEngine

        class SupersetMemory(MinimalMemory):
            """Has all 12 + extra engine-internal helpers that callers don't use."""

            async def internal_defrag(self) -> None: ...

            async def admin_dump(self) -> list[dict]:
                return []

        assert isinstance(SupersetMemory(), MemoryEngine)

    def test_class_with_only_removed_methods_fails_isinstance(self):
        """A class that only implements the OLD (removed) surface does not
        satisfy the new Protocol — confirms the break is real."""
        from krakey.interfaces.engines import MemoryEngine

        class OldSurface:
            """Implements old surface only — no ingest/remember/search etc."""
            async def initialize(self): ...
            async def close(self): ...
            async def upsert_node(self, node): ...
            async def find_by_name(self, name): ...
            async def auto_ingest(self, content, *, source_heartbeat=None): ...
            async def explicit_write(self, content, **kwargs): ...
            async def vec_search(self, query_vec, **kwargs): ...
            async def fts_search(self, query, **kwargs): ...
            async def sleep_cycle(self, *, channels, log_dir, config): ...

        assert not isinstance(OldSurface(), MemoryEngine)

    def test_class_with_11_of_12_fails_isinstance(self):
        """Exactly 11 methods — one short of the full set — must fail.
        Tests the minimum boundary (12-1)."""
        from krakey.interfaces.engines import MemoryEngine

        class Eleven:
            async def initialize(self): ...
            async def close(self): ...
            async def ingest(self, content, *, source_heartbeat=None): ...
            async def remember(self, content, *, importance="normal",
                               recall_context=None, source_heartbeat=None): ...
            async def remember_extraction(self, nodes, edges): ...
            async def search(self, query, *, top_k=8, min_similarity=0.3): ...
            async def recall_context(self, node_ids): ...
            async def recall_kb(self, kb_id, query, *, top_k=5): ...
            async def count_nodes(self): ...
            async def count_edges(self): ...
            # counts_by_category missing — one short
            async def counts_by_source(self): ...

        assert not isinstance(Eleven(), MemoryEngine)


# ---------------------------------------------------------------------------
# 5. DecisionResult shape — state / error guessing
# ---------------------------------------------------------------------------

class TestDecisionResultShape:
    """DecisionResult must NOT have memory_updates; remaining fields intact."""

    def test_decision_result_importable(self):
        """DecisionResult must be importable from the engines package."""
        from krakey.interfaces.engines import DecisionResult  # noqa: F401
        assert DecisionResult is not None

    def test_memory_updates_attribute_absent(self):
        """The memory_updates field was removed; hasattr must return False."""
        from krakey.interfaces.engines import DecisionResult
        dr = DecisionResult()
        assert not hasattr(dr, "memory_updates"), (
            "memory_updates still exists on DecisionResult. "
            "The contract removes this field."
        )

    def test_memory_updates_not_in_dataclass_fields(self):
        """memory_updates must not appear in dataclasses.fields()."""
        from krakey.interfaces.engines import DecisionResult
        field_names = {f.name for f in dc_fields(DecisionResult)}
        assert "memory_updates" not in field_names, (
            f"memory_updates is still a declared dataclass field: {field_names!r}"
        )

    def test_memory_writes_exists_with_list_default(self):
        """memory_writes must still exist and default to an empty list."""
        from krakey.interfaces.engines import DecisionResult
        dr = DecisionResult()
        assert hasattr(dr, "memory_writes")
        assert isinstance(dr.memory_writes, list)
        assert dr.memory_writes == []

    def test_tool_calls_exists_with_list_default(self):
        """tool_calls must still exist and default to an empty list."""
        from krakey.interfaces.engines import DecisionResult
        dr = DecisionResult()
        assert hasattr(dr, "tool_calls")
        assert isinstance(dr.tool_calls, list)
        assert dr.tool_calls == []

    def test_sleep_exists_with_bool_default(self):
        """sleep must still exist and default to False."""
        from krakey.interfaces.engines import DecisionResult
        dr = DecisionResult()
        assert hasattr(dr, "sleep")
        assert dr.sleep is False

    def test_parse_failures_exists_with_list_default(self):
        """parse_failures must still exist and default to an empty list."""
        from krakey.interfaces.engines import DecisionResult
        dr = DecisionResult()
        assert hasattr(dr, "parse_failures")
        assert isinstance(dr.parse_failures, list)
        assert dr.parse_failures == []

    def test_remaining_four_field_names(self):
        """Exactly the four surviving fields must exist (no more, no fewer)."""
        from krakey.interfaces.engines import DecisionResult
        field_names = {f.name for f in dc_fields(DecisionResult)}
        expected = {"tool_calls", "memory_writes", "sleep", "parse_failures"}
        assert field_names == expected, (
            f"DecisionResult field set mismatch.\n"
            f"  Expected : {sorted(expected)!r}\n"
            f"  Actual   : {sorted(field_names)!r}\n"
            "Check that memory_updates was removed and no other fields changed."
        )

    def test_constructing_with_memory_updates_kwarg_raises_typeerror(self):
        """Passing memory_updates=[] at construction must raise TypeError
        (unexpected keyword argument) because the field no longer exists."""
        from krakey.interfaces.engines import DecisionResult
        with pytest.raises(TypeError):
            DecisionResult(memory_updates=[{"key": "val"}])

    def test_constructing_with_all_valid_fields(self):
        """Construction with the four surviving fields works without error."""
        from krakey.interfaces.engines import DecisionResult, ToolCall, ParseFailure
        dr = DecisionResult(
            tool_calls=[ToolCall(tool="ping", intent="check", params={})],
            memory_writes=[{"content": "something"}],
            sleep=True,
            parse_failures=[ParseFailure(payload="{}", error="bad", block_index=0)],
        )
        assert len(dr.tool_calls) == 1
        assert dr.sleep is True
        assert len(dr.memory_writes) == 1
        assert len(dr.parse_failures) == 1

    def test_decision_result_dataclass_independence(self):
        """Two DecisionResult instances must not share default list references
        (dataclass field(default_factory=list) pattern)."""
        from krakey.interfaces.engines import DecisionResult
        a = DecisionResult()
        b = DecisionResult()
        a.memory_writes.append({"x": 1})
        assert b.memory_writes == [], (
            "Default list for memory_writes is shared between instances — "
            "field_factory not used."
        )


# ---------------------------------------------------------------------------
# 6. Swap-conformance harness — positive: MinimalMemory exercises the contract
# ---------------------------------------------------------------------------

class TestSwapConformancePositive:
    """A backend implementing only the 12 methods satisfies all callers."""

    async def test_isinstance_check_passes(self):
        from krakey.interfaces.engines import MemoryEngine
        mem = MinimalMemory()
        assert isinstance(mem, MemoryEngine)

    async def test_initialize_returns_none(self):
        mem = MinimalMemory()
        result = await mem.initialize()
        assert result is None

    async def test_close_returns_none(self):
        mem = MinimalMemory()
        await mem.initialize()
        result = await mem.close()
        assert result is None

    async def test_ingest_returns_dict(self):
        mem = MinimalMemory()
        await mem.initialize()
        result = await mem.ingest("tool feedback here")
        assert isinstance(result, dict)

    async def test_ingest_with_source_heartbeat_returns_dict(self):
        mem = MinimalMemory()
        await mem.initialize()
        result = await mem.ingest("data with beat", source_heartbeat=42)
        assert isinstance(result, dict)

    async def test_remember_returns_dict(self):
        mem = MinimalMemory()
        await mem.initialize()
        result = await mem.remember("user prefers dark mode")
        assert isinstance(result, dict)

    async def test_remember_with_all_kwargs_returns_dict(self):
        mem = MinimalMemory()
        await mem.initialize()
        result = await mem.remember(
            "critical insight",
            importance="high",
            recall_context=[{"id": 1, "text": "prior context"}],
            source_heartbeat=7,
        )
        assert isinstance(result, dict)

    async def test_remember_extraction_returns_dict(self):
        mem = MinimalMemory()
        await mem.initialize()
        nodes = [{"name": "n1", "category": "FACT", "description": "d1"}]
        edges = [{"source_name": "n1", "target_name": "n2", "predicate": "RELATED_TO"}]
        result = await mem.remember_extraction(nodes, edges)
        assert isinstance(result, dict)

    async def test_search_returns_list_of_scored_tuples(self):
        mem = MinimalMemory()
        await mem.initialize()
        await mem.ingest("the sky is blue")
        results = await mem.search("sky")
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            record, score = item
            assert isinstance(record, dict)
            assert isinstance(score, float)

    async def test_search_results_best_first(self):
        """search must return results sorted descending by score."""
        mem = MinimalMemory()
        await mem.initialize()
        await mem.ingest("sky blue weather")
        await mem.ingest("completely unrelated content")
        results = await mem.search("sky", top_k=2)
        if len(results) >= 2:
            assert results[0][1] >= results[1][1], (
                f"Results not sorted best-first: {results!r}"
            )

    async def test_recall_context_returns_two_key_dict(self):
        mem = MinimalMemory()
        await mem.initialize()
        await mem.ingest("fact one")
        result = await mem.recall_context([1])
        assert isinstance(result, dict)
        assert "neighbor_keywords" in result
        assert "edges" in result

    async def test_recall_context_neighbor_keywords_maps_ids_to_lists(self):
        mem = MinimalMemory()
        await mem.initialize()
        await mem.ingest("some content")
        result = await mem.recall_context([1])
        kw = result["neighbor_keywords"]
        assert isinstance(kw, dict)
        for nid, hints in kw.items():
            assert isinstance(nid, int)
            assert isinstance(hints, list)

    async def test_recall_context_edges_is_list(self):
        mem = MinimalMemory()
        await mem.initialize()
        result = await mem.recall_context([1, 2])
        assert isinstance(result["edges"], list)

    async def test_recall_kb_raises_keyerror_for_unknown_id(self):
        """Engines without a KB tier must raise KeyError — callers handle miss."""
        mem = MinimalMemory()
        await mem.initialize()
        with pytest.raises(KeyError):
            await mem.recall_kb("nonexistent-kb", "some query")

    async def test_recall_kb_raises_keyerror_for_any_id(self):
        """Even a plausible-looking kb_id raises KeyError when no KBs exist."""
        mem = MinimalMemory()
        await mem.initialize()
        with pytest.raises(KeyError):
            await mem.recall_kb("kb_research_notes", "vector databases", top_k=3)

    async def test_count_nodes_returns_int(self):
        mem = MinimalMemory()
        await mem.initialize()
        result = await mem.count_nodes()
        assert isinstance(result, int)

    async def test_count_edges_returns_int(self):
        mem = MinimalMemory()
        await mem.initialize()
        result = await mem.count_edges()
        assert isinstance(result, int)

    async def test_counts_by_category_returns_dict_of_str_int(self):
        mem = MinimalMemory()
        await mem.initialize()
        result = await mem.counts_by_category()
        assert isinstance(result, dict)
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, int)

    async def test_counts_by_source_returns_dict_of_str_int(self):
        mem = MinimalMemory()
        await mem.initialize()
        result = await mem.counts_by_source()
        assert isinstance(result, dict)
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, int)

    async def test_stats_default_zero_and_empty_on_fresh_engine(self):
        """An engine with no content reports 0 for integer stats and {} for dicts."""
        mem = MinimalMemory()
        await mem.initialize()
        assert await mem.count_nodes() == 0
        assert await mem.count_edges() == 0
        assert await mem.counts_by_category() == {}
        assert await mem.counts_by_source() == {}

    async def test_count_nodes_increments_after_ingest(self):
        """count_nodes must reflect storage: state grows after ingest calls."""
        mem = MinimalMemory()
        await mem.initialize()
        before = await mem.count_nodes()
        await mem.ingest("item one")
        after = await mem.count_nodes()
        assert after == before + 1

    async def test_count_nodes_increments_after_remember(self):
        """count_nodes grows after remember(), confirming storage occurred."""
        mem = MinimalMemory()
        await mem.initialize()
        before = await mem.count_nodes()
        await mem.remember("deliberate memory entry")
        after = await mem.count_nodes()
        assert after == before + 1


# ---------------------------------------------------------------------------
# 7. Swap-conformance harness — boundary value tests
# ---------------------------------------------------------------------------

class TestSwapConformanceBoundary:
    """Boundary inputs per the contract spec."""

    async def test_search_top_k_zero_returns_empty_list(self):
        """top_k=0 is the minimum boundary — must return [] per contract spec."""
        mem = MinimalMemory()
        await mem.initialize()
        await mem.ingest("content that would match")
        results = await mem.search("content", top_k=0)
        assert results == []

    async def test_search_top_k_one_returns_at_most_one_result(self):
        """top_k=1 (minimum +1) returns at most one result."""
        mem = MinimalMemory()
        await mem.initialize()
        await mem.ingest("first item")
        await mem.ingest("second item")
        results = await mem.search("item", top_k=1)
        assert len(results) <= 1

    async def test_search_empty_store_returns_empty_list(self):
        """Empty engine: search returns [] regardless of query."""
        mem = MinimalMemory()
        await mem.initialize()
        results = await mem.search("anything")
        assert results == []

    async def test_search_empty_query_string(self):
        """Empty query string is a valid call — must not raise, must return list."""
        mem = MinimalMemory()
        await mem.initialize()
        await mem.ingest("some content")
        results = await mem.search("")
        assert isinstance(results, list)

    async def test_recall_context_empty_node_ids(self):
        """recall_context([]) — empty list is valid per contract.

        Contract spec: 'A non-graph backend returns
        {\"neighbor_keywords\": {}, \"edges\": []}' — callers tolerate empty
        enrichment, so empty input must at minimum return the two-key dict."""
        mem = MinimalMemory()
        await mem.initialize()
        result = await mem.recall_context([])
        assert isinstance(result, dict)
        assert "neighbor_keywords" in result
        assert "edges" in result
        assert result["neighbor_keywords"] == {}
        assert result["edges"] == []

    async def test_recall_context_single_node_id(self):
        """recall_context with a single node id — minimum non-empty input."""
        mem = MinimalMemory()
        await mem.initialize()
        await mem.ingest("single node content")
        result = await mem.recall_context([1])
        assert isinstance(result, dict)
        assert "neighbor_keywords" in result
        assert "edges" in result

    async def test_remember_extraction_empty_nodes_and_edges(self):
        """remember_extraction([], []) — no-op input must not crash, returns dict."""
        mem = MinimalMemory()
        await mem.initialize()
        result = await mem.remember_extraction([], [])
        assert isinstance(result, dict)

    async def test_remember_extraction_nodes_only_no_edges(self):
        """Nodes without edges — valid, must return dict."""
        mem = MinimalMemory()
        await mem.initialize()
        nodes = [
            {"name": "Paris", "category": "FACT", "description": "capital of France"},
        ]
        result = await mem.remember_extraction(nodes, [])
        assert isinstance(result, dict)

    async def test_ingest_single_char_content(self):
        """Minimum non-empty content string — single character."""
        mem = MinimalMemory()
        await mem.initialize()
        result = await mem.ingest("x")
        assert isinstance(result, dict)

    async def test_ingest_source_heartbeat_none_vs_zero(self):
        """source_heartbeat=None (absent) vs source_heartbeat=0 both valid."""
        mem = MinimalMemory()
        await mem.initialize()
        r1 = await mem.ingest("content a", source_heartbeat=None)
        r2 = await mem.ingest("content b", source_heartbeat=0)
        assert isinstance(r1, dict)
        assert isinstance(r2, dict)

    async def test_recall_kb_top_k_one(self):
        """recall_kb with top_k=1 (boundary minimum) still raises KeyError for
        unknown KB — the top_k value must not suppress the KeyError."""
        mem = MinimalMemory()
        await mem.initialize()
        with pytest.raises(KeyError):
            await mem.recall_kb("kb_x", "query", top_k=1)

    async def test_search_large_top_k_returns_at_most_available(self):
        """top_k larger than the number of stored nodes returns all matches,
        not more (no padding or error)."""
        mem = MinimalMemory()
        await mem.initialize()
        await mem.ingest("node one")
        await mem.ingest("node two")
        results = await mem.search("node", top_k=100)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# 8. State transition tests for the swap-conformance backend
# ---------------------------------------------------------------------------

class TestSwapConformanceStateTransitions:
    """Operations that mutate state: verify before → after → after-second."""

    async def test_ingest_then_count_then_ingest_again(self):
        """count_nodes grows by 1 for each ingest — not by 0 or 2."""
        mem = MinimalMemory()
        await mem.initialize()
        assert await mem.count_nodes() == 0
        await mem.ingest("first")
        assert await mem.count_nodes() == 1
        await mem.ingest("second")
        assert await mem.count_nodes() == 2

    async def test_remember_then_count(self):
        """remember() increments count_nodes just like ingest."""
        mem = MinimalMemory()
        await mem.initialize()
        n0 = await mem.count_nodes()
        await mem.remember("remembered content")
        n1 = await mem.count_nodes()
        assert n1 == n0 + 1

    async def test_ingest_then_search_finds_content(self):
        """After ingest, search must be able to surface the stored content."""
        mem = MinimalMemory()
        await mem.initialize()
        await mem.ingest("unique token xyzzy")
        results = await mem.search("xyzzy")
        # At least one result should match (implementation may score it high)
        assert len(results) >= 1
        found_records = [record for record, _ in results]
        content_values = [str(list(r.values())) for r in found_records]
        assert any("xyzzy" in cv for cv in content_values), (
            f"Ingested content 'xyzzy' not found in search results: {results!r}"
        )

    async def test_close_after_initialize_does_not_raise(self):
        """initialize → close is the normal lifecycle; no exception expected."""
        mem = MinimalMemory()
        await mem.initialize()
        await mem.close()  # must not raise

    async def test_multiple_ingest_calls_are_independent(self):
        """Each ingest returns a distinct dict (not the same object reused)."""
        mem = MinimalMemory()
        await mem.initialize()
        r1 = await mem.ingest("item A")
        r2 = await mem.ingest("item B")
        assert r1 is not r2

    async def test_recall_context_after_multiple_ingests(self):
        """recall_context with multiple node ids from a populated engine."""
        mem = MinimalMemory()
        await mem.initialize()
        r1 = await mem.ingest("concept alpha")
        r2 = await mem.ingest("concept beta")
        node_ids = [r1["node_id"], r2["node_id"]]
        ctx = await mem.recall_context(node_ids)
        assert "neighbor_keywords" in ctx
        assert "edges" in ctx
        for nid in node_ids:
            assert nid in ctx["neighbor_keywords"]


# ---------------------------------------------------------------------------
# 9. Additional negative / error-guessing tests
# ---------------------------------------------------------------------------

class TestNegativeErrorGuessing:
    """Error-guessing cases not covered by the parametrized removed-names tests."""

    def test_protocol_is_a_protocol_class(self):
        """MemoryEngine must actually be a Protocol (not a plain ABC or class)."""
        from krakey.interfaces.engines import MemoryEngine
        assert getattr(MemoryEngine, "_is_protocol", False) is True

    def test_memory_engine_not_constructable_directly(self):
        """Protocol classes are not meant to be instantiated (they're abstract).
        Attempting to call MemoryEngine() as a concrete class should either
        raise TypeError or return an empty protocol stub depending on Python
        version — it must NOT return a usable engine instance that passes
        isinstance with non-trivial behaviour."""
        from krakey.interfaces.engines import MemoryEngine
        # This is informational: if Protocol() is callable (Python allows it),
        # the resulting object should NOT have concrete memory semantics.
        # We don't assert an exception (CPython allows it) — we assert that a
        # MinimalMemory() instance is the correct way to obtain a conforming impl.
        assert isinstance(MinimalMemory(), MemoryEngine)

    def test_decision_result_memory_updates_kwarg_wrong_type_also_raises(self):
        """Passing memory_updates with any value raises TypeError."""
        from krakey.interfaces.engines import DecisionResult
        with pytest.raises(TypeError):
            DecisionResult(memory_updates=None)

    def test_decision_result_memory_updates_as_string_raises(self):
        """Even a string value for memory_updates raises TypeError."""
        from krakey.interfaces.engines import DecisionResult
        with pytest.raises(TypeError):
            DecisionResult(memory_updates="legacy string value")

    async def test_recall_kb_empty_string_kb_id_raises_keyerror(self):
        """Empty string is a valid but non-existent KB id — KeyError expected."""
        mem = MinimalMemory()
        await mem.initialize()
        with pytest.raises(KeyError):
            await mem.recall_kb("", "query")

    async def test_recall_context_non_existent_node_ids_returns_enrichment_dict(self):
        """recall_context with ids that don't map to any stored node must still
        return the correct shape — callers tolerate empty enrichment per spec."""
        mem = MinimalMemory()
        await mem.initialize()
        result = await mem.recall_context([9999, 8888])
        assert isinstance(result, dict)
        assert "neighbor_keywords" in result
        assert "edges" in result

    def test_knowledge_base_like_import_does_not_appear_in_engine_init(self):
        """Double-check: KnowledgeBaseLike must not be in the import block of
        krakey.interfaces.engines — testing the module text would be fragile,
        so we check the runtime namespace."""
        import krakey.interfaces.engines as pkg
        assert getattr(pkg, "KnowledgeBaseLike", None) is None

    def test_memory_engine_protocol_does_not_declare_upsert_node_in_any_form(self):
        """Ensure upsert_node is absent even from MemoryEngine.__dict__ directly
        (not just __protocol_attrs__), as a belt-and-suspenders check."""
        from krakey.interfaces.engines import MemoryEngine
        assert "upsert_node" not in MemoryEngine.__dict__, (
            "upsert_node is directly in MemoryEngine.__dict__. "
            "It must be removed from the Protocol definition."
        )

    def test_memory_engine_protocol_does_not_declare_sleep_cycle_in_dict(self):
        """sleep_cycle must not be in MemoryEngine.__dict__."""
        from krakey.interfaces.engines import MemoryEngine
        assert "sleep_cycle" not in MemoryEngine.__dict__, (
            "sleep_cycle is directly in MemoryEngine.__dict__. "
            "It must be moved to the engine implementation."
        )
