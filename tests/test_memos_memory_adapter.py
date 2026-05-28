"""Edge tests for MemOSMemoryEngine -- the optional MemOS adapter.

These tests validate the contract mapping between MemOSMemoryEngine and the
MemoryEngine + KnowledgeBaseLike Protocols. They run WITHOUT MemOS installed
by monkeypatching the lazy-import seam (_new_mos).

Test organisation:
  - FakeMOS / helpers at top
  - Tests grouped by protocol method / concern
  - Each group uses Positive / BVA / State-Transition / Negative sections

All async tests use bare `async def` -- pytest-asyncio in auto mode (see
pytest.ini: asyncio_mode = auto).
"""
from __future__ import annotations

import sys
import pytest

import krakey.engines.memory.memos as memos_mod
from krakey.interfaces.engines import KnowledgeBaseLike, MemoryEngine


# ---------------------------------------------------------------------------
# FakeMOS -- synchronous MOS stand-in (adapter wraps calls in asyncio.to_thread)
# ---------------------------------------------------------------------------

class FakeMOS:
    """Synchronous fake matching the MOS surface the adapter uses.

    Constructor accepts canned_items which populate the store used by
    search() / get_all() so tests can assert mapping without pre-calling add().
    """

    def __init__(self, canned_items=None):
        self._canned = canned_items or []
        self.create_user_calls = []
        self.register_mem_cube_calls = []
        self.add_calls = []
        self.search_calls = []
        self.get_all_calls = []
        self.update_calls = []
        self.delete_calls = []
        self.delete_all_calls = []

    def create_user(self, user_id=None, role=None, user_name=None, **kw):
        self.create_user_calls.append(
            {"user_id": user_id, "role": role, "user_name": user_name, **kw}
        )
        return user_id

    def register_mem_cube(self, name_or_path=None, mem_cube_id=None, user_id=None, **kw):
        self.register_mem_cube_calls.append(
            {"name_or_path": name_or_path, "mem_cube_id": mem_cube_id, "user_id": user_id, **kw}
        )

    def add(self, messages=None, memory_content=None, doc_path=None,
            mem_cube_id=None, user_id=None, session_id=None, **kw):
        self.add_calls.append({
            "messages": messages, "memory_content": memory_content,
            "doc_path": doc_path, "mem_cube_id": mem_cube_id,
            "user_id": user_id, "session_id": session_id, **kw,
        })

    def search(self, query, user_id=None, install_cube_ids=None,
               top_k=None, mode="fast", **kw):
        self.search_calls.append({
            "query": query, "user_id": user_id,
            "install_cube_ids": install_cube_ids, "top_k": top_k,
        })
        items = self._canned
        if install_cube_ids:
            items = [i for i in items if i.get("cube_id") in install_cube_ids]
        if top_k is not None:
            items = items[:top_k]
        return _make_search_result(items)

    def get_all(self, mem_cube_id=None, user_id=None, **kw):
        self.get_all_calls.append({"mem_cube_id": mem_cube_id, "user_id": user_id})
        items = self._canned
        if mem_cube_id is not None:
            items = [i for i in items if i.get("cube_id") == mem_cube_id]
        return _make_search_result(items)

    def update(self, mem_cube_id, memory_id, text_memory_item, user_id=None, **kw):
        self.update_calls.append({
            "mem_cube_id": mem_cube_id, "memory_id": memory_id,
            "text_memory_item": text_memory_item, "user_id": user_id,
        })

    def delete(self, mem_cube_id, memory_id, user_id=None, **kw):
        self.delete_calls.append({
            "mem_cube_id": mem_cube_id, "memory_id": memory_id, "user_id": user_id,
        })

    def delete_all(self, mem_cube_id=None, user_id=None, **kw):
        self.delete_all_calls.append({"mem_cube_id": mem_cube_id, "user_id": user_id})


def _make_item(cube_id, item_id, memory, source=None):
    """Build a canned item in MOSSearchResult item shape."""
    return {"cube_id": cube_id, "id": item_id, "memory": memory,
            "metadata": {"source": source}}


def _make_search_result(items):
    """Build the MOSSearchResult envelope, grouping by cube_id."""
    by_cube = {}
    for item in items:
        cid = item.get("cube_id", "default")
        by_cube.setdefault(cid, []).append(item)
    text_mem = [{"cube_id": cid, "memories": mems} for cid, mems in by_cube.items()]
    return {"text_mem": text_mem, "act_mem": [], "para_mem": [], "pref_mem": []}


def _make_engine(monkeypatch, *, fake=None, user_id="krakey",
                 mem_cube_id="krakey_main", mem_cube_path="",
                 mos_config_path="fake_config.yaml"):
    """Construct + patch a MemOSMemoryEngine ready for initialize()."""
    if fake is None:
        fake = FakeMOS()
    monkeypatch.setattr(memos_mod, "_new_mos", lambda *, mos_config_path: fake)
    engine = memos_mod.MemOSMemoryEngine(config={
        "mos_config_path": mos_config_path,
        "user_id": user_id,
        "mem_cube_id": mem_cube_id,
        "mem_cube_path": mem_cube_path,
    })
    return engine, fake


# ===========================================================================
# SECTION 1 - Protocol isinstance checks
# ===========================================================================

class TestProtocolConformance:
    """Positive: verify runtime_checkable isinstance() works at construction
    and after initialize().
    """

    def test_engine_is_instance_of_memory_engine_before_initialize(self, monkeypatch):
        """isinstance() check must pass immediately after construction --
        runtime_checkable Protocols are structural so no initialize() needed.
        """
        engine, _ = _make_engine(monkeypatch)
        assert isinstance(engine, MemoryEngine)

    async def test_engine_is_instance_of_memory_engine_after_initialize(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        assert isinstance(engine, MemoryEngine)

    async def test_create_kb_returns_knowledge_base_like(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        kb = await engine.create_kb("proto_kb", name="Proto")
        assert isinstance(kb, KnowledgeBaseLike)


# ===========================================================================
# SECTION 2 - initialize() wiring
# ===========================================================================

class TestInitialize:

    # ---- positive --------------------------------------------------------

    async def test_initialize_calls_new_mos_exactly_once(self, monkeypatch):
        call_count = []
        fake = FakeMOS()

        def capturing_new_mos(*, mos_config_path):
            call_count.append(mos_config_path)
            return fake

        monkeypatch.setattr(memos_mod, "_new_mos", capturing_new_mos)
        engine = memos_mod.MemOSMemoryEngine(config={"mos_config_path": "cfg.yaml"})
        await engine.initialize()

        assert len(call_count) == 1, "Expected _new_mos called exactly once"

    async def test_initialize_passes_mos_config_path(self, monkeypatch):
        received = {}
        fake = FakeMOS()

        def capturing_new_mos(*, mos_config_path):
            received["path"] = mos_config_path
            return fake

        monkeypatch.setattr(memos_mod, "_new_mos", capturing_new_mos)
        engine = memos_mod.MemOSMemoryEngine(config={"mos_config_path": "/my/mos.yaml"})
        await engine.initialize()

        assert received["path"] == "/my/mos.yaml"

    async def test_initialize_calls_create_user_with_configured_user_id(self, monkeypatch):
        engine, fake = _make_engine(monkeypatch, user_id="testuser")
        await engine.initialize()

        assert len(fake.create_user_calls) >= 1
        user_ids = [c["user_id"] for c in fake.create_user_calls]
        assert "testuser" in user_ids

    async def test_initialize_calls_register_mem_cube_with_primary_cube(self, monkeypatch):
        engine, fake = _make_engine(monkeypatch, mem_cube_id="my_cube")
        await engine.initialize()

        assert len(fake.register_mem_cube_calls) >= 1
        cube_ids = [c["mem_cube_id"] for c in fake.register_mem_cube_calls]
        assert "my_cube" in cube_ids

    async def test_initialize_default_user_id_is_krakey(self, monkeypatch):
        fake = FakeMOS()
        monkeypatch.setattr(memos_mod, "_new_mos", lambda *, mos_config_path: fake)
        engine = memos_mod.MemOSMemoryEngine(config={"mos_config_path": "x.yaml"})
        await engine.initialize()

        user_ids = [c["user_id"] for c in fake.create_user_calls]
        assert "krakey" in user_ids

    async def test_initialize_default_mem_cube_id_is_krakey_main(self, monkeypatch):
        fake = FakeMOS()
        monkeypatch.setattr(memos_mod, "_new_mos", lambda *, mos_config_path: fake)
        engine = memos_mod.MemOSMemoryEngine(config={"mos_config_path": "x.yaml"})
        await engine.initialize()

        cube_ids = [c["mem_cube_id"] for c in fake.register_mem_cube_calls]
        assert "krakey_main" in cube_ids

    async def test_close_does_not_raise_after_initialize(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        await engine.close()

    async def test_close_does_not_raise_before_initialize(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.close()

    # ---- negative --------------------------------------------------------

    async def test_missing_mos_config_path_raises_value_error(self, monkeypatch):
        """Missing mos_config_path -> ValueError.
        Dev may raise at construction OR initialize(); test accepts either.
        """
        fake = FakeMOS()
        monkeypatch.setattr(memos_mod, "_new_mos", lambda *, mos_config_path: fake)

        with pytest.raises(ValueError):
            engine = memos_mod.MemOSMemoryEngine(config={})
            await engine.initialize()

    async def test_empty_string_mos_config_path_raises_value_error(self, monkeypatch):
        """Empty string for mos_config_path is treated as missing."""
        fake = FakeMOS()
        monkeypatch.setattr(memos_mod, "_new_mos", lambda *, mos_config_path: fake)

        with pytest.raises(ValueError):
            engine = memos_mod.MemOSMemoryEngine(config={"mos_config_path": ""})
            await engine.initialize()

    async def test_none_mos_config_path_raises_value_error(self, monkeypatch):
        """None value for mos_config_path is treated as missing."""
        fake = FakeMOS()
        monkeypatch.setattr(memos_mod, "_new_mos", lambda *, mos_config_path: fake)

        with pytest.raises(ValueError):
            engine = memos_mod.MemOSMemoryEngine(config={"mos_config_path": None})
            await engine.initialize()

    async def test_none_config_raises_value_error(self, monkeypatch):
        """None config entirely also lacks mos_config_path."""
        fake = FakeMOS()
        monkeypatch.setattr(memos_mod, "_new_mos", lambda *, mos_config_path: fake)

        with pytest.raises(ValueError):
            engine = memos_mod.MemOSMemoryEngine(config=None)
            await engine.initialize()

    async def test_memos_not_installed_raises_import_error(self):
        """When MemOS is not installed, initialize() should raise ImportError.
        Tolerant: if MemOS IS installed, skip; otherwise verify the engine
        propagates ImportError when _new_mos is NOT monkeypatched.
        """
        try:
            import memos  # type: ignore  # noqa: F401
            pytest.skip("MemOS is installed; cannot test ImportError path")
        except ImportError:
            pass

        engine = memos_mod.MemOSMemoryEngine(config={"mos_config_path": "x.yaml"})
        with pytest.raises((ImportError, Exception)):
            await engine.initialize()


# ===========================================================================
# SECTION 3 - auto_ingest
# ===========================================================================

class TestAutoIngest:

    # ---- positive --------------------------------------------------------

    async def test_auto_ingest_calls_fake_add_with_content(self, monkeypatch):
        engine, fake = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.auto_ingest("Remember this fact")
        assert len(fake.add_calls) >= 1
        contents = [c["memory_content"] for c in fake.add_calls]
        assert "Remember this fact" in contents

    async def test_auto_ingest_returns_dict(self, monkeypatch):
        engine, fake = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.auto_ingest("Some content")
        assert isinstance(result, dict)

    async def test_auto_ingest_routes_to_primary_cube(self, monkeypatch):
        engine, fake = _make_engine(monkeypatch, mem_cube_id="primary_cube")
        await engine.initialize()
        await engine.auto_ingest("Fact about primary cube")
        primary_calls = [c for c in fake.add_calls if c.get("mem_cube_id") == "primary_cube"]
        assert len(primary_calls) >= 1

    async def test_auto_ingest_with_source_heartbeat(self, monkeypatch):
        engine, fake = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.auto_ingest("Heartbeat content", source_heartbeat=42)
        assert isinstance(result, dict)
        assert len(fake.add_calls) >= 1

    async def test_auto_ingest_without_source_heartbeat(self, monkeypatch):
        engine, fake = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.auto_ingest("No heartbeat")
        assert isinstance(result, dict)

    # ---- BVA -------------------------------------------------------------

    async def test_auto_ingest_empty_string_content(self, monkeypatch):
        """Empty string content should not crash."""
        engine, fake = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.auto_ingest("")
        assert isinstance(result, dict)

    async def test_auto_ingest_very_long_content(self, monkeypatch):
        """10k-char content must be accepted and forwarded without truncation."""
        engine, fake = _make_engine(monkeypatch)
        await engine.initialize()
        long_content = "x" * 10_000
        result = await engine.auto_ingest(long_content)
        assert isinstance(result, dict)
        contents = [c["memory_content"] for c in fake.add_calls]
        assert long_content in contents

    async def test_auto_ingest_unicode_content(self, monkeypatch):
        engine, fake = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.auto_ingest("日本語テスト: naive resume cafe")
        assert isinstance(result, dict)

    # ---- state transition ------------------------------------------------

    async def test_multiple_auto_ingest_calls_each_record_add(self, monkeypatch):
        """Each call to auto_ingest must produce a separate add record."""
        engine, fake = _make_engine(monkeypatch)
        await engine.initialize()
        await engine.auto_ingest("First fact")
        await engine.auto_ingest("Second fact")
        await engine.auto_ingest("Third fact")
        contents = [c["memory_content"] for c in fake.add_calls]
        assert "First fact" in contents
        assert "Second fact" in contents
        assert "Third fact" in contents
        assert len(fake.add_calls) >= 3


# ===========================================================================
# SECTION 4 - explicit_write
# ===========================================================================

class TestExplicitWrite:

    # ---- positive --------------------------------------------------------

    async def test_explicit_write_calls_add_with_content(self, monkeypatch):
        engine, fake = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.explicit_write("Explicit content")
        assert isinstance(result, dict)
        contents = [c["memory_content"] for c in fake.add_calls]
        assert "Explicit content" in contents

    async def test_explicit_write_returns_dict(self, monkeypatch):
        engine, fake = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.explicit_write("Something important")
        assert isinstance(result, dict)

    async def test_explicit_write_importance_does_not_crash(self, monkeypatch):
        engine, fake = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.explicit_write("High priority", importance="high")
        assert isinstance(result, dict)

    async def test_explicit_write_recall_context_list_of_dicts(self, monkeypatch):
        """recall_context as list of dicts must not crash."""
        engine, fake = _make_engine(monkeypatch)
        await engine.initialize()
        context = [{"node_id": 1, "text": "prior fact"}, {"node_id": 2, "text": "another"}]
        result = await engine.explicit_write("With context", recall_context=context)
        assert isinstance(result, dict)

    async def test_explicit_write_recall_context_none(self, monkeypatch):
        engine, fake = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.explicit_write("No context", recall_context=None)
        assert isinstance(result, dict)

    async def test_explicit_write_recall_context_empty_list(self, monkeypatch):
        engine, fake = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.explicit_write("Empty context", recall_context=[])
        assert isinstance(result, dict)

    # ---- BVA -------------------------------------------------------------

    async def test_explicit_write_empty_content(self, monkeypatch):
        engine, fake = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.explicit_write("")
        assert isinstance(result, dict)

    async def test_explicit_write_all_importance_levels(self, monkeypatch):
        engine, fake = _make_engine(monkeypatch)
        await engine.initialize()
        for importance in ("normal", "high", "low", "critical"):
            result = await engine.explicit_write(f"Content at {importance}", importance=importance)
            assert isinstance(result, dict)

    # ---- negative --------------------------------------------------------

    async def test_explicit_write_importance_not_forwarded_to_mos_add(self, monkeypatch):
        """The spec says importance/recall_context are DROPPED (not passed to add)."""
        engine, fake = _make_engine(monkeypatch)
        await engine.initialize()
        await engine.explicit_write("Dropped params", importance="critical",
                                    recall_context=[{"id": 1}])
        assert len(fake.add_calls) >= 1
        last_call = fake.add_calls[-1]
        assert "importance" not in last_call or last_call.get("importance") is None
        assert "recall_context" not in last_call or last_call.get("recall_context") is None


# ===========================================================================
# SECTION 5 - fts_search
# ===========================================================================

class TestFtsSearch:

    # ---- positive - 1 item -----------------------------------------------

    async def test_fts_search_single_item_has_required_keys(self, monkeypatch):
        item = _make_item("krakey_main", "id-1", "The capital of France is Paris", source="wiki")
        fake = FakeMOS(canned_items=[item])
        engine, _ = _make_engine(monkeypatch, fake=fake)
        await engine.initialize()
        results = await engine.fts_search("France")
        assert len(results) == 1
        node = results[0]
        for key in ("id", "name", "description", "category", "source_type", "importance", "metadata"):
            assert key in node, f"Missing key: {key}"

    async def test_fts_search_description_equals_item_memory_text(self, monkeypatch):
        item = _make_item("krakey_main", "id-1", "Specific memory text here", source=None)
        fake = FakeMOS(canned_items=[item])
        engine, _ = _make_engine(monkeypatch, fake=fake)
        await engine.initialize()
        results = await engine.fts_search("memory")
        assert results[0]["description"] == "Specific memory text here"

    async def test_fts_search_id_is_int(self, monkeypatch):
        item = _make_item("krakey_main", "abc-123", "Some fact")
        fake = FakeMOS(canned_items=[item])
        engine, _ = _make_engine(monkeypatch, fake=fake)
        await engine.initialize()
        results = await engine.fts_search("fact")
        assert isinstance(results[0]["id"], int)

    async def test_fts_search_category_is_fact(self, monkeypatch):
        item = _make_item("krakey_main", "id-1", "Any text", source="test")
        fake = FakeMOS(canned_items=[item])
        engine, _ = _make_engine(monkeypatch, fake=fake)
        await engine.initialize()
        results = await engine.fts_search("text")
        assert results[0]["category"] == "FACT"

    async def test_fts_search_source_type_from_metadata(self, monkeypatch):
        item = _make_item("krakey_main", "id-1", "Content", source="wikipedia")
        fake = FakeMOS(canned_items=[item])
        engine, _ = _make_engine(monkeypatch, fake=fake)
        await engine.initialize()
        results = await engine.fts_search("content")
        assert results[0]["source_type"] == "wikipedia"

    async def test_fts_search_importance_is_float(self, monkeypatch):
        item = _make_item("krakey_main", "id-1", "Some fact")
        fake = FakeMOS(canned_items=[item])
        engine, _ = _make_engine(monkeypatch, fake=fake)
        await engine.initialize()
        results = await engine.fts_search("fact")
        assert isinstance(results[0]["importance"], float)

    async def test_fts_search_metadata_is_dict(self, monkeypatch):
        item = _make_item("krakey_main", "id-1", "Content", source="src")
        fake = FakeMOS(canned_items=[item])
        engine, _ = _make_engine(monkeypatch, fake=fake)
        await engine.initialize()
        results = await engine.fts_search("content")
        assert isinstance(results[0]["metadata"], dict)

    # ---- positive - N items -----------------------------------------------

    async def test_fts_search_multiple_items_all_mapped(self, monkeypatch):
        items = [_make_item("krakey_main", f"id-{i}", f"Fact number {i}", source="src") for i in range(5)]
        fake = FakeMOS(canned_items=items)
        engine, _ = _make_engine(monkeypatch, fake=fake)
        await engine.initialize()
        results = await engine.fts_search("Fact", top_k=10)
        assert len(results) == 5

    async def test_fts_search_descriptions_match_items(self, monkeypatch):
        items = [_make_item("krakey_main", f"id-{i}", f"Memory text {i}") for i in range(3)]
        fake = FakeMOS(canned_items=items)
        engine, _ = _make_engine(monkeypatch, fake=fake)
        await engine.initialize()
        results = await engine.fts_search("Memory text", top_k=3)
        expected = {f"Memory text {i}" for i in range(3)}
        actual = {r["description"] for r in results}
        assert actual == expected

    # ---- BVA - 0 items ---------------------------------------------------

    async def test_fts_search_empty_canned_returns_empty_list(self, monkeypatch):
        fake = FakeMOS(canned_items=[])
        engine, _ = _make_engine(monkeypatch, fake=fake)
        await engine.initialize()
        results = await engine.fts_search("anything")
        assert results == []

    async def test_fts_search_empty_text_mem_envelope(self, monkeypatch):
        """MOSSearchResult with text_mem=[] -> empty results."""
        class EmptyResultFakeMOS(FakeMOS):
            def search(self, query, **kw):
                return {"text_mem": [], "act_mem": [], "para_mem": [], "pref_mem": []}

        fake = EmptyResultFakeMOS()
        engine, _ = _make_engine(monkeypatch, fake=fake)
        await engine.initialize()
        results = await engine.fts_search("query")
        assert results == []

    async def test_fts_search_text_mem_entry_with_empty_memories(self, monkeypatch):
        """text_mem entry present but memories list is empty."""
        class EmptyMemFakeMOS(FakeMOS):
            def search(self, query, **kw):
                return {"text_mem": [{"cube_id": "c", "memories": []}],
                        "act_mem": [], "para_mem": [], "pref_mem": []}

        fake = EmptyMemFakeMOS()
        engine, _ = _make_engine(monkeypatch, fake=fake)
        await engine.initialize()
        results = await engine.fts_search("query")
        assert results == []

    # ---- BVA - top_k passed through --------------------------------------

    async def test_fts_search_top_k_forwarded_to_mos(self, monkeypatch):
        fake = FakeMOS(canned_items=[_make_item("krakey_main", f"id-{i}", f"item {i}") for i in range(10)])
        engine, _ = _make_engine(monkeypatch, fake=fake)
        await engine.initialize()
        await engine.fts_search("item", top_k=3)
        assert fake.search_calls[-1]["top_k"] == 3

    async def test_fts_search_top_k_one_returns_at_most_one(self, monkeypatch):
        fake = FakeMOS(canned_items=[_make_item("krakey_main", f"id-{i}", f"item {i}") for i in range(5)])
        engine, _ = _make_engine(monkeypatch, fake=fake)
        await engine.initialize()
        results = await engine.fts_search("item", top_k=1)
        assert len(results) <= 1

    async def test_fts_search_default_top_k_is_five(self, monkeypatch):
        fake = FakeMOS(canned_items=[_make_item("krakey_main", f"id-{i}", f"item {i}") for i in range(10)])
        engine, _ = _make_engine(monkeypatch, fake=fake)
        await engine.initialize()
        await engine.fts_search("item")
        assert fake.search_calls[-1]["top_k"] == 5

    # ---- BVA - missing metadata ------------------------------------------

    async def test_fts_search_empty_metadata_source_gives_none_source_type(self, monkeypatch):
        item = {"cube_id": "krakey_main", "id": "no-src", "memory": "text no source", "metadata": {}}
        fake = FakeMOS(canned_items=[item])
        engine, _ = _make_engine(monkeypatch, fake=fake)
        await engine.initialize()
        results = await engine.fts_search("text")
        assert results[0]["source_type"] is None

    async def test_fts_search_absent_metadata_key_gives_none_source_type(self, monkeypatch):
        item = {"cube_id": "krakey_main", "id": "no-meta", "memory": "no metadata at all"}
        fake = FakeMOS(canned_items=[item])
        engine, _ = _make_engine(monkeypatch, fake=fake)
        await engine.initialize()
        results = await engine.fts_search("no metadata")
        assert results[0]["source_type"] is None

    async def test_fts_search_empty_memory_text_no_crash(self, monkeypatch):
        """Item with empty memory string -> description == '' (not crash)."""
        item = _make_item("krakey_main", "empty-id", "", source=None)
        fake = FakeMOS(canned_items=[item])
        engine, _ = _make_engine(monkeypatch, fake=fake)
        await engine.initialize()
        results = await engine.fts_search("")
        if results:
            assert results[0]["description"] == ""


# ===========================================================================
# SECTION 6 - vec_search always returns []
# ===========================================================================

class TestVecSearch:

    async def test_vec_search_returns_empty_list(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        assert await engine.vec_search([0.1, 0.2, 0.3]) == []

    async def test_vec_search_empty_vector(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        assert await engine.vec_search([]) == []

    async def test_vec_search_with_kwargs(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        assert await engine.vec_search([1.0] * 768, top_k=20, min_similarity=0.0) == []

    async def test_vec_search_does_not_raise_multiple_calls(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        await engine.vec_search([0.0])
        await engine.vec_search([0.5, 0.5], top_k=1)
        await engine.vec_search([0.1] * 100, min_similarity=0.99)


# ===========================================================================
# SECTION 7 - KB fleet: create_kb / open_kb / list_kbs / delete_kb
# ===========================================================================

class TestKBFleet:

    # ---- positive --------------------------------------------------------

    async def test_create_kb_calls_register_mem_cube_with_kb_id(self, monkeypatch):
        engine, fake = _make_engine(monkeypatch)
        await engine.initialize()
        await engine.create_kb("my_kb", name="My KB")
        cube_ids = [c["mem_cube_id"] for c in fake.register_mem_cube_calls]
        assert "my_kb" in cube_ids

    async def test_create_kb_returns_knowledge_base_like(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        kb = await engine.create_kb("kb1", name="KB One")
        assert isinstance(kb, KnowledgeBaseLike)

    async def test_open_kb_returns_same_instance_as_create_kb(self, monkeypatch):
        """open_kb on a previously created kb_id returns the SAME wrapper object."""
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        kb_created = await engine.create_kb("persistent_kb", name="Persistent")
        kb_opened = await engine.open_kb("persistent_kb")
        assert kb_created is kb_opened

    async def test_list_kbs_includes_created_kb_id(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        await engine.create_kb("listed_kb", name="Listed KB")
        kbs = await engine.list_kbs()
        kb_ids = [kb["kb_id"] for kb in kbs if "kb_id" in kb]
        assert "listed_kb" in kb_ids

    async def test_list_kbs_returns_list(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.list_kbs()
        assert isinstance(result, list)

    async def test_delete_kb_calls_delete_all_with_kb_id(self, monkeypatch):
        engine, fake = _make_engine(monkeypatch)
        await engine.initialize()
        await engine.create_kb("to_delete", name="Delete Me")
        await engine.delete_kb("to_delete")
        ids = [c["mem_cube_id"] for c in fake.delete_all_calls]
        assert "to_delete" in ids

    async def test_delete_kb_removes_from_list_kbs(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        await engine.create_kb("gone_kb", name="Gone")
        await engine.delete_kb("gone_kb")
        kbs = await engine.list_kbs()
        assert "gone_kb" not in [kb.get("kb_id") for kb in kbs]

    async def test_create_multiple_kbs_all_in_list_kbs(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        await engine.create_kb("kb_a", name="Alpha")
        await engine.create_kb("kb_b", name="Beta")
        await engine.create_kb("kb_c", name="Gamma")
        kbs = await engine.list_kbs()
        kb_ids = {kb.get("kb_id") for kb in kbs}
        assert {"kb_a", "kb_b", "kb_c"}.issubset(kb_ids)

    # ---- state transition ------------------------------------------------

    async def test_open_kb_after_delete_raises(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        await engine.create_kb("ephemeral", name="Ephemeral")
        await engine.delete_kb("ephemeral")
        with pytest.raises(Exception):
            await engine.open_kb("ephemeral")

    async def test_create_open_delete_list_round_trip(self, monkeypatch):
        """Full round-trip: create -> open (same instance) -> delete -> list (absent)."""
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        kb1 = await engine.create_kb("rt_kb", name="Round Trip")
        kb2 = await engine.open_kb("rt_kb")
        assert kb1 is kb2
        await engine.delete_kb("rt_kb")
        remaining = [kb.get("kb_id") for kb in await engine.list_kbs()]
        assert "rt_kb" not in remaining

    # ---- negative --------------------------------------------------------

    async def test_open_kb_unknown_id_raises(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        with pytest.raises(Exception):
            await engine.open_kb("definitely_does_not_exist")

    # ---- no-op methods do not raise -------------------------------------

    async def test_set_archived_does_not_raise(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        await engine.create_kb("arch_kb", name="Archive Test")
        await engine.set_archived("arch_kb", True)
        await engine.set_archived("arch_kb", False)

    async def test_set_archived_unknown_kb_does_not_raise(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        await engine.set_archived("ghost_kb", True)

    async def test_set_index_embedding_does_not_raise(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        await engine.create_kb("emb_kb", name="Embedding Test")
        await engine.set_index_embedding("emb_kb", [0.1, 0.2, 0.3])
        await engine.set_index_embedding("emb_kb", None)

    async def test_close_all_kbs_with_kbs_does_not_raise(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        await engine.create_kb("kb_c1", name="C1")
        await engine.create_kb("kb_c2", name="C2")
        await engine.close_all_kbs()

    async def test_close_all_kbs_empty_does_not_raise(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        await engine.close_all_kbs()


# ===========================================================================
# SECTION 8 - KB wrapper methods
# ===========================================================================

class TestKBWrapper:

    async def _get_kb(self, monkeypatch, kb_id="test_kb", canned_items=None):
        fake = FakeMOS(canned_items=canned_items or [])
        engine, _ = _make_engine(monkeypatch, fake=fake)
        await engine.initialize()
        kb = await engine.create_kb(kb_id, name="Test KB")
        return kb, fake

    # ---- write_entry -----------------------------------------------------

    async def test_write_entry_returns_int(self, monkeypatch):
        kb, _ = await self._get_kb(monkeypatch)
        result = await kb.write_entry("Entry content")
        assert isinstance(result, int)

    async def test_write_entry_calls_add_with_content(self, monkeypatch):
        kb, fake = await self._get_kb(monkeypatch)
        await kb.write_entry("My KB entry")
        contents = [c["memory_content"] for c in fake.add_calls]
        assert "My KB entry" in contents

    async def test_write_entry_targets_correct_cube_id(self, monkeypatch):
        kb, fake = await self._get_kb(monkeypatch, kb_id="specific_kb")
        await kb.write_entry("Targeted entry")
        calls_to_cube = [c for c in fake.add_calls if c.get("mem_cube_id") == "specific_kb"]
        assert len(calls_to_cube) >= 1

    async def test_two_write_entries_return_ints(self, monkeypatch):
        kb, _ = await self._get_kb(monkeypatch)
        id1 = await kb.write_entry("Entry one")
        id2 = await kb.write_entry("Entry two")
        assert isinstance(id1, int)
        assert isinstance(id2, int)

    # ---- vec_search ------------------------------------------------------

    async def test_kb_vec_search_returns_empty_list(self, monkeypatch):
        kb, _ = await self._get_kb(monkeypatch)
        assert await kb.vec_search([0.1, 0.2, 0.3]) == []

    async def test_kb_vec_search_does_not_raise(self, monkeypatch):
        kb, _ = await self._get_kb(monkeypatch)
        await kb.vec_search([])
        await kb.vec_search([1.0] * 512, top_k=10, min_similarity=0.0)

    # ---- write_edge ------------------------------------------------------

    async def test_kb_write_edge_returns_dict(self, monkeypatch):
        kb, _ = await self._get_kb(monkeypatch)
        result = await kb.write_edge(1, 2, "relates_to")
        assert isinstance(result, dict)

    async def test_kb_write_edge_does_not_raise(self, monkeypatch):
        kb, _ = await self._get_kb(monkeypatch)
        await kb.write_edge(1, 2, "rel")
        await kb.write_edge(100, 200, "another_pred")

    # ---- count_entries ---------------------------------------------------

    async def test_kb_count_entries_returns_nonneg_int(self, monkeypatch):
        canned = [_make_item("test_kb", f"id-{i}", f"item {i}") for i in range(3)]
        kb, _ = await self._get_kb(monkeypatch, canned_items=canned)
        result = await kb.count_entries()
        assert isinstance(result, int)
        assert result >= 0

    # ---- list_active_entries ---------------------------------------------

    async def test_kb_list_active_entries_returns_list(self, monkeypatch):
        canned = [_make_item("test_kb", f"id-{i}", f"item {i}") for i in range(3)]
        kb, _ = await self._get_kb(monkeypatch, canned_items=canned)
        result = await kb.list_active_entries(limit=10)
        assert isinstance(result, list)

    async def test_kb_list_active_entries_limit_one(self, monkeypatch):
        canned = [_make_item("test_kb", f"id-{i}", f"item {i}") for i in range(5)]
        kb, _ = await self._get_kb(monkeypatch, canned_items=canned)
        result = await kb.list_active_entries(limit=1)
        assert isinstance(result, list)
        assert len(result) <= 1

    # ---- search ----------------------------------------------------------

    async def test_kb_search_returns_list(self, monkeypatch):
        canned = [_make_item("test_kb", "id-1", "matching text", source="src")]
        kb, _ = await self._get_kb(monkeypatch, canned_items=canned)
        result = await kb.search("matching")
        assert isinstance(result, list)

    async def test_kb_search_items_have_id_and_content(self, monkeypatch):
        canned = [_make_item("test_kb", "id-1", "kb entry text", source="src")]
        kb, _ = await self._get_kb(monkeypatch, canned_items=canned)
        result = await kb.search("entry", top_k=5)
        if result:
            assert "id" in result[0]
            assert "content" in result[0]

    async def test_kb_search_passes_install_cube_ids(self, monkeypatch):
        """KB search must call fake.search with install_cube_ids=[kb_id]."""
        canned = [_make_item("scoped_kb", "id-1", "scoped text")]
        fake = FakeMOS(canned_items=canned)
        engine, _ = _make_engine(monkeypatch, fake=fake)
        await engine.initialize()
        kb = await engine.create_kb("scoped_kb", name="Scoped")
        await kb.search("scoped")
        search_call = fake.search_calls[-1]
        assert search_call["install_cube_ids"] == ["scoped_kb"]

    async def test_kb_search_empty_returns_list(self, monkeypatch):
        kb, _ = await self._get_kb(monkeypatch, canned_items=[])
        result = await kb.search("nothing matches")
        assert isinstance(result, list)

    # ---- initialize / close / merge_entry --------------------------------

    async def test_kb_initialize_does_not_raise(self, monkeypatch):
        kb, _ = await self._get_kb(monkeypatch)
        await kb.initialize()

    async def test_kb_close_does_not_raise(self, monkeypatch):
        kb, _ = await self._get_kb(monkeypatch)
        await kb.close()

    async def test_kb_merge_entry_does_not_raise(self, monkeypatch):
        kb, _ = await self._get_kb(monkeypatch)
        entry_id = await kb.write_entry("Original content")
        await kb.merge_entry(
            entry_id, new_content="Updated", new_embedding=[0.1, 0.2],
            new_importance=2.0, new_tags=["tag1"],
        )

    async def test_kb_merge_entry_none_embedding_does_not_raise(self, monkeypatch):
        kb, _ = await self._get_kb(monkeypatch)
        await kb.merge_entry(
            999, new_content="x", new_embedding=None,
            new_importance=1.0, new_tags=None,
        )


# ===========================================================================
# SECTION 9 - Stub methods returning empty structures
# ===========================================================================

class TestStubMethods:
    """All stubbed methods must RETURN EMPTY structures, NEVER raise."""

    async def test_find_by_name_returns_none(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        assert await engine.find_by_name("anything") is None

    async def test_update_node_category_returns_false(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        assert await engine.update_node_category("node_name", "NEW_CAT") is False

    async def test_count_by_category_returns_zero(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        assert await engine.count_by_category("FACT") == 0

    async def test_counts_by_category_returns_empty_dict(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        assert await engine.counts_by_category() == {}

    async def test_counts_by_source_returns_empty_dict(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        assert await engine.counts_by_source() == {}

    async def test_delete_by_category_returns_zero(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        assert await engine.delete_by_category("FACT") == 0

    async def test_list_edges_named_returns_empty_list(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.list_edges_named(limit=100)
        assert result == []

    async def test_list_edges_named_limit_one_returns_empty(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.list_edges_named(limit=1)
        assert isinstance(result, list)

    async def test_count_edges_returns_zero(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        assert await engine.count_edges() == 0

    async def test_get_neighbor_keywords_empty_input_returns_empty_dict(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        assert await engine.get_neighbor_keywords([]) == {}

    async def test_get_neighbor_keywords_three_ids(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        assert await engine.get_neighbor_keywords([1, 2, 3]) == {1: [], 2: [], 3: []}

    async def test_get_neighbor_keywords_single_id(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        assert await engine.get_neighbor_keywords([42]) == {42: []}

    async def test_get_edges_among_returns_empty_list(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        assert await engine.get_edges_among([1, 2, 3]) == []

    async def test_get_edges_among_empty_input(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        assert await engine.get_edges_among([]) == []

    async def test_insert_edge_with_cycle_check_returns_dict(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.insert_edge_with_cycle_check(1, 2, "predicate")
        assert isinstance(result, dict)

    async def test_insert_edge_with_cycle_check_self_loop_no_raise(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.insert_edge_with_cycle_check(1, 1, "self_loop")
        assert isinstance(result, dict)

    async def test_insert_edge_with_cycle_check_varied_args(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        await engine.insert_edge_with_cycle_check(100, 200, "long_predicate")

    async def test_classify_and_link_pending_classified_is_zero(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.classify_and_link_pending()
        assert isinstance(result, dict)
        assert result.get("classified") == 0

    async def test_classify_and_link_pending_does_not_raise(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        await engine.classify_and_link_pending()

    async def test_sleep_cycle_returns_empty_dict(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.sleep_cycle(channels=None, log_dir="logs", config={})
        assert result == {}

    async def test_sleep_cycle_does_not_raise_varied_args(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        await engine.sleep_cycle(channels=None, log_dir="x", config={})
        await engine.sleep_cycle(channels=None, log_dir="", config={"key": "val"})
        await engine.sleep_cycle(channels=object(), log_dir="logs", config={})


# ===========================================================================
# SECTION 10 - upsert_node
# ===========================================================================

class TestUpsertNode:

    async def test_upsert_node_returns_int(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.upsert_node({"name": "Node A", "category": "FACT"})
        assert isinstance(result, int)

    async def test_two_upserts_return_ints(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        r1 = await engine.upsert_node({"name": "Node X"})
        r2 = await engine.upsert_node({"name": "Node Y"})
        assert isinstance(r1, int)
        assert isinstance(r2, int)

    async def test_upsert_node_empty_dict_returns_int(self, monkeypatch):
        """Even an empty dict must return an int (fabricated id)."""
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.upsert_node({})
        assert isinstance(result, int)

    async def test_upsert_node_minimal_name_dict(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.upsert_node({"name": "N"})
        assert isinstance(result, int)


# ===========================================================================
# SECTION 11 - list_nodes and count_nodes
# ===========================================================================

class TestListAndCountNodes:

    async def test_count_nodes_returns_nonneg_int(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.count_nodes()
        assert isinstance(result, int)
        assert result >= 0

    async def test_list_nodes_returns_list(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        assert isinstance(await engine.list_nodes(), list)

    async def test_list_nodes_with_category_filter(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        assert isinstance(await engine.list_nodes(category="FACT"), list)

    async def test_list_nodes_with_limit_respects_bound(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.list_nodes(limit=5)
        assert isinstance(result, list)
        assert len(result) <= 5

    async def test_list_nodes_limit_zero_returns_empty(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        result = await engine.list_nodes(limit=0)
        assert isinstance(result, list)
        assert len(result) == 0

    async def test_count_nodes_after_auto_ingest_is_nonneg(self, monkeypatch):
        engine, _ = _make_engine(monkeypatch)
        await engine.initialize()
        await engine.auto_ingest("Some fact")
        result = await engine.count_nodes()
        assert isinstance(result, int)
        assert result >= 0


# ===========================================================================
# SECTION 12 - config defaults
# ===========================================================================

class TestConfigDefaults:

    async def test_user_id_defaults_to_krakey(self, monkeypatch):
        fake = FakeMOS()
        monkeypatch.setattr(memos_mod, "_new_mos", lambda *, mos_config_path: fake)
        engine = memos_mod.MemOSMemoryEngine(config={"mos_config_path": "cfg.yaml"})
        await engine.initialize()
        assert "krakey" in [c["user_id"] for c in fake.create_user_calls]

    async def test_mem_cube_id_defaults_to_krakey_main(self, monkeypatch):
        fake = FakeMOS()
        monkeypatch.setattr(memos_mod, "_new_mos", lambda *, mos_config_path: fake)
        engine = memos_mod.MemOSMemoryEngine(config={"mos_config_path": "cfg.yaml"})
        await engine.initialize()
        assert "krakey_main" in [c["mem_cube_id"] for c in fake.register_mem_cube_calls]

    async def test_custom_user_id_overrides_default(self, monkeypatch):
        engine, fake = _make_engine(monkeypatch, user_id="custom_user")
        await engine.initialize()
        assert "custom_user" in [c["user_id"] for c in fake.create_user_calls]

    async def test_custom_mem_cube_id_overrides_default(self, monkeypatch):
        engine, fake = _make_engine(monkeypatch, mem_cube_id="custom_cube")
        await engine.initialize()
        assert "custom_cube" in [c["mem_cube_id"] for c in fake.register_mem_cube_calls]

    async def test_mem_cube_path_default_no_crash(self, monkeypatch):
        """mem_cube_path defaults to '' -- ensure no crash with default."""
        fake = FakeMOS()
        monkeypatch.setattr(memos_mod, "_new_mos", lambda *, mos_config_path: fake)
        engine = memos_mod.MemOSMemoryEngine(config={"mos_config_path": "x.yaml"})
        await engine.initialize()

    async def test_explicit_mem_cube_path_no_crash(self, monkeypatch):
        engine, fake = _make_engine(monkeypatch, mem_cube_path="/data/cubes/main")
        await engine.initialize()
        assert len(fake.register_mem_cube_calls) >= 1
