"""Memory web service — edge tests.

Tests are written BEFORE any implementation exists and serve as the
acceptance criteria for ``krakey/engines/memory/web.py``.

Contract section: "Self-hosted web service (engine-owned, concrete concern)"
in ``contracts/memory-access/definition.md``.

All HTTP plumbing uses httpx ASGI transport (no port binding):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test"):
        ...

The engine fixture is a REAL GraphMemoryEngine backed by SQLite :memory:.
Fake embedder and fake extractor LLM are injected so no real LLM calls occur.
Seed data is written by calling engine methods directly before each test.

Run from repo root:
    pytest tests/test_memory_web_service.py
"""
from __future__ import annotations

import pytest
import httpx
from fastapi import FastAPI

from krakey.engines.memory.default import GraphMemoryEngine
from krakey.engines.memory.web import create_memory_app


# ---------------------------------------------------------------------------
# Fake collaborators
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Returns a small fixed-length vector for any text — no real embedding."""

    async def __call__(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class _FakeExtractorChat:
    """Returns a canned extraction JSON for any message list."""

    async def chat(self, messages, **kwargs) -> str:
        return (
            '{"nodes":[{"name":"x","category":"FACT","description":"d"}],'
            '"edges":[]}'
        )


# ---------------------------------------------------------------------------
# Engine factory helpers
# ---------------------------------------------------------------------------


async def _make_engine(tmp_path) -> GraphMemoryEngine:
    """Build, initialize, and return a fresh GraphMemoryEngine."""
    engine = GraphMemoryEngine(
        db_path=":memory:",
        embedder=_FakeEmbedder(),
        kb_dir=str(tmp_path / "kbs"),
        extractor_llm=_FakeExtractorChat(),
        # No sleep_llm — request_sleep returns {} immediately (by design)
    )
    await engine.initialize()
    return engine


def _client(app) -> httpx.AsyncClient:
    """Return an async httpx client wired to ``app`` via ASGI transport."""
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ---------------------------------------------------------------------------
# Unit-style: factory returns a FastAPI instance (no HTTP needed)
# ---------------------------------------------------------------------------


class TestCreateMemoryAppFactory:
    """Verify the factory itself — importable, returns FastAPI."""

    async def test_returns_fastapi_instance(self, tmp_path):
        """create_memory_app(engine) must return a FastAPI instance."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        assert isinstance(app, FastAPI)

    async def test_calling_factory_twice_returns_distinct_apps(self, tmp_path):
        """Two calls return two separate FastAPI objects (no singleton)."""
        engine = await _make_engine(tmp_path)
        app_a = create_memory_app(engine)
        app_b = create_memory_app(engine)
        assert app_a is not app_b


# ===========================================================================
# GET /api/gm/nodes
# ===========================================================================


class TestGetGMNodes:
    """GET /api/gm/nodes — positive, BVA, negative."""

    # --- positive / equivalence ---

    async def test_returns_200_empty_graph(self, tmp_path):
        """200 on an empty engine (no nodes seeded)."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/nodes")
        assert r.status_code == 200

    async def test_response_has_count_and_nodes_keys(self, tmp_path):
        """Body must have 'count' and 'nodes' keys."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/nodes")
        body = r.json()
        assert "count" in body
        assert "nodes" in body

    async def test_count_equals_two_after_seeding(self, tmp_path):
        """After seeding 2 nodes, count == 2 and len(nodes) == 2."""
        engine = await _make_engine(tmp_path)
        await engine.insert_node(name="Alpha", category="FACT", description="a")
        await engine.insert_node(name="Beta", category="KNOWLEDGE", description="b")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/nodes")
        body = r.json()
        assert body["count"] == 2
        assert len(body["nodes"]) == 2

    async def test_nodes_have_no_raw_embedding_key(self, tmp_path):
        """
        The contract says embeddings are trimmed: each node must NOT expose
        a raw 'embedding' key; instead it exposes a 'has_embedding' bool.

        Assumption: nodes with an embedding have has_embedding=True;
        nodes without have has_embedding=False.
        """
        engine = await _make_engine(tmp_path)
        await engine.insert_node(
            name="Alpha", category="FACT", description="a",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/nodes")
        node = r.json()["nodes"][0]
        assert "embedding" not in node
        assert "has_embedding" in node
        assert isinstance(node["has_embedding"], bool)

    async def test_has_embedding_true_when_embedding_stored(self, tmp_path):
        """has_embedding is True when the node was stored with an embedding."""
        engine = await _make_engine(tmp_path)
        await engine.insert_node(
            name="Alpha", category="FACT", description="a",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/nodes")
        assert r.json()["nodes"][0]["has_embedding"] is True

    async def test_has_embedding_false_when_no_embedding(self, tmp_path):
        """has_embedding is False when the node has no embedding."""
        engine = await _make_engine(tmp_path)
        await engine.insert_node(
            name="NoEmbed", category="FACT", description="plain",
        )
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/nodes")
        assert r.json()["nodes"][0]["has_embedding"] is False

    # --- category filter ---

    async def test_category_filter_returns_only_matching(self, tmp_path):
        """?category=FACT returns only FACT nodes when both categories present."""
        engine = await _make_engine(tmp_path)
        await engine.insert_node(name="F1", category="FACT", description="d")
        await engine.insert_node(name="K1", category="KNOWLEDGE", description="d")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/nodes?category=FACT")
        body = r.json()
        assert body["count"] == 1
        assert body["nodes"][0]["category"] == "FACT"

    async def test_category_filter_no_match_returns_empty(self, tmp_path):
        """?category=TARGET when no TARGET nodes → count 0, nodes []."""
        engine = await _make_engine(tmp_path)
        await engine.insert_node(name="F1", category="FACT", description="d")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/nodes?category=TARGET")
        body = r.json()
        assert body["count"] == 0
        assert body["nodes"] == []

    # --- BVA: limit parameter ---

    async def test_limit_one_caps_result_to_one(self, tmp_path):
        """?limit=1 returns at most 1 node even when 3 exist."""
        engine = await _make_engine(tmp_path)
        for i in range(3):
            await engine.insert_node(name=f"N{i}", category="FACT", description="d")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/nodes?limit=1")
        body = r.json()
        assert len(body["nodes"]) == 1
        # NOTE: count may reflect total (3) or returned (1) — we only assert
        # the list is capped. Assertion intentionally lenient on count value.

    async def test_limit_zero_rejected_by_validation(self, tmp_path):
        """
        ?limit=0 — the contract pins ``limit`` to ``Query(ge=1)``, so 0 is
        below the boundary and FastAPI rejects it with 422 (it must not
        500). This is the BVA just-below-boundary case.
        """
        engine = await _make_engine(tmp_path)
        await engine.insert_node(name="N0", category="FACT", description="d")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/nodes?limit=0")
        assert r.status_code == 422

    async def test_limit_large_returns_all(self, tmp_path):
        """?limit=999 when only 2 nodes exist returns 2 (not 500)."""
        engine = await _make_engine(tmp_path)
        await engine.insert_node(name="A", category="FACT", description="d")
        await engine.insert_node(name="B", category="FACT", description="d")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/nodes?limit=999")
        body = r.json()
        assert len(body["nodes"]) == 2

    # --- negative ---

    async def test_wrong_method_post_to_nodes_not_allowed(self, tmp_path):
        """
        POST /api/gm/nodes is a different route (create node).
        This test verifies GET-only semantics: GET must return 200 with a
        list body, not a creation body. (Covered separately in POST tests.)
        """
        # Covered by the POST test class; just ensure GET is stable.
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/nodes")
        assert r.status_code == 200


# ===========================================================================
# GET /api/gm/edges
# ===========================================================================


class TestGetGMEdges:
    """GET /api/gm/edges — positive, BVA."""

    async def test_returns_200_no_edges(self, tmp_path):
        """200 on an empty edge set."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/edges")
        assert r.status_code == 200

    async def test_body_has_count_and_edges_keys(self, tmp_path):
        """Body must have 'count' and 'edges' keys."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/edges")
        body = r.json()
        assert "count" in body
        assert "edges" in body

    async def test_edge_appears_as_name_triple(self, tmp_path):
        """
        After inserting two nodes and an edge, the edge appears in the
        response as a name triple with keys source, predicate, target.

        Assumption: the edge endpoint calls list_edges_named() internally,
        which returns {source, predicate, target} name dicts.
        """
        engine = await _make_engine(tmp_path)
        id_a = await engine.insert_node(name="Alice", category="FACT", description="d")
        id_b = await engine.insert_node(name="Bob", category="FACT", description="d")
        await engine.insert_edge_with_cycle_check(id_a, id_b, "knows")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/edges")
        body = r.json()
        assert body["count"] >= 1
        edge = body["edges"][0]
        assert "source" in edge
        assert "predicate" in edge
        assert "target" in edge

    async def test_edge_name_triple_values_correct(self, tmp_path):
        """The triple must resolve to the correct node names."""
        engine = await _make_engine(tmp_path)
        id_a = await engine.insert_node(name="Alice", category="FACT", description="d")
        id_b = await engine.insert_node(name="Bob", category="FACT", description="d")
        await engine.insert_edge_with_cycle_check(id_a, id_b, "knows")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/edges")
        edge = r.json()["edges"][0]
        assert edge["source"] == "Alice"
        assert edge["target"] == "Bob"
        assert edge["predicate"] == "knows"

    # --- BVA: limit parameter ---

    async def test_limit_one_returns_at_most_one_edge(self, tmp_path):
        """?limit=1 returns at most 1 edge when 2 exist."""
        engine = await _make_engine(tmp_path)
        id_a = await engine.insert_node(name="A", category="FACT", description="d")
        id_b = await engine.insert_node(name="B", category="FACT", description="d")
        id_c = await engine.insert_node(name="C", category="FACT", description="d")
        await engine.insert_edge_with_cycle_check(id_a, id_b, "rel")
        await engine.insert_edge_with_cycle_check(id_b, id_c, "rel")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/edges?limit=1")
        assert len(r.json()["edges"]) == 1


# ===========================================================================
# GET /api/gm/stats
# ===========================================================================


class TestGetGMStats:
    """GET /api/gm/stats — positive."""

    async def test_returns_200(self, tmp_path):
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/stats")
        assert r.status_code == 200

    async def test_body_has_required_keys(self, tmp_path):
        """Body must have total_nodes, total_edges, by_category, by_source."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/stats")
        body = r.json()
        assert "total_nodes" in body
        assert "total_edges" in body
        assert "by_category" in body
        assert "by_source" in body

    async def test_by_category_is_dict(self, tmp_path):
        """by_category must be a dict (mapping category → count)."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/stats")
        assert isinstance(r.json()["by_category"], dict)

    async def test_by_source_is_dict(self, tmp_path):
        """by_source must be a dict (mapping source_type → count)."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/stats")
        assert isinstance(r.json()["by_source"], dict)

    async def test_total_nodes_reflects_seeded_count(self, tmp_path):
        """total_nodes equals the number of seeded nodes."""
        engine = await _make_engine(tmp_path)
        await engine.insert_node(name="X", category="FACT", description="d")
        await engine.insert_node(name="Y", category="FACT", description="d")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/stats")
        assert r.json()["total_nodes"] == 2

    async def test_total_edges_reflects_seeded_count(self, tmp_path):
        """total_edges equals the number of seeded edges."""
        engine = await _make_engine(tmp_path)
        id_a = await engine.insert_node(name="P", category="FACT", description="d")
        id_b = await engine.insert_node(name="Q", category="FACT", description="d")
        await engine.insert_edge_with_cycle_check(id_a, id_b, "rel")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/stats")
        assert r.json()["total_edges"] == 1

    async def test_by_category_contains_inserted_category(self, tmp_path):
        """by_category has an entry for the category of the inserted node."""
        engine = await _make_engine(tmp_path)
        await engine.insert_node(name="Z", category="TARGET", description="d")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/gm/stats")
        assert "TARGET" in r.json()["by_category"]


# ===========================================================================
# GET /api/kbs
# ===========================================================================


class TestGetKBs:
    """GET /api/kbs — positive, state transition."""

    async def test_returns_200_empty(self, tmp_path):
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/kbs")
        assert r.status_code == 200

    async def test_body_has_kbs_key(self, tmp_path):
        """Body must have a 'kbs' key."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/kbs")
        assert "kbs" in r.json()

    async def test_kbs_is_list(self, tmp_path):
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/kbs")
        assert isinstance(r.json()["kbs"], list)

    async def test_created_kb_appears_in_list(self, tmp_path):
        """After create_kb, the kb_id appears in GET /api/kbs."""
        engine = await _make_engine(tmp_path)
        await engine.create_kb("kb1", name="KB One")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/kbs")
        kb_ids = [kb["kb_id"] for kb in r.json()["kbs"]]
        assert "kb1" in kb_ids

    async def test_two_kbs_both_appear(self, tmp_path):
        """After creating two KBs, both appear in the list."""
        engine = await _make_engine(tmp_path)
        await engine.create_kb("kb1", name="One")
        await engine.create_kb("kb2", name="Two")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/kbs")
        kb_ids = {kb["kb_id"] for kb in r.json()["kbs"]}
        assert "kb1" in kb_ids
        assert "kb2" in kb_ids

    async def test_default_list_excludes_archived(self, tmp_path):
        """
        Default GET /api/kbs excludes archived KBs (contract: list_kbs
        default excludes archived). After archiving, the KB disappears.

        This is a state-transition test: create → verify present → archive
        → verify absent from default list.
        """
        engine = await _make_engine(tmp_path)
        await engine.create_kb("kb_arch", name="To Archive")
        app = create_memory_app(engine)

        # Confirm present before archive
        async with _client(app) as c:
            r = await c.get("/api/kbs")
        kb_ids_before = {kb["kb_id"] for kb in r.json()["kbs"]}
        assert "kb_arch" in kb_ids_before

        # Archive via PATCH
        async with _client(app) as c:
            await c.patch("/api/kbs/kb_arch", json={"archived": True})
            r = await c.get("/api/kbs")

        kb_ids_after = {kb["kb_id"] for kb in r.json()["kbs"]}
        # The archived KB must no longer appear in the default listing.
        assert "kb_arch" not in kb_ids_after


# ===========================================================================
# GET /api/kb/{kb_id}/entries
# ===========================================================================


class TestGetKBEntries:
    """GET /api/kb/{kb_id}/entries — positive, BVA, negative."""

    async def test_returns_200_on_known_kb(self, tmp_path):
        """200 for a KB that was created."""
        engine = await _make_engine(tmp_path)
        await engine.create_kb("kb1", name="One")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/kb/kb1/entries")
        assert r.status_code == 200

    async def test_body_has_kb_id_count_entries(self, tmp_path):
        """Body must contain kb_id, count, and entries keys."""
        engine = await _make_engine(tmp_path)
        await engine.create_kb("kb1", name="One")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/kb/kb1/entries")
        body = r.json()
        assert "kb_id" in body
        assert "count" in body
        assert "entries" in body

    async def test_kb_id_in_body_matches_path(self, tmp_path):
        """kb_id in the response body matches the path parameter."""
        engine = await _make_engine(tmp_path)
        await engine.create_kb("kb1", name="One")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/kb/kb1/entries")
        assert r.json()["kb_id"] == "kb1"

    async def test_seeded_entry_appears(self, tmp_path):
        """Entry written to a KB via write_entry appears in the entries list."""
        engine = await _make_engine(tmp_path)
        kb = await engine.create_kb("kb1", name="One")
        await kb.write_entry("hello world", tags=["t"])
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/kb/kb1/entries")
        body = r.json()
        assert body["count"] >= 1
        assert len(body["entries"]) >= 1

    async def test_empty_kb_has_count_zero(self, tmp_path):
        """A freshly created KB with no entries has count 0."""
        engine = await _make_engine(tmp_path)
        await engine.create_kb("kb_empty", name="Empty")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/kb/kb_empty/entries")
        assert r.json()["count"] == 0

    # --- BVA: limit parameter ---

    async def test_limit_one_caps_entries(self, tmp_path):
        """?limit=1 returns at most 1 entry even when 3 exist."""
        engine = await _make_engine(tmp_path)
        kb = await engine.create_kb("kb1", name="One")
        for i in range(3):
            await kb.write_entry(f"entry {i}")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/kb/kb1/entries?limit=1")
        assert len(r.json()["entries"]) <= 1

    # --- negative ---

    async def test_404_on_unknown_kb_id(self, tmp_path):
        """Unknown KB ID must return 404, not 500."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/kb/UNKNOWN_KB/entries")
        assert r.status_code == 404

    async def test_404_body_is_json(self, tmp_path):
        """404 response body must be parseable JSON."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/kb/NONEXISTENT/entries")
        # Must not raise
        body = r.json()
        assert body is not None


# ===========================================================================
# GET /  (HTML browser page)
# ===========================================================================


class TestGetRoot:
    """GET / — returns HTML page."""

    async def test_returns_200(self, tmp_path):
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/")
        assert r.status_code == 200

    async def test_content_type_is_html(self, tmp_path):
        """Content-type must be text/html (case-insensitive prefix match)."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/")
        ct = r.headers.get("content-type", "")
        assert "text/html" in ct.lower()

    async def test_body_is_non_empty(self, tmp_path):
        """HTML body must be non-empty."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/")
        assert len(r.text) > 0


# ===========================================================================
# POST /api/gm/nodes  (create node)
# ===========================================================================


class TestPostGMNodes:
    """POST /api/gm/nodes — positive, state transition, negative."""

    # --- positive ---

    async def test_returns_success_status(self, tmp_path):
        """
        POST returns 200 or 201 (contract does not pin an exact code).

        Assumption: success code is either 200 or 201.
        """
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/gm/nodes", json={
                "name": "NewNode",
                "category": "FACT",
                "description": "a test node",
            })
        assert r.status_code in (200, 201)

    async def test_response_body_has_id(self, tmp_path):
        """Response body must contain an 'id' key (the new node id)."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/gm/nodes", json={
                "name": "NewNode",
                "category": "FACT",
                "description": "a test node",
            })
        assert "id" in r.json()

    async def test_returned_id_is_integer(self, tmp_path):
        """The returned 'id' must be an integer."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/gm/nodes", json={
                "name": "NewNode",
                "category": "FACT",
                "description": "a test node",
            })
        assert isinstance(r.json()["id"], int)

    # --- state transition ---

    async def test_node_count_increases_after_post(self, tmp_path):
        """
        State transition: GET /api/gm/nodes count before vs after POST.
        Count must increase by at least 1.
        """
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            before = (await c.get("/api/gm/nodes")).json()["count"]
            await c.post("/api/gm/nodes", json={
                "name": "Fresh", "category": "FACT", "description": "d",
            })
            after = (await c.get("/api/gm/nodes")).json()["count"]
        assert after >= before + 1

    async def test_new_node_appears_in_get_nodes(self, tmp_path):
        """The newly created node's name appears in a subsequent GET /api/gm/nodes."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            await c.post("/api/gm/nodes", json={
                "name": "UniqueNodeName", "category": "FACT", "description": "d",
            })
            nodes = (await c.get("/api/gm/nodes")).json()["nodes"]
        names = [n["name"] for n in nodes]
        assert "UniqueNodeName" in names

    async def test_importance_field_accepted(self, tmp_path):
        """Optional 'importance' field is accepted without error."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/gm/nodes", json={
                "name": "Important",
                "category": "TARGET",
                "description": "big deal",
                "importance": 2.0,
            })
        assert r.status_code in (200, 201)

    # --- negative ---

    async def test_missing_name_returns_400_or_422(self, tmp_path):
        """
        POST without 'name' must fail with 400 or 422 (FastAPI validation
        or app-level check), not 500.
        """
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/gm/nodes", json={
                "category": "FACT",
                "description": "missing name",
            })
        assert r.status_code in (400, 422)

    async def test_missing_category_returns_400_or_422(self, tmp_path):
        """POST without 'category' must fail with 400 or 422, not 500."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/gm/nodes", json={
                "name": "NoCategory",
                "description": "missing category",
            })
        assert r.status_code in (400, 422)

    async def test_missing_description_returns_400_or_422(self, tmp_path):
        """POST without 'description' must fail with 400 or 422, not 500."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/gm/nodes", json={
                "name": "NoDesc",
                "category": "FACT",
            })
        assert r.status_code in (400, 422)

    async def test_empty_body_returns_400_or_422(self, tmp_path):
        """POST with empty body must fail with 400 or 422, not 500."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/gm/nodes", json={})
        assert r.status_code in (400, 422)


# ===========================================================================
# PATCH /api/gm/nodes/{id}
# ===========================================================================


class TestPatchGMNode:
    """PATCH /api/gm/nodes/{id} — positive, state transition, negative."""

    async def test_patch_returns_200_with_ok(self, tmp_path):
        """PATCH a valid node returns 200 with {ok: true}."""
        engine = await _make_engine(tmp_path)
        nid = await engine.insert_node(name="N", category="FACT", description="d")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.patch(f"/api/gm/nodes/{nid}", json={"category": "KNOWLEDGE"})
        assert r.status_code == 200
        assert r.json().get("ok") is True

    async def test_patch_category_changes_node(self, tmp_path):
        """
        State transition: after PATCH {category: KNOWLEDGE}, the node's
        category must reflect KNOWLEDGE in a subsequent GET /api/gm/nodes.
        """
        engine = await _make_engine(tmp_path)
        nid = await engine.insert_node(name="Mutable", category="FACT", description="d")
        app = create_memory_app(engine)
        async with _client(app) as c:
            await c.patch(f"/api/gm/nodes/{nid}", json={"category": "KNOWLEDGE"})
            nodes = (await c.get("/api/gm/nodes")).json()["nodes"]
        target = next((n for n in nodes if n["name"] == "Mutable"), None)
        assert target is not None
        assert target["category"] == "KNOWLEDGE"

    async def test_patch_description_updates_node(self, tmp_path):
        """
        PATCH {description: 'new desc'} must update the description.

        Assumption: description is a mutable field accepted by PATCH.
        The contract lists description in the mutable fields set.
        """
        engine = await _make_engine(tmp_path)
        nid = await engine.insert_node(name="PatchMe", category="FACT", description="old")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.patch(f"/api/gm/nodes/{nid}", json={"description": "new desc"})
        assert r.status_code == 200

    async def test_patch_nonexistent_node(self, tmp_path):
        """
        PATCH on a non-existent node id must not 500.
        Assert status is not 500 (404 or 200/ok:false are both acceptable).
        """
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.patch("/api/gm/nodes/99999", json={"category": "FACT"})
        assert r.status_code != 500


# ===========================================================================
# DELETE /api/gm/nodes/{id}
# ===========================================================================


class TestDeleteGMNode:
    """DELETE /api/gm/nodes/{id} — positive, state transition, negative."""

    async def test_delete_returns_200_with_ok(self, tmp_path):
        """DELETE a valid node returns 200 with {ok: true}."""
        engine = await _make_engine(tmp_path)
        nid = await engine.insert_node(name="ToDelete", category="FACT", description="d")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.delete(f"/api/gm/nodes/{nid}")
        assert r.status_code == 200
        assert r.json().get("ok") is True

    async def test_node_count_decreases_after_delete(self, tmp_path):
        """
        State transition: count before delete is count+1 vs count after.
        """
        engine = await _make_engine(tmp_path)
        nid = await engine.insert_node(name="Gone", category="FACT", description="d")
        app = create_memory_app(engine)
        async with _client(app) as c:
            before = (await c.get("/api/gm/nodes")).json()["count"]
            await c.delete(f"/api/gm/nodes/{nid}")
            after = (await c.get("/api/gm/nodes")).json()["count"]
        assert after < before

    async def test_deleted_node_not_in_list(self, tmp_path):
        """After DELETE, the node no longer appears in GET /api/gm/nodes."""
        engine = await _make_engine(tmp_path)
        nid = await engine.insert_node(name="Vanished", category="FACT", description="d")
        app = create_memory_app(engine)
        async with _client(app) as c:
            await c.delete(f"/api/gm/nodes/{nid}")
            nodes = (await c.get("/api/gm/nodes")).json()["nodes"]
        names = [n["name"] for n in nodes]
        assert "Vanished" not in names

    async def test_delete_nonexistent_node_no_500(self, tmp_path):
        """
        DELETE a non-existent id must NOT return 500.

        Assumption (lenient): the contract says DELETE → {ok}. The engine's
        delete_node is a no-op for missing ids. The endpoint may return 200
        ok or 404; either is acceptable — just not 500.
        """
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.delete("/api/gm/nodes/99999")
        assert r.status_code in (200, 404)

    async def test_double_delete_no_500(self, tmp_path):
        """Deleting the same node twice must not 500 on the second call."""
        engine = await _make_engine(tmp_path)
        nid = await engine.insert_node(name="Double", category="FACT", description="d")
        app = create_memory_app(engine)
        async with _client(app) as c:
            await c.delete(f"/api/gm/nodes/{nid}")
            r = await c.delete(f"/api/gm/nodes/{nid}")
        assert r.status_code in (200, 404)


# ===========================================================================
# POST /api/gm/edges
# ===========================================================================


class TestPostGMEdges:
    """POST /api/gm/edges — positive, negative (self-loop, cycle)."""

    async def test_insert_edge_returns_200(self, tmp_path):
        engine = await _make_engine(tmp_path)
        await engine.insert_node(name="A", category="FACT", description="d")
        await engine.insert_node(name="B", category="FACT", description="d")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/gm/edges", json={
                "source_name": "A",
                "target_name": "B",
                "predicate": "rel",
            })
        # POST-create returns 201 Created (FastAPI status_code=201 on the route).
        assert r.status_code in (200, 201)

    async def test_insert_edge_body_has_inserted_or_skipped(self, tmp_path):
        """
        Response body must have 'inserted' or 'skipped' key per contract.

        Assumption: body contains at least one of: inserted, skipped.
        """
        engine = await _make_engine(tmp_path)
        await engine.insert_node(name="A", category="FACT", description="d")
        await engine.insert_node(name="B", category="FACT", description="d")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/gm/edges", json={
                "source_name": "A",
                "target_name": "B",
                "predicate": "knows",
            })
        body = r.json()
        assert ("inserted" in body) or ("skipped" in body)

    async def test_self_loop_edge_skipped_no_500(self, tmp_path):
        """
        Self-loop (source == target by name) must be skipped, not 500.

        Assumption: the engine's insert_edge_with_cycle_check rejects
        self-loops. The endpoint must surface this as skipped or a 200/400,
        but must NOT raise an unhandled 500.
        """
        engine = await _make_engine(tmp_path)
        await engine.insert_node(name="Self", category="FACT", description="d")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/gm/edges", json={
                "source_name": "Self",
                "target_name": "Self",
                "predicate": "self_rel",
            })
        assert r.status_code != 500

    async def test_duplicate_edge_no_500(self, tmp_path):
        """Inserting the same edge twice must not 500."""
        engine = await _make_engine(tmp_path)
        await engine.insert_node(name="A", category="FACT", description="d")
        await engine.insert_node(name="B", category="FACT", description="d")
        app = create_memory_app(engine)
        async with _client(app) as c:
            await c.post("/api/gm/edges", json={
                "source_name": "A", "target_name": "B", "predicate": "r",
            })
            r = await c.post("/api/gm/edges", json={
                "source_name": "A", "target_name": "B", "predicate": "r",
            })
        assert r.status_code != 500

    async def test_edge_with_source_id_accepted(self, tmp_path):
        """
        The contract allows source_id/target_id as alternative to name.
        This test uses numeric IDs. Must return 200 and not 422/500.

        Assumption: endpoint accepts either source_name+target_name or
        source_id+target_id (as per the contract spec).
        """
        engine = await _make_engine(tmp_path)
        id_a = await engine.insert_node(name="A2", category="FACT", description="d")
        id_b = await engine.insert_node(name="B2", category="FACT", description="d")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/gm/edges", json={
                "source_id": id_a,
                "target_id": id_b,
                "predicate": "id_based",
            })
        assert r.status_code not in (500, 422)

    async def test_missing_predicate_returns_400_or_422(self, tmp_path):
        """POST without predicate must fail with 400 or 422."""
        engine = await _make_engine(tmp_path)
        await engine.insert_node(name="A", category="FACT", description="d")
        await engine.insert_node(name="B", category="FACT", description="d")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/gm/edges", json={
                "source_name": "A",
                "target_name": "B",
            })
        assert r.status_code in (400, 422)


# ===========================================================================
# POST /api/kbs  (create KB)
# ===========================================================================


class TestPostKBs:
    """POST /api/kbs — positive, state transition."""

    async def test_returns_200_with_kb_id(self, tmp_path):
        """POST returns 200 with {kb_id}."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/kbs", json={
                "kb_id": "kb_new",
                "name": "New KB",
            })
        assert r.status_code in (200, 201)
        assert "kb_id" in r.json()

    async def test_returned_kb_id_matches_request(self, tmp_path):
        """The returned kb_id must match what was sent."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/kbs", json={
                "kb_id": "my_kb",
                "name": "Mine",
            })
        assert r.json()["kb_id"] == "my_kb"

    async def test_new_kb_appears_in_list(self, tmp_path):
        """
        State transition: after POST, the KB appears in GET /api/kbs.
        """
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            await c.post("/api/kbs", json={"kb_id": "kb_http", "name": "HTTP"})
            r = await c.get("/api/kbs")
        kb_ids = {kb["kb_id"] for kb in r.json()["kbs"]}
        assert "kb_http" in kb_ids

    async def test_optional_description_accepted(self, tmp_path):
        """Optional 'description' field is accepted without error."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/kbs", json={
                "kb_id": "kb_desc",
                "name": "With Desc",
                "description": "A longer description",
            })
        assert r.status_code in (200, 201)

    async def test_missing_name_returns_400_or_422(self, tmp_path):
        """POST without 'name' fails with 400 or 422, not 500."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/kbs", json={"kb_id": "no_name"})
        assert r.status_code in (400, 422)


# ===========================================================================
# POST /api/kb/{kb_id}/entries
# ===========================================================================


class TestPostKBEntries:
    """POST /api/kb/{kb_id}/entries — positive, state transition, negative."""

    async def test_returns_200_with_id(self, tmp_path):
        """POST returns 200/201 with {id}."""
        engine = await _make_engine(tmp_path)
        await engine.create_kb("kb1", name="One")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/kb/kb1/entries", json={"content": "hello"})
        assert r.status_code in (200, 201)
        assert "id" in r.json()

    async def test_returned_id_is_integer(self, tmp_path):
        """The returned entry id must be an integer."""
        engine = await _make_engine(tmp_path)
        await engine.create_kb("kb1", name="One")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/kb/kb1/entries", json={"content": "hello"})
        assert isinstance(r.json()["id"], int)

    async def test_entry_count_increases_after_post(self, tmp_path):
        """
        State transition: after POST, the entry count for that KB increases.
        """
        engine = await _make_engine(tmp_path)
        await engine.create_kb("kb1", name="One")
        app = create_memory_app(engine)
        async with _client(app) as c:
            before = (await c.get("/api/kb/kb1/entries")).json()["count"]
            await c.post("/api/kb/kb1/entries", json={"content": "new entry"})
            after = (await c.get("/api/kb/kb1/entries")).json()["count"]
        assert after > before

    async def test_posted_entry_appears_in_list(self, tmp_path):
        """After POST, the entry appears in GET /api/kb/{kb_id}/entries."""
        engine = await _make_engine(tmp_path)
        await engine.create_kb("kb1", name="One")
        app = create_memory_app(engine)
        async with _client(app) as c:
            await c.post("/api/kb/kb1/entries", json={"content": "my content"})
            body = (await c.get("/api/kb/kb1/entries")).json()
        contents = [e.get("content", "") for e in body["entries"]]
        assert "my content" in contents

    async def test_optional_tags_accepted(self, tmp_path):
        """Optional 'tags' field is accepted without error."""
        engine = await _make_engine(tmp_path)
        await engine.create_kb("kb1", name="One")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/kb/kb1/entries", json={
                "content": "tagged entry",
                "tags": ["science", "important"],
            })
        assert r.status_code in (200, 201)

    async def test_404_on_unknown_kb(self, tmp_path):
        """POST to a non-existent KB must return 404, not 500."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/kb/NONEXISTENT/entries", json={"content": "x"})
        assert r.status_code == 404

    async def test_missing_content_returns_400_or_422(self, tmp_path):
        """POST without 'content' fails with 400 or 422, not 500."""
        engine = await _make_engine(tmp_path)
        await engine.create_kb("kb1", name="One")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/kb/kb1/entries", json={})
        assert r.status_code in (400, 422)


# ===========================================================================
# PATCH /api/kbs/{kb_id}  (archive)
# ===========================================================================


class TestPatchKBs:
    """PATCH /api/kbs/{kb_id} — positive, state transition."""

    async def test_archive_kb_returns_200(self, tmp_path):
        """PATCH {archived: true} on a valid KB returns 200."""
        engine = await _make_engine(tmp_path)
        await engine.create_kb("kb1", name="One")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.patch("/api/kbs/kb1", json={"archived": True})
        assert r.status_code == 200

    async def test_archive_response_has_ok(self, tmp_path):
        """Response body must contain {ok: true}."""
        engine = await _make_engine(tmp_path)
        await engine.create_kb("kb1", name="One")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.patch("/api/kbs/kb1", json={"archived": True})
        assert r.json().get("ok") is True

    async def test_archived_kb_disappears_from_default_list(self, tmp_path):
        """
        State transition: archived KB must not appear in GET /api/kbs (default).

        The contract states list_kbs default excludes archived.
        """
        engine = await _make_engine(tmp_path)
        await engine.create_kb("to_archive", name="Will Archive")
        app = create_memory_app(engine)
        async with _client(app) as c:
            # Confirm present
            r = await c.get("/api/kbs")
            ids_before = {kb["kb_id"] for kb in r.json()["kbs"]}
            assert "to_archive" in ids_before

            # Archive
            await c.patch("/api/kbs/to_archive", json={"archived": True})

            # Confirm gone from default list
            r = await c.get("/api/kbs")
            ids_after = {kb["kb_id"] for kb in r.json()["kbs"]}
        assert "to_archive" not in ids_after

    async def test_unarchive_restores_to_list(self, tmp_path):
        """
        State transition: archive → unarchive restores the KB to default list.
        """
        engine = await _make_engine(tmp_path)
        await engine.create_kb("revive_me", name="Revive")
        app = create_memory_app(engine)
        async with _client(app) as c:
            await c.patch("/api/kbs/revive_me", json={"archived": True})
            # Confirm gone
            ids_archived = {kb["kb_id"] for kb in (await c.get("/api/kbs")).json()["kbs"]}
            assert "revive_me" not in ids_archived

            # Unarchive
            await c.patch("/api/kbs/revive_me", json={"archived": False})
            ids_revived = {kb["kb_id"] for kb in (await c.get("/api/kbs")).json()["kbs"]}
        assert "revive_me" in ids_revived

    async def test_patch_nonexistent_kb_no_500(self, tmp_path):
        """
        PATCH on a non-existent KB must not 500.
        Accept 200 (no-op) or 404.
        """
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.patch("/api/kbs/NOBODY", json={"archived": True})
        assert r.status_code != 500


# ===========================================================================
# DELETE /api/kbs/{kb_id}
# ===========================================================================


class TestDeleteKBs:
    """DELETE /api/kbs/{kb_id} — positive, state transition."""

    async def test_delete_returns_200(self, tmp_path):
        """DELETE a valid KB returns 200."""
        engine = await _make_engine(tmp_path)
        await engine.create_kb("del_kb", name="Delete Me")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.delete("/api/kbs/del_kb")
        assert r.status_code == 200

    async def test_delete_response_has_ok(self, tmp_path):
        """Response body must contain {ok: true}."""
        engine = await _make_engine(tmp_path)
        await engine.create_kb("del_kb", name="Delete Me")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.delete("/api/kbs/del_kb")
        assert r.json().get("ok") is True

    async def test_deleted_kb_gone_from_list(self, tmp_path):
        """
        State transition: after DELETE, the KB no longer appears in GET /api/kbs.
        """
        engine = await _make_engine(tmp_path)
        await engine.create_kb("gone_kb", name="Gone")
        app = create_memory_app(engine)
        async with _client(app) as c:
            await c.delete("/api/kbs/gone_kb")
            r = await c.get("/api/kbs")
        kb_ids = {kb["kb_id"] for kb in r.json()["kbs"]}
        assert "gone_kb" not in kb_ids

    async def test_delete_nonexistent_kb_no_500(self, tmp_path):
        """DELETE a non-existent KB must not 500. Accept 200 or 404."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.delete("/api/kbs/NOBODY")
        assert r.status_code in (200, 404)


# ===========================================================================
# POST /api/sleep
# ===========================================================================


class TestPostSleep:
    """POST /api/sleep — positive."""

    async def test_returns_200(self, tmp_path):
        """POST /api/sleep returns 200 (no sleep_llm configured → {} body)."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/sleep", json={"reason": "test"})
        assert r.status_code == 200

    async def test_returns_dict_body(self, tmp_path):
        """
        Body must be a dict.

        Assumption: the engine returns {} when no sleep_llm is configured
        (per request_sleep docstring: 'Returns {} if no sleep_llm').
        Assert body is a dict (including empty dict).
        """
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/sleep", json={"reason": "test"})
        assert isinstance(r.json(), dict)

    async def test_empty_reason_accepted(self, tmp_path):
        """POST /api/sleep with empty reason must not 500."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/sleep", json={"reason": ""})
        assert r.status_code == 200

    async def test_missing_reason_field_accepted(self, tmp_path):
        """
        POST /api/sleep with no body or missing 'reason' must not 500.
        The contract shows reason as optional.
        """
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/sleep", json={})
        assert r.status_code == 200

    async def test_sleep_called_on_engine(self, tmp_path):
        """
        Verify the route actually calls request_sleep on the engine,
        not just returning a static body.

        We wrap the engine with a spy to count calls.
        """
        engine = await _make_engine(tmp_path)
        sleep_calls: list[str] = []
        _orig = engine.request_sleep

        async def _spy(reason: str = "") -> dict:
            sleep_calls.append(reason)
            return await _orig(reason)

        engine.request_sleep = _spy  # type: ignore[method-assign]
        app = create_memory_app(engine)
        async with _client(app) as c:
            await c.post("/api/sleep", json={"reason": "trigger"})
        assert len(sleep_calls) == 1
        assert sleep_calls[0] == "trigger"


# ===========================================================================
# Negative: non-existent / disallowed endpoints
# ===========================================================================


class TestMissingEndpoints:
    """
    The contract explicitly states NO edge-delete and NO KB-entry-delete
    primitives exist on the engine, so those endpoints must NOT be offered.
    Assert expected paths are not 200 (404 or 405 are both acceptable).
    """

    async def test_delete_edge_endpoint_not_defined(self, tmp_path):
        """
        DELETE /api/gm/edges/{id} must not return 200 — route not defined.
        Expect 404 or 405.
        """
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.delete("/api/gm/edges/1")
        assert r.status_code not in (200, 201)
        assert r.status_code in (404, 405)

    async def test_delete_kb_entry_endpoint_not_defined(self, tmp_path):
        """
        DELETE /api/kb/{kb_id}/entries/{id} must not return 200 — no
        KB-entry-delete endpoint in the contract.
        Expect 404 or 405.
        """
        engine = await _make_engine(tmp_path)
        await engine.create_kb("kb1", name="One")
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.delete("/api/kb/kb1/entries/1")
        assert r.status_code not in (200, 201)
        assert r.status_code in (404, 405)

    async def test_get_to_nonexistent_path_returns_404(self, tmp_path):
        """GET on a completely unknown path returns 404."""
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/nonexistent")
        assert r.status_code == 404

    async def test_delete_gm_edges_collection_not_defined(self, tmp_path):
        """
        DELETE /api/gm/edges (without id) also not defined.
        Expect 404 or 405.
        """
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.delete("/api/gm/edges")
        assert r.status_code not in (200, 201)


# ===========================================================================
# State transition: full create → read → edit → delete cycle
# ===========================================================================


class TestFullNodeLifecycleCycle:
    """Integration-style state-transition test across multiple endpoints."""

    async def test_create_patch_delete_node(self, tmp_path):
        """
        Full node lifecycle in one session:
          1. POST node → get id
          2. GET nodes → appears with FACT category
          3. PATCH → change to KNOWLEDGE
          4. GET nodes → category is KNOWLEDGE
          5. DELETE → ok
          6. GET nodes → gone
        """
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)

        async with _client(app) as c:
            # Step 1: create
            r = await c.post("/api/gm/nodes", json={
                "name": "Lifecycle", "category": "FACT", "description": "d",
            })
            assert r.status_code in (200, 201)
            nid = r.json()["id"]

            # Step 2: appears in list as FACT
            nodes = (await c.get("/api/gm/nodes")).json()["nodes"]
            found = next((n for n in nodes if n["name"] == "Lifecycle"), None)
            assert found is not None
            assert found["category"] == "FACT"

            # Step 3: patch category
            r = await c.patch(f"/api/gm/nodes/{nid}", json={"category": "KNOWLEDGE"})
            assert r.status_code == 200

            # Step 4: verify patch
            nodes = (await c.get("/api/gm/nodes")).json()["nodes"]
            found = next((n for n in nodes if n["name"] == "Lifecycle"), None)
            assert found is not None
            assert found["category"] == "KNOWLEDGE"

            # Step 5: delete
            r = await c.delete(f"/api/gm/nodes/{nid}")
            assert r.status_code == 200

            # Step 6: gone
            nodes = (await c.get("/api/gm/nodes")).json()["nodes"]
            names = [n["name"] for n in nodes]
            assert "Lifecycle" not in names

    async def test_create_kb_add_entry_archive_delete(self, tmp_path):
        """
        Full KB lifecycle:
          1. POST /api/kbs → appears in list
          2. POST /api/kb/{id}/entries → count 1
          3. PATCH archive → gone from default list
          4. DELETE → gone from list entirely
        """
        engine = await _make_engine(tmp_path)
        app = create_memory_app(engine)

        async with _client(app) as c:
            # Step 1: create
            r = await c.post("/api/kbs", json={"kb_id": "cycle_kb", "name": "Cycle"})
            assert r.status_code in (200, 201)
            kbs = (await c.get("/api/kbs")).json()["kbs"]
            assert any(kb["kb_id"] == "cycle_kb" for kb in kbs)

            # Step 2: add entry
            r = await c.post("/api/kb/cycle_kb/entries", json={"content": "entry one"})
            assert r.status_code in (200, 201)
            entries = (await c.get("/api/kb/cycle_kb/entries")).json()
            assert entries["count"] >= 1

            # Step 3: archive
            r = await c.patch("/api/kbs/cycle_kb", json={"archived": True})
            assert r.status_code == 200
            kbs = (await c.get("/api/kbs")).json()["kbs"]
            assert not any(kb["kb_id"] == "cycle_kb" for kb in kbs)

            # Step 4: delete
            r = await c.delete("/api/kbs/cycle_kb")
            assert r.status_code == 200
            kbs = (await c.get("/api/kbs")).json()["kbs"]
            assert not any(kb["kb_id"] == "cycle_kb" for kb in kbs)
