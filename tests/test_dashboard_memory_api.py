"""Phase 3.F.5: REST endpoints for read-only Memory Browser."""
import asyncio

import httpx
import pytest

from src.dashboard.server import create_app
from src.memory.graph_memory import GraphMemory
from src.memory.knowledge_base import KBRegistry


class StubEmbed:
    async def __call__(self, text):
        return [0.0]


class _Stub:
    """Minimal runtime stand-in exposing gm + kb_registry."""
    def __init__(self, gm, reg):
        self.gm = gm
        self.kb_registry = reg


async def _make_runtime(tmp_path):
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=StubEmbed())
    await gm.initialize()
    reg = KBRegistry(gm, kb_dir=tmp_path / "kbs", embedder=StubEmbed())
    return _Stub(gm, reg), gm, reg


def _client(runtime):
    app = create_app(runtime=runtime)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ---------------- GM nodes ----------------

async def test_gm_nodes_endpoint_returns_list(tmp_path):
    runtime, gm, _ = await _make_runtime(tmp_path)
    await gm.insert_node(name="apple", category="FACT", description="red")
    await gm.insert_node(name="banana", category="FACT", description="yellow")
    async with _client(runtime) as c:
        r = await c.get("/api/gm/nodes")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    names = [n["name"] for n in body["nodes"]]
    assert "apple" in names and "banana" in names


async def test_gm_nodes_filter_by_category(tmp_path):
    runtime, gm, _ = await _make_runtime(tmp_path)
    await gm.insert_node(name="f1", category="FACT", description="")
    await gm.insert_node(name="t1", category="TARGET", description="")
    async with _client(runtime) as c:
        r = await c.get("/api/gm/nodes?category=TARGET")
    assert r.status_code == 200
    names = [n["name"] for n in r.json()["nodes"]]
    assert names == ["t1"]


async def test_gm_nodes_limit(tmp_path):
    runtime, gm, _ = await _make_runtime(tmp_path)
    for i in range(15):
        await gm.insert_node(name=f"n{i}", category="FACT", description="")
    async with _client(runtime) as c:
        r = await c.get("/api/gm/nodes?limit=5")
    assert len(r.json()["nodes"]) == 5


# ---------------- GM edges ----------------

async def test_gm_edges_endpoint(tmp_path):
    runtime, gm, _ = await _make_runtime(tmp_path)
    a = await gm.insert_node(name="a", category="FACT", description="")
    b = await gm.insert_node(name="b", category="FACT", description="")
    await gm.insert_edge_with_cycle_check(a, b, "RELATED_TO")
    async with _client(runtime) as c:
        r = await c.get("/api/gm/edges")
    body = r.json()
    assert body["count"] == 1
    e = body["edges"][0]
    assert {e["source"], e["target"]} == {"a", "b"}
    assert e["predicate"] == "RELATED_TO"


# ---------------- GM stats ----------------

async def test_gm_stats_endpoint(tmp_path):
    runtime, gm, _ = await _make_runtime(tmp_path)
    await gm.insert_node(name="x", category="FACT", description="")
    await gm.insert_node(name="y", category="TARGET", description="")
    async with _client(runtime) as c:
        r = await c.get("/api/gm/stats")
    body = r.json()
    assert body["total_nodes"] == 2
    assert body["by_category"]["FACT"] == 1
    assert body["by_category"]["TARGET"] == 1


# ---------------- KB registry + entries ----------------

async def test_kbs_endpoint_lists_registered(tmp_path):
    runtime, gm, reg = await _make_runtime(tmp_path)
    await reg.create_kb("astro", name="Astronomy", description="space")
    async with _client(runtime) as c:
        r = await c.get("/api/kbs")
    body = r.json()
    assert any(k["kb_id"] == "astro" for k in body["kbs"])


async def test_kb_entries_endpoint(tmp_path):
    runtime, gm, reg = await _make_runtime(tmp_path)
    kb = await reg.create_kb("astro", name="Astronomy", description="space")
    await kb.write_entry("Sun is a star", tags=["star"])
    await kb.write_entry("Earth orbits Sun")
    async with _client(runtime) as c:
        r = await c.get("/api/kb/astro/entries")
    body = r.json()
    assert body["count"] == 2
    contents = [e["content"] for e in body["entries"]]
    assert "Sun is a star" in contents


async def test_kb_unknown_id_returns_404(tmp_path):
    runtime, _, _ = await _make_runtime(tmp_path)
    async with _client(runtime) as c:
        r = await c.get("/api/kb/nonexistent/entries")
    assert r.status_code == 404


async def test_endpoints_unavailable_when_no_runtime():
    """Dashboard skeleton tests use runtime=None; memory endpoints
    should respond 503 (Service Unavailable) rather than crash."""
    app = create_app(runtime=None)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                    base_url="http://test") as c:
        r = await c.get("/api/gm/nodes")
    assert r.status_code == 503


# ---------------- prompt log ----------------

async def test_prompts_endpoint_returns_recent_prompts(tmp_path):
    runtime, _, _ = await _make_runtime(tmp_path)
    # Simulate runtime recording heartbeat prompts
    runtime._prompt_log = __import__("collections").deque(maxlen=3)
    runtime._record_prompt = lambda hb, p: runtime._prompt_log.append(
        {"heartbeat_id": hb, "ts": "2026-04-22T00:00:00", "full_prompt": p}
    )
    runtime.recent_prompts = lambda limit=None: list(reversed(list(runtime._prompt_log)))[:limit or 999]
    runtime._record_prompt(1, "hello #1")
    runtime._record_prompt(2, "hello #2")
    runtime._record_prompt(3, "hello #3")
    runtime._record_prompt(4, "hello #4")  # evicts #1 (maxlen=3)
    async with _client(runtime) as c:
        r = await c.get("/api/prompts?limit=10")
    body = r.json()
    assert body["count"] == 3
    hbs = [p["heartbeat_id"] for p in body["prompts"]]
    assert hbs == [4, 3, 2]   # newest first


async def test_prompts_endpoint_503_when_no_runtime():
    app = create_app(runtime=None)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                    base_url="http://test") as c:
        r = await c.get("/api/prompts")
    assert r.status_code == 503
