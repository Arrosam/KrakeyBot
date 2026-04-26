"""Phase 2.3c: rebuild GM Index Graph after Sleep migration."""
import json

import pytest

from src.memory.graph_memory import GraphMemory
from src.memory.knowledge_base import KBRegistry
from src.memory.sleep.index_rebuild import rebuild_index_graph


class FixedEmbed:
    def __init__(self, mapping=None):
        self._m = mapping or {}

    async def __call__(self, text):
        return list(self._m.get(text, [0.0, 0.0]))


class ScriptedLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return self.response


class NeverCallLLM:
    async def chat(self, messages, **kwargs):
        raise AssertionError("LLM should not be called")


async def _setup(tmp_path):
    embed = FixedEmbed()
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=embed)
    await gm.initialize()
    reg = KBRegistry(gm, kb_dir=tmp_path / "kbs", embedder=embed)
    return gm, reg


# ---------------- empty / single KB ----------------

async def test_no_kbs_creates_no_index_nodes(tmp_path):
    gm, reg = await _setup(tmp_path)
    res = await rebuild_index_graph(gm, reg, llm=NeverCallLLM(),
                                       embedder=FixedEmbed())
    assert res["index_nodes"] == 0
    assert res["edges_added"] == 0
    assert await gm.count_nodes() == 0
    await reg.close_all()
    await gm.close()


async def test_single_kb_creates_one_index_node(tmp_path):
    gm, reg = await _setup(tmp_path)
    kb = await reg.create_kb("astro", name="Astronomy",
                                description="planets and stars")
    await kb.write_entry("Sun is a star")

    res = await rebuild_index_graph(gm, reg, llm=NeverCallLLM(),
                                       embedder=FixedEmbed())
    assert res["index_nodes"] == 1
    assert res["edges_added"] == 0  # only 1 KB, no relations LLM call

    nodes = await gm.list_nodes(category="KNOWLEDGE")
    assert len(nodes) == 1
    n = nodes[0]
    assert n["metadata"].get("is_kb_index") is True
    assert n["metadata"].get("kb_id") == "astro"
    assert n["metadata"].get("entry_count") == 1
    await reg.close_all()
    await gm.close()


# ---------------- multiple KBs + LLM relations ----------------

async def test_multiple_kbs_invoke_llm_for_inter_kb_relations(tmp_path):
    gm, reg = await _setup(tmp_path)
    await reg.create_kb("astro", name="Astronomy", description="space")
    await reg.create_kb("biology", name="Biology", description="life")

    llm = ScriptedLLM(json.dumps({
        "edges": [{"source_kb_id": "astro", "target_kb_id": "biology",
                   "predicate": "RELATED_TO"}],
    }))
    res = await rebuild_index_graph(gm, reg, llm=llm,
                                       embedder=FixedEmbed())
    assert res["index_nodes"] == 2
    assert res["edges_added"] == 1
    assert len(llm.calls) == 1
    # Check edge exists
    assert await gm.count_edges() == 1
    await reg.close_all()
    await gm.close()


async def test_index_node_upsert_idempotent(tmp_path):
    """Running rebuild twice should not duplicate KNOWLEDGE index nodes."""
    gm, reg = await _setup(tmp_path)
    kb = await reg.create_kb("astro", name="Astronomy", description="space")
    await kb.write_entry("entry1")

    await rebuild_index_graph(gm, reg, llm=NeverCallLLM(),
                                 embedder=FixedEmbed())
    nodes_first = await gm.list_nodes(category="KNOWLEDGE")
    assert len(nodes_first) == 1

    # Add an entry; re-run rebuild
    await kb.write_entry("entry2")
    await rebuild_index_graph(gm, reg, llm=NeverCallLLM(),
                                 embedder=FixedEmbed())
    nodes_second = await gm.list_nodes(category="KNOWLEDGE")
    assert len(nodes_second) == 1  # still one
    # entry_count updated
    assert nodes_second[0]["metadata"]["entry_count"] == 2
    await reg.close_all()
    await gm.close()


async def test_llm_returns_unknown_kb_id_skipped(tmp_path):
    gm, reg = await _setup(tmp_path)
    await reg.create_kb("astro", name="Astronomy", description="")
    await reg.create_kb("biology", name="Biology", description="")
    llm = ScriptedLLM(json.dumps({
        "edges": [{"source_kb_id": "astro", "target_kb_id": "ghost_kb",
                   "predicate": "RELATED_TO"}],
    }))
    res = await rebuild_index_graph(gm, reg, llm=llm,
                                       embedder=FixedEmbed())
    # Ghost target → no edge created
    assert res["edges_added"] == 0
    await reg.close_all()
    await gm.close()


async def test_kb_registry_entry_count_updated(tmp_path):
    gm, reg = await _setup(tmp_path)
    kb = await reg.create_kb("k", name="K", description="")
    await kb.write_entry("a")
    await kb.write_entry("b")

    await rebuild_index_graph(gm, reg, llm=NeverCallLLM(),
                                 embedder=FixedEmbed())
    metas = await reg.list_kbs()
    assert metas[0]["entry_count"] == 2
    await reg.close_all()
    await gm.close()
