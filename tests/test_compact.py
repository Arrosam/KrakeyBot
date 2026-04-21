"""Phase 1.4: compact — evict oldest rounds into GM when window overflows."""
import json

import pytest

from src.memory.graph_memory import GraphMemory
from src.runtime.compact import compact_if_needed
from src.runtime.sliding_window import SlidingWindow, SlidingWindowRound


class Embed:
    async def __call__(self, text): return [0.0]


class ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        if not self._responses:
            return json.dumps({"nodes": [], "edges": []})
        return self._responses.pop(0)


class NeverCallLLM:
    async def chat(self, messages, **kwargs):
        raise AssertionError("compact must not run")


async def _gm(tmp_path):
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=Embed())
    await gm.initialize()
    return gm


def _recall_none():
    async def _r(text):
        return []
    return _r


# ---------------- no-op path ----------------

async def test_compact_noop_when_window_within_limit(tmp_path):
    gm = await _gm(tmp_path)
    w = SlidingWindow(max_tokens=4096)
    w.append(SlidingWindowRound(1, "small", "small", ""))
    await compact_if_needed(w, gm, NeverCallLLM(), recall_fn=_recall_none())
    assert len(w.rounds) == 1
    await gm.close()


# ---------------- basic compact ----------------

async def test_compact_evicts_and_writes_nodes_to_gm(tmp_path):
    gm = await _gm(tmp_path)
    llm = ScriptedLLM([json.dumps({
        "nodes": [{"name": "user likes tea", "category": "FACT",
                    "description": "user prefers tea"}],
        "edges": [],
    })])

    w = SlidingWindow(max_tokens=5)  # tight limit
    w.append(SlidingWindowRound(1, "user: I like tea", "reply greet", ""))
    w.append(SlidingWindowRound(2, "user: thanks", "no action", ""))
    assert w.needs_compact() is True

    await compact_if_needed(w, gm, llm, recall_fn=_recall_none())

    assert await gm.count_nodes() >= 1
    names = [n["name"] for n in await gm.list_nodes()]
    assert "user likes tea" in names
    await gm.close()


async def test_compact_writes_edges_among_extracted_nodes(tmp_path):
    gm = await _gm(tmp_path)
    llm = ScriptedLLM([json.dumps({
        "nodes": [
            {"name": "tea", "category": "FACT", "description": ""},
            {"name": "caffeine", "category": "FACT", "description": ""},
        ],
        "edges": [{"source_name": "tea", "target_name": "caffeine",
                    "predicate": "CAUSES"}],
    })])

    w = SlidingWindow(max_tokens=5)
    w.append(SlidingWindowRound(1, "a" * 40, "b" * 40, ""))
    w.append(SlidingWindowRound(2, "short", "", ""))

    await compact_if_needed(w, gm, llm, recall_fn=_recall_none())

    assert await gm.count_nodes() >= 2
    assert await gm.count_edges() >= 1
    await gm.close()


async def test_compact_loops_until_under_limit(tmp_path):
    gm = await _gm(tmp_path)
    # Always return empty extraction — pure eviction behavior
    llm = ScriptedLLM([json.dumps({"nodes": [], "edges": []})] * 10)

    w = SlidingWindow(max_tokens=10)
    for i in range(5):
        w.append(SlidingWindowRound(i, "x" * 200, "y" * 200, ""))
    assert w.needs_compact()

    await compact_if_needed(w, gm, llm, recall_fn=_recall_none())

    # After compact, window either under limit, or only 1 round left
    # (single-round branch would further split — tested separately).
    assert len(w.rounds) <= 1 or not w.needs_compact()
    await gm.close()


async def test_compact_references_existing_nodes_from_recall(tmp_path):
    gm = await _gm(tmp_path)
    # Pre-seed a node; recall should surface it
    await gm.insert_node(name="tea", category="FACT", description="drink")

    llm = ScriptedLLM([json.dumps({"nodes": [], "edges": []})])

    async def recall_fn(text):
        return await gm.fts_search(text, top_k=5)

    w = SlidingWindow(max_tokens=5)
    w.append(SlidingWindowRound(1, "tea " * 40, "", ""))
    w.append(SlidingWindowRound(2, "short", "", ""))

    await compact_if_needed(w, gm, llm, recall_fn=recall_fn)

    prompt = json.dumps(llm.calls[0], ensure_ascii=False)
    assert "tea" in prompt  # existing node included as reference
    await gm.close()


async def test_compact_tolerates_markdown_fenced_json(tmp_path):
    gm = await _gm(tmp_path)
    llm = ScriptedLLM(["```json\n" + json.dumps({
        "nodes": [{"name": "n", "category": "FACT", "description": ""}],
        "edges": [],
    }) + "\n```"])

    w = SlidingWindow(max_tokens=5)
    w.append(SlidingWindowRound(1, "a" * 50, "", ""))
    w.append(SlidingWindowRound(2, "b", "", ""))

    await compact_if_needed(w, gm, llm, recall_fn=_recall_none())
    assert await gm.count_nodes() >= 1
    await gm.close()


# ---------------- single oversized round → split ----------------

async def test_single_oversized_round_is_split_into_chunks(tmp_path):
    gm = await _gm(tmp_path)
    # LLM returns one node per chunk
    responses = [json.dumps({
        "nodes": [{"name": f"chunk_{i}", "category": "FACT", "description": ""}],
        "edges": [],
    }) for i in range(20)]
    llm = ScriptedLLM(responses)

    w = SlidingWindow(max_tokens=20)  # very tight
    big_text = "x" * 1000
    w.append(SlidingWindowRound(1, big_text, big_text, ""))
    assert w.needs_compact()
    assert len(w.rounds) == 1  # only one round but oversized

    await compact_if_needed(w, gm, llm, recall_fn=_recall_none(),
                              split_chunk_tokens=100)

    # Multiple chunks → multiple LLM calls → multiple nodes
    assert len(llm.calls) >= 2
    nodes = await gm.list_nodes()
    assert len(nodes) >= 2
    await gm.close()
