"""Phase 1.2d: explicit_write (LLM extraction)."""
import json

import pytest

from src.memory.graph_memory import GraphMemory


class ScriptedLLM:
    def __init__(self, response):
        self._response = response
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return self._response


class FakeEmbedder:
    async def __call__(self, text: str) -> list[float]:
        return [0.0]


async def _gm(tmp_path, llm):
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=FakeEmbedder(),
                      extractor_llm=llm)
    await gm.initialize()
    return gm


async def test_explicit_nodes_marked_classified(tmp_path):
    """Phase 1 fix: explicit nodes already have category set, so they should
    be flagged classified=1 to keep classify_and_link_pending from spinning
    on them and to make stats clearer.
    """
    llm = ScriptedLLM(json.dumps({
        "nodes": [{"name": "n", "category": "FACT", "description": ""}],
        "edges": [],
    }))
    gm = await _gm(tmp_path, llm)
    res = await gm.explicit_write("x")
    node = await gm.get_node(res["node_ids"][0])
    assert node["metadata"].get("classified") is True
    await gm.close()


async def test_writes_single_node_from_llm(tmp_path):
    llm = ScriptedLLM(json.dumps({
        "nodes": [{"name": "User likes tea", "category": "FACT",
                    "description": "User prefers tea over coffee."}],
        "edges": [],
    }))
    gm = await _gm(tmp_path, llm)
    result = await gm.explicit_write("User said: I prefer tea.")

    assert len(result["node_ids"]) == 1
    node = await gm.get_node(result["node_ids"][0])
    assert node["name"] == "User likes tea"
    assert node["category"] == "FACT"
    assert node["source_type"] == "explicit"
    await gm.close()


async def test_writes_nodes_and_edges(tmp_path):
    llm = ScriptedLLM(json.dumps({
        "nodes": [
            {"name": "tea", "category": "FACT", "description": "a drink"},
            {"name": "caffeine", "category": "FACT", "description": "stimulant"},
        ],
        "edges": [
            {"source_name": "tea", "target_name": "caffeine",
             "predicate": "CAUSES"},
        ],
    }))
    gm = await _gm(tmp_path, llm)
    await gm.explicit_write("Tea contains caffeine.")

    assert await gm.count_nodes() == 2
    assert await gm.count_edges() == 1
    await gm.close()


async def test_high_importance_boosts_value(tmp_path):
    llm_high = ScriptedLLM(json.dumps({
        "nodes": [{"name": "n1", "category": "FACT", "description": ""}],
        "edges": [],
    }))
    llm_norm = ScriptedLLM(json.dumps({
        "nodes": [{"name": "n2", "category": "FACT", "description": ""}],
        "edges": [],
    }))

    (tmp_path / "h").mkdir()
    (tmp_path / "n").mkdir()
    gm_h = await _gm(tmp_path / "h", llm_high)
    gm_n = await _gm(tmp_path / "n", llm_norm)

    hr = await gm_h.explicit_write("x", importance="high")
    nr = await gm_n.explicit_write("x", importance="normal")

    h_node = await gm_h.get_node(hr["node_ids"][0])
    n_node = await gm_n.get_node(nr["node_ids"][0])

    assert h_node["importance"] > n_node["importance"]
    await gm_h.close()
    await gm_n.close()


async def test_references_existing_node_merges_via_upsert(tmp_path):
    llm = ScriptedLLM(json.dumps({
        "nodes": [{"name": "apple", "category": "FACT",
                    "description": "updated"}],
        "edges": [],
    }))
    gm = await _gm(tmp_path, llm)
    # Pre-create a node with the same name+category
    existing = await gm.insert_node(name="apple", category="FACT",
                                      description="original")

    result = await gm.explicit_write("Extra info about apple")
    assert result["node_ids"] == [existing]
    assert await gm.count_nodes() == 1

    node = await gm.get_node(existing)
    assert node["description"] == "updated"
    await gm.close()


async def test_prompt_includes_recall_context(tmp_path):
    llm = ScriptedLLM(json.dumps({"nodes": [], "edges": []}))
    gm = await _gm(tmp_path, llm)

    recall = [{"name": "prev", "category": "FACT", "description": "seen before"}]
    await gm.explicit_write("new content", recall_context=recall)

    combined = json.dumps(llm.calls[0], ensure_ascii=False)
    assert "prev" in combined
    await gm.close()


async def test_edges_skipped_when_node_not_found(tmp_path):
    """If LLM references a node name that isn't in its own node list and
    doesn't already exist, the edge is silently skipped."""
    llm = ScriptedLLM(json.dumps({
        "nodes": [{"name": "onlyone", "category": "FACT", "description": ""}],
        "edges": [
            {"source_name": "onlyone", "target_name": "ghost",
             "predicate": "RELATED_TO"},
        ],
    }))
    gm = await _gm(tmp_path, llm)
    await gm.explicit_write("x")

    assert await gm.count_nodes() == 1
    assert await gm.count_edges() == 0
    await gm.close()


async def test_tolerates_markdown_fenced_json(tmp_path):
    llm = ScriptedLLM("```json\n" + json.dumps({
        "nodes": [{"name": "n", "category": "FACT", "description": ""}],
        "edges": [],
    }) + "\n```")
    gm = await _gm(tmp_path, llm)
    result = await gm.explicit_write("x")
    assert len(result["node_ids"]) == 1
    await gm.close()
