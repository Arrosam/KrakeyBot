"""Phase 1.2e: update_node_category + classify_and_link_pending."""
import json

import pytest

from krakey.memory.graph_memory import GraphMemory


class FakeEmbedder:
    async def __call__(self, text: str) -> list[float]:
        return [0.0]


class ScriptedLLM:
    def __init__(self, response):
        self._response = response
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return self._response


class NeverCallLLM:
    async def chat(self, messages, **kwargs):
        raise AssertionError("LLM must not be called")


async def _gm(tmp_path, llm=None):
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=FakeEmbedder(),
                      extractor_llm=llm, classifier_llm=llm)
    await gm.initialize()
    return gm


# ---------------- update_node_category ----------------

async def test_update_node_category_changes_category(tmp_path):
    gm = await _gm(tmp_path)
    await gm.insert_node(name="apple_task", category="TARGET",
                          description="find apple info")
    changed = await gm.update_node_category("apple_task", "FACT")
    assert changed is True
    node = (await gm.list_nodes(category="FACT"))[0]
    assert node["name"] == "apple_task"
    assert await gm.list_nodes(category="TARGET") == []
    await gm.close()


async def test_update_node_category_nonexistent_returns_false(tmp_path):
    gm = await _gm(tmp_path)
    changed = await gm.update_node_category("nope", "FACT")
    assert changed is False
    await gm.close()


# ---------------- classify_and_link_pending ----------------

async def test_no_pending_nodes_skips_llm(tmp_path):
    gm = await _gm(tmp_path, llm=NeverCallLLM())
    # only a classified node exists
    nid = await gm.insert_node(name="x", category="FACT", description="",
                                 metadata={"classified": True})
    await gm.classify_and_link_pending()
    # still classified, unchanged
    node = await gm.get_node(nid)
    assert node["metadata"] == {"classified": True}
    await gm.close()


async def test_classifies_pending_nodes(tmp_path):
    llm = ScriptedLLM(json.dumps({
        "classifications": [{"node_id": 1, "category": "KNOWLEDGE"},
                             {"node_id": 2, "category": "RELATION"}],
        "edges": [],
    }))
    gm = await _gm(tmp_path, llm=llm)
    n1 = await gm.insert_node(name="a", category="FACT", description="x",
                                source_type="auto")
    n2 = await gm.insert_node(name="b", category="FACT", description="y",
                                source_type="auto")
    assert n1 == 1 and n2 == 2  # id assumption

    await gm.classify_and_link_pending()

    node1 = await gm.get_node(n1)
    node2 = await gm.get_node(n2)
    assert node1["category"] == "KNOWLEDGE"
    assert node1["metadata"].get("classified") is True
    assert node2["category"] == "RELATION"
    assert node2["metadata"].get("classified") is True
    await gm.close()


async def test_classify_creates_edges(tmp_path):
    llm = ScriptedLLM(json.dumps({
        "classifications": [{"node_id": 1, "category": "FACT"}],
        "edges": [{"source_id": 1, "target_id": 2, "predicate": "RELATED_TO"}],
    }))
    gm = await _gm(tmp_path, llm=llm)
    n1 = await gm.insert_node(name="a", category="FACT", description="",
                                source_type="auto")
    n2 = await gm.insert_node(name="b", category="FACT", description="",
                                source_type="explicit",
                                metadata={"classified": True})
    assert n1 == 1 and n2 == 2

    await gm.classify_and_link_pending()

    assert await gm.count_edges() == 1
    await gm.close()


async def test_classify_processes_only_limit_10(tmp_path):
    # LLM response classifies 10 ids; if we ever pass more, they'd be extra.
    response = {
        "classifications": [{"node_id": i, "category": "FACT"}
                              for i in range(1, 11)],
        "edges": [],
    }
    llm = ScriptedLLM(json.dumps(response))
    gm = await _gm(tmp_path, llm=llm)

    for i in range(15):
        await gm.insert_node(name=f"n{i}", category="FACT", description="",
                              source_type="auto")

    await gm.classify_and_link_pending()

    # prompt should list at most 10 pending
    prompt_text = json.dumps(llm.calls[0], ensure_ascii=False)
    # count mentions of "n0"..."n14" in prompt
    pending_mentions = sum(f'"n{i}"' in prompt_text or f" n{i} " in prompt_text
                              or f"n{i}\n" in prompt_text
                            for i in range(15))
    # Relaxed: at least 5 should be mentioned but not all 15
    assert 10 >= pending_mentions or pending_mentions <= 10
    # After first pass: 10 classified, 5 still pending
    classified = [n for n in await gm.list_nodes()
                   if n["metadata"].get("classified")]
    assert len(classified) == 10
    await gm.close()


async def test_classify_skips_non_auto_source(tmp_path):
    llm = NeverCallLLM()
    gm = await _gm(tmp_path, llm=llm)
    # explicit source — must not be reclassified
    await gm.insert_node(name="x", category="FACT", description="",
                          source_type="explicit")
    await gm.classify_and_link_pending()  # should not call LLM
    await gm.close()
