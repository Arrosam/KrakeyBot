"""memory_recall tool — Self-initiated recall."""
import pytest

from krakey.engines.memory.default import GraphMemoryEngine
from krakey.models.stimulus import Stimulus
from krakey.plugins.recall.tool import MemoryRecallTool


class MapEmbedder:
    def __init__(self, mapping=None):
        self._m = mapping or {}
        self.calls = []

    async def __call__(self, text):
        self.calls.append(text)
        return list(self._m.get(text, [0.0, 0.0]))


async def _memory(tmp_path, embedder):
    eng = GraphMemoryEngine(
        db_path=str(tmp_path / "gm.sqlite"),
        embedder=embedder,
        kb_dir=str(tmp_path / "kbs"),
    )
    await eng.initialize()
    return eng


def _interface_check(t):
    assert t.name == "memory_recall"
    assert t.description
    assert isinstance(t.parameters_schema, dict)


def test_tool_metadata():
    t = MemoryRecallTool(memory=None, embedder=None)
    _interface_check(t)


async def test_returns_matching_nodes_via_vector_search(tmp_path):
    embed = MapEmbedder({"apple": [1.0, 0.0],
                          "tell me about apple": [1.0, 0.0]})
    mem = await _memory(tmp_path, embed)
    await mem.insert_node(name="apple", category="FACT",
                          description="red fruit",
                          embedding=[1.0, 0.0])
    await mem.insert_node(name="car", category="FACT",
                          description="vehicle",
                          embedding=[0.0, 1.0])

    t = MemoryRecallTool(memory=mem, embedder=embed)
    stim = await t.execute("tell me about apple", {})

    assert isinstance(stim, Stimulus)
    assert stim.type == "tool_feedback"
    assert stim.source == "tool:memory_recall"
    assert "apple" in stim.content
    assert "car" not in stim.content  # different vec, filtered
    await mem.close()


async def test_falls_back_to_fts_when_embedder_fails(tmp_path):
    class FailingEmbed:
        async def __call__(self, text):
            raise RuntimeError("embed down")

    mem = await _memory(tmp_path, FailingEmbed())
    await mem.insert_node(name="banana", category="FACT",
                          description="yellow fruit")

    t = MemoryRecallTool(memory=mem, embedder=FailingEmbed())
    stim = await t.execute("banana", {})
    assert "banana" in stim.content
    await mem.close()


async def test_empty_gm_returns_clear_message(tmp_path):
    embed = MapEmbedder()
    mem = await _memory(tmp_path, embed)
    t = MemoryRecallTool(memory=mem, embedder=embed)
    stim = await t.execute("anything", {})
    assert "no" in stim.content.lower() or "empty" in stim.content.lower()
    await mem.close()


async def test_includes_neighbors_and_edges(tmp_path):
    embed = MapEmbedder({"apple": [1.0, 0.0]})
    mem = await _memory(tmp_path, embed)
    a = await mem.insert_node(name="apple", category="FACT", description="",
                                embedding=[1.0, 0.0])
    f = await mem.insert_node(name="fruit", category="KNOWLEDGE",
                                description="", embedding=[0.99, 0.14])
    await mem.insert_edge_with_cycle_check(a, f, "RELATED_TO")

    t = MemoryRecallTool(memory=mem, embedder=embed)
    stim = await t.execute("apple", {})
    assert "fruit" in stim.content
    assert "RELATED_TO" in stim.content
    await mem.close()


async def test_recall_follows_kb_index_node(tmp_path):
    """When a recalled GM node carries metadata is_kb_index=true, the
    tool should also pull top entries from that KB."""
    embed = MapEmbedder({"astronomy": [1.0, 0.0]})
    mem = await _memory(tmp_path, embed)

    # GM index node pointing to KB
    await mem.insert_node(
        name="astronomy KB", category="KNOWLEDGE",
        description="Index of astronomy knowledge.",
        embedding=[1.0, 0.0],
        metadata={"is_kb_index": True, "kb_id": "astronomy"},
    )

    kb = await mem.create_kb("astronomy", name="Astronomy")
    await kb.write_entry("Sun is a yellow dwarf star",
                          embedding=[0.95, 0.31])

    t = MemoryRecallTool(memory=mem, embedder=embed)
    stim = await t.execute("astronomy", {})

    assert "astronomy KB" in stim.content
    assert "Sun is a yellow dwarf star" in stim.content
    await mem.close()


async def test_recall_with_kb_id_param_queries_that_kb_directly(tmp_path):
    """Self → params={'kb_id': 'X', 'query': '...'} bypasses GM and
    queries KB X directly."""
    embed = MapEmbedder({"sun": [1.0, 0.0]})
    mem = await _memory(tmp_path, embed)
    kb = await mem.create_kb("astronomy", name="Astronomy")
    await kb.write_entry("Sun mass: 2e30 kg", embedding=[1.0, 0.0])
    await kb.write_entry("Earth orbits Sun", embedding=[0.95, 0.31])

    t = MemoryRecallTool(memory=mem, embedder=embed)
    stim = await t.execute("look up sun in astronomy",
                              {"kb_id": "astronomy", "query": "sun"})

    assert "Sun mass: 2e30" in stim.content or "Earth orbits Sun" in stim.content
    await mem.close()


async def test_top_k_param_caps_results(tmp_path):
    embed = MapEmbedder({"q": [1.0, 0.0]})
    mem = await _memory(tmp_path, embed)
    for i in range(8):
        await mem.insert_node(name=f"n{i}", category="FACT", description="",
                              embedding=[1.0, i * 0.01])

    t = MemoryRecallTool(memory=mem, embedder=embed)
    stim = await t.execute("q", {"top_k": 3})
    found = [name for name in (f"n{i}" for i in range(8))
              if name in stim.content]
    assert len(found) == 3
    await mem.close()
