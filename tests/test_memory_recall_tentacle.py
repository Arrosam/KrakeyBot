"""Phase 1 extension: memory_recall tentacle — Self-initiated recall."""
import pytest

from src.memory.graph_memory import GraphMemory
from src.models.stimulus import Stimulus
from src.tentacles.memory_recall import MemoryRecallTentacle


class MapEmbedder:
    def __init__(self, mapping=None):
        self._m = mapping or {}
        self.calls = []

    async def __call__(self, text):
        self.calls.append(text)
        return list(self._m.get(text, [0.0, 0.0]))


async def _gm(tmp_path, embedder):
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=embedder)
    await gm.initialize()
    return gm


def _interface_check(t):
    assert t.name == "memory_recall"
    assert t.description
    assert isinstance(t.parameters_schema, dict)


def test_tentacle_metadata():
    t = MemoryRecallTentacle(gm=None, embedder=None)
    _interface_check(t)


async def test_returns_matching_nodes_via_vector_search(tmp_path):
    embed = MapEmbedder({"apple": [1.0, 0.0],
                          "tell me about apple": [1.0, 0.0]})
    gm = await _gm(tmp_path, embed)
    await gm.insert_node(name="apple", category="FACT",
                          description="red fruit",
                          embedding=[1.0, 0.0])
    await gm.insert_node(name="car", category="FACT",
                          description="vehicle",
                          embedding=[0.0, 1.0])

    t = MemoryRecallTentacle(gm=gm, embedder=embed)
    stim = await t.execute("tell me about apple", {})

    assert isinstance(stim, Stimulus)
    assert stim.type == "tentacle_feedback"
    assert stim.source == "tentacle:memory_recall"
    assert "apple" in stim.content
    assert "car" not in stim.content  # different vec, filtered
    await gm.close()


async def test_falls_back_to_fts_when_embedder_fails(tmp_path):
    class FailingEmbed:
        async def __call__(self, text):
            raise RuntimeError("embed down")

    gm = await _gm(tmp_path, FailingEmbed())
    await gm.insert_node(name="banana", category="FACT",
                          description="yellow fruit")

    t = MemoryRecallTentacle(gm=gm, embedder=FailingEmbed())
    stim = await t.execute("banana", {})
    assert "banana" in stim.content
    await gm.close()


async def test_empty_gm_returns_clear_message(tmp_path):
    embed = MapEmbedder()
    gm = await _gm(tmp_path, embed)
    t = MemoryRecallTentacle(gm=gm, embedder=embed)
    stim = await t.execute("anything", {})
    assert "no" in stim.content.lower() or "无" in stim.content or "empty" in stim.content.lower()
    await gm.close()


async def test_includes_neighbors_and_edges(tmp_path):
    embed = MapEmbedder({"apple": [1.0, 0.0]})
    gm = await _gm(tmp_path, embed)
    a = await gm.insert_node(name="apple", category="FACT", description="",
                                embedding=[1.0, 0.0])
    f = await gm.insert_node(name="fruit", category="KNOWLEDGE",
                                description="", embedding=[0.99, 0.14])
    await gm.insert_edge_with_cycle_check(a, f, "RELATED_TO")

    t = MemoryRecallTentacle(gm=gm, embedder=embed)
    stim = await t.execute("apple", {})
    assert "fruit" in stim.content
    assert "RELATED_TO" in stim.content
    await gm.close()


async def test_top_k_param_caps_results(tmp_path):
    embed = MapEmbedder({"q": [1.0, 0.0]})
    gm = await _gm(tmp_path, embed)
    for i in range(8):
        await gm.insert_node(name=f"n{i}", category="FACT", description="",
                              embedding=[1.0, i * 0.01])

    t = MemoryRecallTentacle(gm=gm, embedder=embed)
    stim = await t.execute("q", {"top_k": 3})
    found = [name for name in (f"n{i}" for i in range(8))
              if name in stim.content]
    assert len(found) == 3
    await gm.close()
