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


async def test_recall_follows_kb_index_node(tmp_path):
    """When a recalled GM node carries metadata is_kb_index=true, the
    tentacle should also pull top entries from that KB."""
    from src.memory.knowledge_base import KBRegistry

    embed = MapEmbedder({"astronomy": [1.0, 0.0]})
    gm = await _gm(tmp_path, embed)

    # GM index node pointing to KB
    await gm.insert_node(
        name="astronomy KB", category="KNOWLEDGE",
        description="Index of astronomy knowledge.",
        embedding=[1.0, 0.0],
        metadata={"is_kb_index": True, "kb_id": "astronomy"},
    )

    reg = KBRegistry(gm, kb_dir=tmp_path / "kbs", embedder=embed)
    kb = await reg.create_kb("astronomy", name="Astronomy")
    await kb.write_entry("Sun is a yellow dwarf star",
                          embedding=[0.95, 0.31])

    from src.tentacles.memory_recall import MemoryRecallTentacle
    t = MemoryRecallTentacle(gm=gm, embedder=embed, kb_registry=reg)
    stim = await t.execute("astronomy", {})

    assert "astronomy KB" in stim.content
    assert "Sun is a yellow dwarf star" in stim.content
    await reg.close_all()
    await gm.close()


async def test_recall_with_kb_id_param_queries_that_kb_directly(tmp_path):
    """Self → params={'kb_id': 'X', 'query': '...'} bypasses GM and
    queries KB X directly."""
    from src.memory.knowledge_base import KBRegistry

    embed = MapEmbedder({"sun": [1.0, 0.0]})
    gm = await _gm(tmp_path, embed)
    reg = KBRegistry(gm, kb_dir=tmp_path / "kbs", embedder=embed)
    kb = await reg.create_kb("astronomy", name="Astronomy")
    await kb.write_entry("Sun mass: 2e30 kg", embedding=[1.0, 0.0])
    await kb.write_entry("Earth orbits Sun", embedding=[0.95, 0.31])

    from src.tentacles.memory_recall import MemoryRecallTentacle
    t = MemoryRecallTentacle(gm=gm, embedder=embed, kb_registry=reg)
    stim = await t.execute("look up sun in astronomy",
                              {"kb_id": "astronomy", "query": "sun"})

    assert "Sun mass: 2e30" in stim.content or "Earth orbits Sun" in stim.content
    await reg.close_all()
    await gm.close()


async def test_recall_without_kb_registry_still_works(tmp_path):
    """Tentacle constructed without kb_registry behaves like Phase 1."""
    embed = MapEmbedder({"x": [1.0, 0.0]})
    gm = await _gm(tmp_path, embed)
    await gm.insert_node(name="apple", category="FACT", description="",
                          embedding=[1.0, 0.0])

    from src.tentacles.memory_recall import MemoryRecallTentacle
    t = MemoryRecallTentacle(gm=gm, embedder=embed)  # no kb_registry
    stim = await t.execute("x", {})
    assert "apple" in stim.content
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
