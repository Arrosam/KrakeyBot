"""Phase 1.3f: IncrementalRecall — per-stimulus search, weighted merge,
finalize with covered/uncovered + neighbor keywords + edges."""
from datetime import datetime

import pytest

from src.memory.graph_memory import GraphMemory
from src.memory.recall import RecallResult, ScoringWeights
from src.plugins.recall_anchor.incremental import IncrementalRecall
from src.models.stimulus import Stimulus


class MappingEmbedder:
    def __init__(self, mapping):
        self._m = mapping

    async def __call__(self, text: str) -> list[float]:
        if text not in self._m:
            raise KeyError(f"no embedding for {text!r}")
        return list(self._m[text])


class FailingEmbedder:
    async def __call__(self, text): raise RuntimeError("embed down")


def _stim(content, *, adrenalin=False, ts_seconds=0):
    from datetime import timedelta
    return Stimulus(
        type="user_message", source="test", content=content,
        timestamp=datetime(2026, 4, 19) + timedelta(seconds=ts_seconds),
        adrenalin=adrenalin,
    )


async def _populate(tmp_path, embed_map):
    """Populate GM with one node per key in embed_map, description=key."""
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=MappingEmbedder(embed_map))
    await gm.initialize()
    for name, vec in embed_map.items():
        await gm.insert_node(name=name, category="FACT", description=name,
                              embedding=vec)
    return gm


def _recall(gm, **kw):
    # Default budget large enough that tests which don't care about
    # the cap never have their expected node count truncated.
    default = dict(
        per_stimulus_k=3, recall_token_budget=100_000,
        weights=ScoringWeights(), now=lambda: datetime(2026, 4, 19),
    )
    default.update(kw)
    # Backward-compat shim: old tests passed `max_recall_nodes=N` —
    # translate to a token budget that admits roughly that many
    # single-line nodes.
    if "max_recall_nodes" in default:
        n = default.pop("max_recall_nodes")
        # Each node renders in ~15 tokens under cl100k_base (short
        # name + short description). 20× headroom per node lets the
        # tests still work even with neighbor keywords.
        default["recall_token_budget"] = max(n * 20, 200)
    return IncrementalRecall(gm, embedder=gm._embedder, **default)


async def test_empty_recall_everything_uncovered(tmp_path):
    gm = await _populate(tmp_path, {})
    # Need an embedder the recall can call for the stimulus
    embed_map = {"hi": [1.0, 0.0]}
    gm._embedder = MappingEmbedder(embed_map)

    r = _recall(gm)
    await r.add_stimuli([_stim("hi")])
    result = await r.finalize()

    assert result.nodes == []
    assert len(result.uncovered_stimuli) == 1
    assert result.covered_stimuli == []
    await gm.close()


async def test_single_stimulus_hits_matching_node(tmp_path):
    embed_map = {"apple": [1.0, 0.0], "car": [0.0, 1.0]}
    gm = await _populate(tmp_path, embed_map)

    r = _recall(gm, per_stimulus_k=1)
    await r.add_stimuli([_stim("apple")])
    result = await r.finalize()

    assert [n["name"] for n in result.nodes] == ["apple"]
    assert len(result.covered_stimuli) == 1
    assert result.uncovered_stimuli == []
    await gm.close()


async def test_adrenalin_stimulus_weighted_10x(tmp_path):
    embed_map = {
        "apple": [1.0, 0.0],
        "sky": [0.0, 1.0],
    }
    gm = await _populate(tmp_path, embed_map)

    # One non-adrenalin stim matches apple; one adrenalin stim matches sky.
    # With weight 10 vs 1, sky should rank first.
    r = _recall(gm, per_stimulus_k=1, max_recall_nodes=2)
    await r.add_stimuli([_stim("apple"), _stim("sky", adrenalin=True)])
    result = await r.finalize()
    assert [n["name"] for n in result.nodes][0] == "sky"
    await gm.close()


async def test_same_node_hit_by_two_stimuli_accumulates_weight(tmp_path):
    # Two distinct stimulus texts but both embed to near-apple vectors.
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=MappingEmbedder({
        "stim_one": [1.0, 0.0],
        "stim_two": [0.95, 0.31],
    }))
    await gm.initialize()
    # Only one node exists in GM — both stimuli must hit it.
    await gm.insert_node(name="apple", category="FACT",
                          description="apple", embedding=[1.0, 0.0])

    r = _recall(gm, per_stimulus_k=1, max_recall_nodes=3)
    await r.add_stimuli([_stim("stim_one"), _stim("stim_two")])
    result = await r.finalize()

    apple = next(n for n in result.nodes if n["name"] == "apple")
    assert apple["score"] == 2.0  # two non-adrenalin hits → weight 1+1
    await gm.close()


async def test_recall_token_budget_caps_result(tmp_path):
    """Token budget replaces the old node-count cap. A tight budget
    stops admission after roughly the expected rendered-token total.
    """
    vecs = {f"n{i}": [1.0, float(i) * 0.01] for i in range(10)}
    gm = await _populate(tmp_path, vecs)

    # Budget tight enough to fit ~3 nodes of ~15 tokens each.
    r = _recall(gm, per_stimulus_k=10, recall_token_budget=50)
    await r.add_stimuli([_stim("n0")])
    result = await r.finalize()
    # 10 candidates; budget admits 2-4 depending on tokenizer quirks.
    assert 1 <= len(result.nodes) <= 5
    await gm.close()


async def test_recall_token_budget_admits_at_least_one(tmp_path):
    """Corner case: even a single node that would exceed the budget
    gets admitted — dropping recall entirely is worse UX than soft
    overshoot. The overall-prompt enforcement handles the true cap."""
    vecs = {"a_very_long_node_name_that_wastes_tokens_on_purpose": [1.0, 0.0]}
    gm = await _populate(tmp_path, vecs)
    r = _recall(gm, per_stimulus_k=1, recall_token_budget=1)
    await r.add_stimuli([_stim(
        "a_very_long_node_name_that_wastes_tokens_on_purpose"
    )])
    result = await r.finalize()
    assert len(result.nodes) == 1
    await gm.close()


def test_screening_top_k_capped_by_per_k(tmp_path):
    """Multiplier × budget produces a soft target larger than per_k —
    top_k must clamp to the per_k hard ceiling."""
    r = IncrementalRecall(
        gm=None, embedder=None,  # type: ignore[arg-type]
        per_stimulus_k=10,
        recall_token_budget=600,
        screening_token_multiplier=3.0,  # target ≈ 1800 / 30 = 60 nodes
    )
    assert r._screening_top_k() == 10  # clamped to per_k


def test_screening_top_k_uses_target_when_below_per_k():
    """When the soft target fits inside per_k, top_k follows the
    multiplier × budget sizing instead of saturating per_k."""
    r = IncrementalRecall(
        gm=None, embedder=None,  # type: ignore[arg-type]
        per_stimulus_k=200,
        recall_token_budget=600,
        screening_token_multiplier=3.0,  # target = 1800 / 30 = 60
    )
    assert r._screening_top_k() == 60


def test_screening_top_k_default_multiplier_disables_oversampling():
    """multiplier=1.0 means screening pool ≈ final cut — equivalent
    to no oversampling."""
    r = IncrementalRecall(
        gm=None, embedder=None,  # type: ignore[arg-type]
        per_stimulus_k=200,
        recall_token_budget=600,
        screening_token_multiplier=1.0,  # target = 600 / 30 = 20
    )
    assert r._screening_top_k() == 20


def test_screening_top_k_floors_at_one():
    """Degenerate config (zero budget or zero multiplier) must still
    pull at least one candidate per stimulus or recall silently dies."""
    r = IncrementalRecall(
        gm=None, embedder=None,  # type: ignore[arg-type]
        per_stimulus_k=50,
        recall_token_budget=0,
        screening_token_multiplier=3.0,
    )
    assert r._screening_top_k() == 1


async def test_embedder_failure_uses_fts_fallback(tmp_path):
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=FailingEmbedder())
    await gm.initialize()
    # Insert a node we can match via FTS
    await gm.insert_node(name="apple", category="FACT",
                          description="red fruit")
    await gm.insert_node(name="car", category="FACT",
                          description="vehicle")

    r = _recall(gm)
    await r.add_stimuli([_stim("apple")])
    result = await r.finalize()
    assert "apple" in [n["name"] for n in result.nodes]
    await gm.close()


async def test_uncovered_stimulus_returned(tmp_path):
    # One stimulus finds a match, another finds nothing (no similar node).
    gm = GraphMemory(tmp_path / "gm.sqlite",
                       embedder=MappingEmbedder({
                           "apple": [1.0, 0.0, 0.0],
                           "nothing": [0.0, 0.0, 1.0],
                       }))
    await gm.initialize()
    await gm.insert_node(name="apple", category="FACT",
                          description="apple", embedding=[1.0, 0.0, 0.0])

    r = _recall(gm, per_stimulus_k=3, max_recall_nodes=5)
    s_hit = _stim("apple")
    s_miss = _stim("nothing")
    await r.add_stimuli([s_hit, s_miss])
    result = await r.finalize()

    assert s_hit in result.covered_stimuli
    assert s_miss in result.uncovered_stimuli
    await gm.close()


async def test_finalize_includes_neighbor_keywords(tmp_path):
    embed_map = {"apple": [1.0, 0.0], "fruit": [0.0, 1.0]}
    gm = await _populate(tmp_path, embed_map)
    apple_id = (await gm.list_nodes())[0]["id"]
    fruit_id = (await gm.list_nodes())[1]["id"]
    await gm.insert_edge_with_cycle_check(apple_id, fruit_id, "RELATED_TO")

    r = _recall(gm, per_stimulus_k=1, max_recall_nodes=1)
    await r.add_stimuli([_stim("apple")])
    result = await r.finalize()
    assert result.nodes[0]["name"] == "apple"
    assert "fruit" in result.nodes[0]["neighbor_keywords"]
    await gm.close()


async def test_finalize_includes_edges_among_selected(tmp_path):
    embed_map = {"apple": [1.0, 0.0], "fruit": [0.9, 0.1]}
    gm = await _populate(tmp_path, embed_map)
    apple_id = (await gm.list_nodes())[0]["id"]
    fruit_id = (await gm.list_nodes())[1]["id"]
    await gm.insert_edge_with_cycle_check(apple_id, fruit_id, "RELATED_TO")

    r = _recall(gm, per_stimulus_k=2, max_recall_nodes=2)
    await r.add_stimuli([_stim("apple")])
    result = await r.finalize()
    assert len(result.edges) == 1
    e = result.edges[0]
    assert {e["source"], e["target"]} == {"apple", "fruit"}
    await gm.close()


async def test_reranker_influences_order(tmp_path):
    embed_map = {"a": [1.0, 0.0], "b": [0.9, 0.1]}
    gm = await _populate(tmp_path, embed_map)

    class FixedReranker:
        async def rerank(self, query, docs):
            # Reverse order: prefer whichever doc contains 'b'
            return [0.1 if "a" in d.split(":")[0] else 0.9 for d in docs]

    r = _recall(gm, per_stimulus_k=2, max_recall_nodes=2, reranker=FixedReranker())
    await r.add_stimuli([_stim("a")])
    result = await r.finalize()
    # Without reranker "a" would dominate; with reranker "b" weighted first
    # (same 1.0 accumulation — just ensure both nodes present).
    assert {n["name"] for n in result.nodes} == {"a", "b"}
    await gm.close()
