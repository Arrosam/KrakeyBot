"""Phase 1.3a+b: vector search API + scripted weighted ranking."""
from datetime import datetime, timedelta

import pytest

from krakey.memory.recall import (
    ScoringWeights, category_weight, scripted_score, time_decay,
)


def test_category_weights():
    assert category_weight("TARGET") == 1.5
    assert category_weight("FOCUS") == 1.5
    assert category_weight("KNOWLEDGE") == 1.2
    assert category_weight("RELATION") == 1.0
    assert category_weight("FACT") == 0.8
    assert category_weight("unknown") == 1.0


def test_time_decay_fresh_is_1():
    now = datetime(2026, 4, 19)
    assert time_decay(now, now, half_life_seconds=86400) == 1.0


def test_time_decay_halves_at_half_life():
    now = datetime(2026, 4, 19)
    old = now - timedelta(seconds=86400)
    assert time_decay(old, now, half_life_seconds=86400) == pytest.approx(0.5)


def test_time_decay_monotonic():
    now = datetime(2026, 4, 19)
    d1 = time_decay(now - timedelta(hours=1), now)
    d10 = time_decay(now - timedelta(hours=10), now)
    d100 = time_decay(now - timedelta(hours=100), now)
    assert d1 > d10 > d100


def _node(**kw):
    base = {
        "id": 1, "name": "n", "category": "FACT", "description": "",
        "importance": 1.0, "access_count": 0,
        "created_at": "2026-04-19 00:00:00",
    }
    base.update(kw)
    return base


def test_scripted_score_rewards_vec_similarity():
    now = datetime(2026, 4, 19)
    weights = ScoringWeights()
    lo = scripted_score(_node(), vec_sim=0.1, now=now, weights=weights)
    hi = scripted_score(_node(), vec_sim=0.9, now=now, weights=weights)
    assert hi > lo


def test_scripted_score_rewards_recency():
    now = datetime(2026, 4, 19)
    w = ScoringWeights()
    fresh = scripted_score(_node(created_at="2026-04-19 00:00:00"),
                            vec_sim=0.5, now=now, weights=w)
    stale = scripted_score(_node(created_at="2026-04-10 00:00:00"),
                            vec_sim=0.5, now=now, weights=w)
    assert fresh > stale


def test_scripted_score_rewards_importance_and_access():
    now = datetime(2026, 4, 19)
    w = ScoringWeights()
    base = scripted_score(_node(importance=1.0, access_count=0),
                           vec_sim=0.5, now=now, weights=w)
    bumped_imp = scripted_score(_node(importance=5.0, access_count=0),
                                  vec_sim=0.5, now=now, weights=w)
    bumped_acc = scripted_score(_node(importance=1.0, access_count=100),
                                  vec_sim=0.5, now=now, weights=w)
    assert bumped_imp > base
    assert bumped_acc > base


def test_scripted_score_type_weight_differentiation():
    now = datetime(2026, 4, 19)
    w = ScoringWeights()
    target = scripted_score(_node(category="TARGET"),
                              vec_sim=0.5, now=now, weights=w)
    fact = scripted_score(_node(category="FACT"),
                            vec_sim=0.5, now=now, weights=w)
    assert target > fact


async def test_vec_search_returns_top_k_sorted_desc(tmp_path):
    from krakey.memory.graph_memory import GraphMemory

    class Embed:
        async def __call__(self, text): return [0.0]

    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=Embed())
    await gm.initialize()

    await gm.insert_node(name="parallel", category="FACT", description="",
                          embedding=[1.0, 0.0, 0.0])
    await gm.insert_node(name="orthogonal", category="FACT", description="",
                          embedding=[0.0, 1.0, 0.0])
    await gm.insert_node(name="close", category="FACT", description="",
                          embedding=[0.95, 0.3, 0.0])

    results = await gm.vec_search([1.0, 0.0, 0.0], top_k=3)
    names = [n["name"] for (n, _sim) in results]
    assert names == ["parallel", "close", "orthogonal"]

    # top_k limit
    top1 = await gm.vec_search([1.0, 0.0, 0.0], top_k=1)
    assert len(top1) == 1 and top1[0][0]["name"] == "parallel"

    # min_similarity filter
    filtered = await gm.vec_search([1.0, 0.0, 0.0], top_k=10,
                                     min_similarity=0.9)
    assert {n["name"] for (n, _s) in filtered} == {"parallel", "close"}
    await gm.close()
