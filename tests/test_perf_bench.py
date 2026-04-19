"""Phase 3 / G: GM perf benchmark — measure + recommend soft_limit."""
import pytest

from src.tools.perf_bench import measure_at, recommend_soft_limit


async def test_measure_at_returns_expected_keys(tmp_path):
    r = await measure_at(20, dim=8, query_repeats=3, db_path=":memory:")
    for key in ("n", "insert_per_node_ms", "vec_search_ms_p50",
                 "vec_search_ms_p95", "fts_search_ms_p50"):
        assert key in r
    assert r["n"] == 20


async def test_measure_at_n_zero_returns_safe_defaults():
    r = await measure_at(0, dim=4, query_repeats=2)
    assert r["n"] == 0
    assert r["insert_per_node_ms"] == 0


def test_recommend_picks_largest_under_target():
    measurements = [
        {"n": 50, "vec_search_ms_p95": 5.0},
        {"n": 100, "vec_search_ms_p95": 15.0},
        {"n": 200, "vec_search_ms_p95": 80.0},
        {"n": 400, "vec_search_ms_p95": 250.0},  # over 200ms
        {"n": 800, "vec_search_ms_p95": 600.0},
    ]
    assert recommend_soft_limit(measurements, target_p95_ms=200) == 200


def test_recommend_returns_max_when_all_under_target():
    measurements = [
        {"n": 100, "vec_search_ms_p95": 5.0},
        {"n": 200, "vec_search_ms_p95": 8.0},
    ]
    assert recommend_soft_limit(measurements, target_p95_ms=200) == 200


def test_recommend_returns_none_when_all_over_target():
    measurements = [
        {"n": 50, "vec_search_ms_p95": 250.0},
        {"n": 100, "vec_search_ms_p95": 500.0},
    ]
    assert recommend_soft_limit(measurements, target_p95_ms=200) is None


def test_recommend_empty_measurements_returns_none():
    assert recommend_soft_limit([], target_p95_ms=200) is None
