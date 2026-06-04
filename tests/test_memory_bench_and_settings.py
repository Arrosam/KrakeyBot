"""Edge tests for P2 memory engine additions: bench.py + engine settings.

Contract sections tested:
  - "Benchmark + engine settings (user-driven tuning...)" in
    contracts/memory-access/definition.md
  - "gm_node_soft_limit ownership" in same file
  - ``GraphMemoryEngine.__init__`` ``config=`` / ``config_path=`` kwargs
  - ``krakey.engines.memory._internal.bench`` pure functions
  - ``POST /api/benchmark``, ``GET /api/settings``, ``PUT /api/settings``
    endpoints on ``create_memory_app(engine)``

Tests are written BEFORE any implementation exists and target the FINAL
state — all tests are expected to be RED until the implementation lands.

Run from repo root:
    pytest tests/test_memory_bench_and_settings.py

Async plumbing: ``asyncio_mode = auto`` (see pytest.ini) — no explicit
``@pytest.mark.asyncio`` needed on individual tests.

HTTP plumbing: httpx ASGI transport, no port binding required.
"""
from __future__ import annotations

import pytest
import httpx
from fastapi import FastAPI

from krakey.engines.memory.default import GraphMemoryEngine
from krakey.engines.memory.web import create_memory_app


# ---------------------------------------------------------------------------
# Shared fake collaborators (same pattern as test_memory_web_service.py)
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Returns a small fixed-length vector for any text — no real embedding."""

    async def __call__(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class _FakeExtractorChat:
    """Returns a canned extraction JSON for any message list."""

    async def chat(self, messages, **kwargs) -> str:
        return (
            '{"nodes":[{"name":"x","category":"FACT","description":"d"}],'
            '"edges":[]}'
        )


# ---------------------------------------------------------------------------
# Engine factory helpers
# ---------------------------------------------------------------------------

# Standard config_path used in all settings-capable fixtures.
# Relative so chdir(tmp_path) makes it land under tmp.
_SETTINGS_CONFIG_PATH = "data/engines/memory/graph_memory.yaml"


async def _make_engine_with_config(
    tmp_path,
    *,
    config: dict | None = None,
    config_path: str = _SETTINGS_CONFIG_PATH,
) -> GraphMemoryEngine:
    """Build a fresh GraphMemoryEngine with config + config_path kwargs.

    Assumption: GraphMemoryEngine accepts ``config=`` and ``config_path=``
    as constructor keyword arguments (P2 additions). ``config`` is the
    engine's own settings dict (e.g. from a YAML file). ``config_path`` is
    the workspace-relative path the engine uses for GET/PUT /api/settings.
    """
    engine = GraphMemoryEngine(
        db_path=":memory:",
        embedder=_FakeEmbedder(),
        kb_dir=str(tmp_path / "kbs"),
        extractor_llm=_FakeExtractorChat(),
        config=config if config is not None else {},
        config_path=config_path,
    )
    await engine.initialize()
    return engine


def _client(app: FastAPI) -> httpx.AsyncClient:
    """Return an async httpx client wired to ``app`` via ASGI transport."""
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ===========================================================================
# 1. bench.py — unit tests for pure functions
# ===========================================================================


class TestMeasureAt:
    """
    ``measure_at(n_nodes, *, dim, query_repeats, db_path)`` — pure async
    function in ``krakey.engines.memory._internal.bench``.

    Import is deferred inside the test body because the module does not
    exist yet; importing at module level would prevent collection of other
    test classes.
    """

    # --- positive ---

    async def test_returns_dict(self):
        """measure_at returns a dict for any valid n_nodes."""
        from krakey.engines.memory._internal.bench import measure_at
        result = await measure_at(1, dim=8, query_repeats=1)
        assert isinstance(result, dict)

    async def test_all_five_keys_present_for_nonzero_n(self):
        """measure_at returns a dict with all 5 required keys for n > 0."""
        from krakey.engines.memory._internal.bench import measure_at
        result = await measure_at(2, dim=8, query_repeats=2)
        required = {
            "n",
            "insert_per_node_ms",
            "vec_search_ms_p50",
            "vec_search_ms_p95",
            "fts_search_ms_p50",
        }
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )

    async def test_n_field_matches_input(self):
        """result['n'] must equal the n_nodes argument passed in."""
        from krakey.engines.memory._internal.bench import measure_at
        result = await measure_at(2, dim=8, query_repeats=2)
        assert result["n"] == 2

    async def test_numeric_fields_are_non_negative(self):
        """All timing fields must be >= 0 (no negative latencies)."""
        from krakey.engines.memory._internal.bench import measure_at
        result = await measure_at(2, dim=8, query_repeats=2)
        for key in (
            "insert_per_node_ms",
            "vec_search_ms_p50",
            "vec_search_ms_p95",
            "fts_search_ms_p50",
        ):
            assert result[key] >= 0, f"{key} was negative: {result[key]}"

    # --- BVA: boundary values for n_nodes ---

    async def test_n_zero_returns_zeros_dict(self):
        """measure_at(0) returns zeros dict with all 5 keys per contract."""
        from krakey.engines.memory._internal.bench import measure_at
        result = await measure_at(0, dim=8, query_repeats=1)
        required = {
            "n",
            "insert_per_node_ms",
            "vec_search_ms_p50",
            "vec_search_ms_p95",
            "fts_search_ms_p50",
        }
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )
        for key in required:
            assert result[key] == 0, f"{key} should be 0 for n=0, got {result[key]}"

    async def test_n_zero_n_field_is_zero(self):
        """result['n'] == 0 when n_nodes == 0."""
        from krakey.engines.memory._internal.bench import measure_at
        result = await measure_at(0, dim=8, query_repeats=1)
        assert result["n"] == 0

    async def test_n_one_returns_valid_dict(self):
        """n_nodes == 1 (minimum nonzero) returns a valid dict."""
        from krakey.engines.memory._internal.bench import measure_at
        result = await measure_at(1, dim=8, query_repeats=1)
        assert result["n"] == 1
        assert result["vec_search_ms_p95"] >= 0

    async def test_n_two_returns_valid_dict(self):
        """n_nodes == 2 (next above minimum) returns a valid dict."""
        from krakey.engines.memory._internal.bench import measure_at
        result = await measure_at(2, dim=8, query_repeats=2)
        assert result["n"] == 2

    async def test_dim_param_accepted(self):
        """``dim`` parameter is accepted without error (no TypeError)."""
        from krakey.engines.memory._internal.bench import measure_at
        result = await measure_at(1, dim=32, query_repeats=1)
        assert isinstance(result, dict)

    async def test_db_path_memory_accepted(self):
        """``db_path=':memory:'`` (default) is accepted and runs cleanly."""
        from krakey.engines.memory._internal.bench import measure_at
        result = await measure_at(1, dim=8, query_repeats=1, db_path=":memory:")
        assert isinstance(result, dict)

    async def test_does_not_touch_live_gm(self):
        """
        measure_at is a throwaway bench — calling it twice on the same
        process produces independent results (no shared state / no
        interference with any live engine).

        Verify: two calls both return the expected n value.
        """
        from krakey.engines.memory._internal.bench import measure_at
        r1 = await measure_at(1, dim=8, query_repeats=1)
        r2 = await measure_at(2, dim=8, query_repeats=1)
        assert r1["n"] == 1
        assert r2["n"] == 2


class TestRecommendSoftLimit:
    """
    ``recommend_soft_limit(measurements, *, target_p95_ms)`` — pure
    synchronous function in ``krakey.engines.memory._internal.bench``.

    Contract: returns the largest N whose vec_search_ms_p95 <= target_p95_ms.
    Returns None if no measurement qualifies or input is empty.
    """

    # --- positive ---

    async def test_single_qualifying_measurement_returns_its_n(self):
        """One measurement under target → returns its n."""
        from krakey.engines.memory._internal.bench import recommend_soft_limit
        measurements = [
            {
                "n": 100,
                "insert_per_node_ms": 1.0,
                "vec_search_ms_p50": 10.0,
                "vec_search_ms_p95": 50.0,
                "fts_search_ms_p50": 5.0,
            }
        ]
        result = recommend_soft_limit(measurements, target_p95_ms=200.0)
        assert result == 100

    async def test_all_over_target_returns_none(self):
        """All measurements over target → returns None."""
        from krakey.engines.memory._internal.bench import recommend_soft_limit
        measurements = [
            {
                "n": 100,
                "insert_per_node_ms": 1.0,
                "vec_search_ms_p50": 10.0,
                "vec_search_ms_p95": 300.0,
                "fts_search_ms_p50": 5.0,
            },
            {
                "n": 200,
                "insert_per_node_ms": 1.5,
                "vec_search_ms_p50": 20.0,
                "vec_search_ms_p95": 500.0,
                "fts_search_ms_p50": 8.0,
            },
        ]
        result = recommend_soft_limit(measurements, target_p95_ms=200.0)
        assert result is None

    async def test_picks_largest_qualifying_n_from_three(self):
        """
        Given three measurements where n=100 and n=200 qualify but n=300
        does not, returns 200 (the LARGEST qualifying n).
        """
        from krakey.engines.memory._internal.bench import recommend_soft_limit
        measurements = [
            {
                "n": 100,
                "insert_per_node_ms": 0.5,
                "vec_search_ms_p50": 5.0,
                "vec_search_ms_p95": 80.0,     # qualifies
                "fts_search_ms_p50": 3.0,
            },
            {
                "n": 200,
                "insert_per_node_ms": 1.0,
                "vec_search_ms_p50": 15.0,
                "vec_search_ms_p95": 190.0,    # qualifies (under 200)
                "fts_search_ms_p50": 6.0,
            },
            {
                "n": 300,
                "insert_per_node_ms": 2.0,
                "vec_search_ms_p50": 30.0,
                "vec_search_ms_p95": 250.0,    # does NOT qualify
                "fts_search_ms_p50": 10.0,
            },
        ]
        result = recommend_soft_limit(measurements, target_p95_ms=200.0)
        assert result == 200, (
            f"Expected 200 (largest qualifying n), got {result}"
        )

    async def test_picks_largest_not_first_qualifying(self):
        """
        Ensures the function doesn't short-circuit on the first qualifying
        entry — returns the LARGEST, regardless of input ordering.
        """
        from krakey.engines.memory._internal.bench import recommend_soft_limit
        # Present in reverse order (smallest n last)
        measurements = [
            {
                "n": 500,
                "insert_per_node_ms": 3.0,
                "vec_search_ms_p50": 50.0,
                "vec_search_ms_p95": 450.0,    # over target
                "fts_search_ms_p50": 20.0,
            },
            {
                "n": 300,
                "insert_per_node_ms": 2.0,
                "vec_search_ms_p50": 30.0,
                "vec_search_ms_p95": 150.0,    # qualifies
                "fts_search_ms_p50": 12.0,
            },
            {
                "n": 100,
                "insert_per_node_ms": 0.5,
                "vec_search_ms_p50": 8.0,
                "vec_search_ms_p95": 60.0,     # qualifies (but n is smaller)
                "fts_search_ms_p50": 3.0,
            },
        ]
        result = recommend_soft_limit(measurements, target_p95_ms=200.0)
        assert result == 300, (
            f"Expected 300 (largest qualifying n), got {result}"
        )

    # --- BVA: boundary ---

    async def test_exact_target_boundary_qualifies(self):
        """
        A measurement whose p95 == target_p95_ms exactly must qualify
        (contract: <= target).
        """
        from krakey.engines.memory._internal.bench import recommend_soft_limit
        measurements = [
            {
                "n": 250,
                "insert_per_node_ms": 1.0,
                "vec_search_ms_p50": 15.0,
                "vec_search_ms_p95": 200.0,    # exactly at boundary
                "fts_search_ms_p50": 7.0,
            }
        ]
        result = recommend_soft_limit(measurements, target_p95_ms=200.0)
        assert result == 250

    async def test_one_ms_over_target_does_not_qualify(self):
        """
        A measurement whose p95 == target + epsilon must NOT qualify.
        """
        from krakey.engines.memory._internal.bench import recommend_soft_limit
        measurements = [
            {
                "n": 250,
                "insert_per_node_ms": 1.0,
                "vec_search_ms_p50": 15.0,
                "vec_search_ms_p95": 200.001,  # just over boundary
                "fts_search_ms_p50": 7.0,
            }
        ]
        result = recommend_soft_limit(measurements, target_p95_ms=200.0)
        assert result is None

    # --- negative / edge ---

    async def test_empty_list_returns_none(self):
        """recommend_soft_limit([]) must return None."""
        from krakey.engines.memory._internal.bench import recommend_soft_limit
        assert recommend_soft_limit([]) is None

    async def test_empty_list_with_custom_target_returns_none(self):
        """Empty list returns None regardless of target_p95_ms."""
        from krakey.engines.memory._internal.bench import recommend_soft_limit
        assert recommend_soft_limit([], target_p95_ms=50.0) is None

    async def test_single_measurement_over_target_returns_none(self):
        """Single measurement over target → None."""
        from krakey.engines.memory._internal.bench import recommend_soft_limit
        measurements = [
            {
                "n": 500,
                "insert_per_node_ms": 2.0,
                "vec_search_ms_p50": 80.0,
                "vec_search_ms_p95": 201.0,    # over target
                "fts_search_ms_p50": 30.0,
            }
        ]
        assert recommend_soft_limit(measurements, target_p95_ms=200.0) is None

    async def test_zero_target_nothing_qualifies_unless_zero_p95(self):
        """
        With target_p95_ms=0.0, only a measurement with p95==0 would qualify.
        A measurement with p95=1.0 must not qualify.
        """
        from krakey.engines.memory._internal.bench import recommend_soft_limit
        measurements = [
            {
                "n": 10,
                "insert_per_node_ms": 0.1,
                "vec_search_ms_p50": 0.5,
                "vec_search_ms_p95": 1.0,
                "fts_search_ms_p50": 0.2,
            }
        ]
        result = recommend_soft_limit(measurements, target_p95_ms=0.0)
        assert result is None


# ===========================================================================
# 2. Config consumption — GraphMemoryEngine.__init__ config= kwarg
# ===========================================================================


class TestEngineConfigKwarg:
    """
    ``GraphMemoryEngine(config={...})`` — P2 addition.

    Contract: engine reads ``gm_node_soft_limit`` from config dict and
    stores it as ``engine.gm_node_soft_limit`` (int). Default 1000.
    """

    # --- positive ---

    async def test_explicit_value_stored_as_attribute(self, tmp_path):
        """config={'gm_node_soft_limit': 777} → engine.gm_node_soft_limit == 777."""
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=_FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            extractor_llm=_FakeExtractorChat(),
            config={"gm_node_soft_limit": 777},
        )
        assert engine.gm_node_soft_limit == 777

    async def test_attribute_is_public_int(self, tmp_path):
        """engine.gm_node_soft_limit must be an int (not a string or float)."""
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=_FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            extractor_llm=_FakeExtractorChat(),
            config={"gm_node_soft_limit": 500},
        )
        assert isinstance(engine.gm_node_soft_limit, int)

    async def test_attribute_accessible_before_initialize(self, tmp_path):
        """gm_node_soft_limit is set at construction, before initialize()."""
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=_FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            extractor_llm=_FakeExtractorChat(),
            config={"gm_node_soft_limit": 400},
        )
        # Do NOT call initialize() — must still have the attribute
        assert engine.gm_node_soft_limit == 400

    # --- BVA ---

    async def test_zero_is_allowed(self, tmp_path):
        """BVA: gm_node_soft_limit=0 is a valid value (0 = disabled)."""
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=_FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            extractor_llm=_FakeExtractorChat(),
            config={"gm_node_soft_limit": 0},
        )
        assert engine.gm_node_soft_limit == 0

    async def test_value_one_stored_correctly(self, tmp_path):
        """BVA: gm_node_soft_limit=1 (min positive) is stored as 1."""
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=_FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            extractor_llm=_FakeExtractorChat(),
            config={"gm_node_soft_limit": 1},
        )
        assert engine.gm_node_soft_limit == 1

    async def test_large_value_stored_correctly(self, tmp_path):
        """BVA: large value (100000) is stored without truncation."""
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=_FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            extractor_llm=_FakeExtractorChat(),
            config={"gm_node_soft_limit": 100_000},
        )
        assert engine.gm_node_soft_limit == 100_000

    # --- default value (config=None / missing key) ---

    async def test_default_when_config_is_none(self, tmp_path):
        """
        config=None → gm_node_soft_limit defaults to 1000.

        Contract: ``int((config or {}).get('gm_node_soft_limit', 1000))``
        """
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=_FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            extractor_llm=_FakeExtractorChat(),
            config=None,
        )
        assert engine.gm_node_soft_limit == 1000

    async def test_default_when_config_omitted(self, tmp_path):
        """
        config kwarg omitted entirely → gm_node_soft_limit defaults to 1000.
        """
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=_FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            extractor_llm=_FakeExtractorChat(),
        )
        assert engine.gm_node_soft_limit == 1000

    async def test_default_when_key_missing_from_config(self, tmp_path):
        """
        config={} (empty dict, key absent) → gm_node_soft_limit defaults to 1000.
        """
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=_FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            extractor_llm=_FakeExtractorChat(),
            config={},
        )
        assert engine.gm_node_soft_limit == 1000

    # --- type coercion ---

    async def test_string_value_coerced_to_int(self, tmp_path):
        """
        config={'gm_node_soft_limit': '350'} → int coercion → 350.

        Contract: ``int((config or {}).get('gm_node_soft_limit', 1000))``
        meaning any value the YAML parser may produce as a string is
        converted.
        """
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=_FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            extractor_llm=_FakeExtractorChat(),
            config={"gm_node_soft_limit": "350"},
        )
        assert engine.gm_node_soft_limit == 350
        assert isinstance(engine.gm_node_soft_limit, int)

    async def test_float_value_truncated_to_int(self, tmp_path):
        """
        config={'gm_node_soft_limit': 999.9} → int(999.9) == 999 (truncation).
        """
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=_FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            extractor_llm=_FakeExtractorChat(),
            config={"gm_node_soft_limit": 999.9},
        )
        assert engine.gm_node_soft_limit == 999

    # --- extra keys in config are ignored ---

    async def test_extra_config_keys_do_not_break_construction(self, tmp_path):
        """Config dict with unknown keys must not raise at construction time."""
        engine = GraphMemoryEngine(
            db_path=":memory:",
            embedder=_FakeEmbedder(),
            kb_dir=str(tmp_path / "kbs"),
            extractor_llm=_FakeExtractorChat(),
            config={
                "gm_node_soft_limit": 300,
                "unknown_future_key": "some_value",
                "another_unknown": 42,
            },
        )
        assert engine.gm_node_soft_limit == 300


# ===========================================================================
# 3. POST /api/benchmark
# ===========================================================================


class TestPostApiBenchmark:
    """
    ``POST /api/benchmark`` — runs a GM-latency benchmark and returns
    ``{results: [...], recommended_soft_limit: int|None}``.

    Tests keep sizes TINY (e.g. [1, 2]) so they finish quickly.
    """

    # --- positive ---

    async def test_returns_200_with_small_sizes(self, tmp_path, monkeypatch):
        """POST {sizes: [1, 2]} returns HTTP 200."""
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/benchmark", json={"sizes": [1, 2]})
        assert r.status_code == 200

    async def test_response_body_has_results_and_recommended_soft_limit(
        self, tmp_path, monkeypatch
    ):
        """Response body must have 'results' and 'recommended_soft_limit' keys."""
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/benchmark", json={"sizes": [1, 2]})
        body = r.json()
        assert "results" in body, f"Missing 'results' key in {body}"
        assert "recommended_soft_limit" in body, (
            f"Missing 'recommended_soft_limit' key in {body}"
        )

    async def test_results_length_matches_sizes(self, tmp_path, monkeypatch):
        """results list length must equal len(sizes)."""
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/benchmark", json={"sizes": [1, 2]})
        body = r.json()
        assert len(body["results"]) == 2

    async def test_each_result_has_all_five_measure_at_keys(
        self, tmp_path, monkeypatch
    ):
        """Each entry in results must have all 5 keys from measure_at."""
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/benchmark", json={"sizes": [1, 2]})
        required_keys = {
            "n",
            "insert_per_node_ms",
            "vec_search_ms_p50",
            "vec_search_ms_p95",
            "fts_search_ms_p50",
        }
        for entry in r.json()["results"]:
            missing = required_keys - set(entry.keys())
            assert not missing, f"Result entry missing keys: {missing}; got {entry}"

    async def test_recommended_soft_limit_is_int_or_null(
        self, tmp_path, monkeypatch
    ):
        """recommended_soft_limit must be an int or null (None), never a string."""
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/benchmark", json={"sizes": [1, 2]})
        rsl = r.json()["recommended_soft_limit"]
        assert rsl is None or isinstance(rsl, int), (
            f"Expected int or null, got {type(rsl)}: {rsl}"
        )

    async def test_benchmark_does_not_500(self, tmp_path, monkeypatch):
        """Benchmark with sizes=[1] must never return 500."""
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/benchmark", json={"sizes": [1]})
        assert r.status_code != 500

    async def test_single_size_returns_one_result(self, tmp_path, monkeypatch):
        """sizes=[1] → results has exactly 1 entry."""
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/benchmark", json={"sizes": [1]})
        assert len(r.json()["results"]) == 1

    async def test_optional_dim_param_accepted(self, tmp_path, monkeypatch):
        """Optional 'dim' parameter is accepted without error."""
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post(
                "/api/benchmark", json={"sizes": [1], "dim": 16}
            )
        assert r.status_code == 200

    async def test_optional_repeats_param_accepted(self, tmp_path, monkeypatch):
        """Optional 'repeats' parameter is accepted without error."""
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post(
                "/api/benchmark", json={"sizes": [1, 2], "repeats": 2}
            )
        assert r.status_code == 200

    # --- BVA: sizes boundary ---

    async def test_sizes_with_zero_included_is_handled(
        self, tmp_path, monkeypatch
    ):
        """
        sizes=[0, 1] — zero is a valid boundary value; the bench handles
        n=0 (returns zeros). Must not 500.
        """
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/benchmark", json={"sizes": [0, 1]})
        assert r.status_code != 500
        body = r.json()
        assert len(body["results"]) == 2

    # --- accepting empty body (default sizes) ---

    async def test_empty_json_body_accepted_returns_200(
        self, tmp_path, monkeypatch
    ):
        """
        POST {} (all optional fields absent) must return 200, not 422.

        NOTE: Default sizes may be large and slow. We only assert status 200
        and response shape — do NOT assert timing. If default sizes are very
        large this test may be slow; see the fixture note in the class
        docstring. If the default sizes cause a test timeout in CI, this
        test may be marked xfail-slow in future.
        """
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/benchmark", json={})
        assert r.status_code == 200
        body = r.json()
        assert "results" in body
        assert "recommended_soft_limit" in body

    # --- negative ---

    async def test_non_list_sizes_returns_400_or_422(self, tmp_path, monkeypatch):
        """sizes must be a list; a scalar is rejected with 400 or 422."""
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path)
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.post("/api/benchmark", json={"sizes": 42})
        assert r.status_code in (400, 422)

    async def test_benchmark_uses_throwaway_gm_not_live_engine(
        self, tmp_path, monkeypatch
    ):
        """
        After a benchmark call the live engine's node count is unchanged.
        (The bench must use its own in-memory throwaway GM.)
        """
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path)
        app = create_memory_app(engine)
        count_before = await engine.count_nodes()
        async with _client(app) as c:
            await c.post("/api/benchmark", json={"sizes": [1, 2]})
        count_after = await engine.count_nodes()
        assert count_after == count_before, (
            "Benchmark must not insert into the live engine's GM"
        )


# ===========================================================================
# 4. GET /api/settings + PUT /api/settings
# ===========================================================================


class TestGetApiSettings:
    """
    ``GET /api/settings`` — returns the engine's own settings dict.

    Contract: returns {} when no settings have been written (engine
    constructed with config={}), or the previously written dict.
    """

    # --- positive ---

    async def test_returns_200(self, tmp_path, monkeypatch):
        """GET /api/settings returns HTTP 200."""
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path, config={})
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/settings")
        assert r.status_code == 200

    async def test_returns_dict_body(self, tmp_path, monkeypatch):
        """GET /api/settings body is a JSON object (dict)."""
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path, config={})
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/settings")
        assert isinstance(r.json(), dict)

    async def test_fresh_engine_returns_empty_or_dict(self, tmp_path, monkeypatch):
        """
        A fresh engine with config={} and no settings file returns {} or a
        dict (not null, not a list, not an error).
        """
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path, config={})
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/settings")
        body = r.json()
        assert isinstance(body, dict)


class TestPutApiSettings:
    """
    ``PUT /api/settings`` — full-replace write of the engine settings file.

    Contract: returns 200 + {ok: true}; subsequent GET reflects the written
    values.
    """

    # --- positive ---

    async def test_put_returns_200(self, tmp_path, monkeypatch):
        """PUT /api/settings returns HTTP 200."""
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path, config={})
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.put(
                "/api/settings",
                json={"gm_node_soft_limit": 500},
            )
        assert r.status_code == 200

    async def test_put_response_has_ok_true(self, tmp_path, monkeypatch):
        """PUT response body must contain {ok: true}."""
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path, config={})
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.put(
                "/api/settings",
                json={"gm_node_soft_limit": 500},
            )
        body = r.json()
        assert body.get("ok") is True

    async def test_put_then_get_reflects_written_values(self, tmp_path, monkeypatch):
        """
        State transition: PUT {gm_node_soft_limit: 500, note: 'x'} then
        GET /api/settings must return the same dict.
        """
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path, config={})
        app = create_memory_app(engine)
        payload = {"gm_node_soft_limit": 500, "note": "x"}
        async with _client(app) as c:
            await c.put("/api/settings", json=payload)
            r = await c.get("/api/settings")
        body = r.json()
        assert body.get("gm_node_soft_limit") == 500
        assert body.get("note") == "x"

    async def test_second_put_fully_replaces_first(self, tmp_path, monkeypatch):
        """
        State transition: PUT dict A then PUT dict B — GET must return dict
        B (full replace, NOT merge). Keys present only in A must not appear.
        """
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path, config={})
        app = create_memory_app(engine)
        dict_a = {"gm_node_soft_limit": 300, "stale_key": "old_value"}
        dict_b = {"gm_node_soft_limit": 800}
        async with _client(app) as c:
            await c.put("/api/settings", json=dict_a)
            # Confirm stale_key is present after first PUT
            after_a = (await c.get("/api/settings")).json()
            assert "stale_key" in after_a, (
                "stale_key should be present after first PUT (pre-condition)"
            )
            # Second PUT with a different dict
            await c.put("/api/settings", json=dict_b)
            after_b = (await c.get("/api/settings")).json()
        # Full replace: stale_key from dict_a must be gone
        assert "stale_key" not in after_b, (
            f"stale_key must be absent after full-replace PUT; got {after_b}"
        )
        # New value for gm_node_soft_limit reflects dict_b
        assert after_b.get("gm_node_soft_limit") == 800

    async def test_put_empty_dict_clears_settings(self, tmp_path, monkeypatch):
        """
        PUT {} is a valid full replace that writes an empty settings dict.
        A subsequent GET returns {} (all previous keys gone).
        """
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path, config={})
        app = create_memory_app(engine)
        async with _client(app) as c:
            # Write something first
            await c.put("/api/settings", json={"gm_node_soft_limit": 600})
            # Wipe it
            r = await c.put("/api/settings", json={})
            assert r.status_code == 200
            body = (await c.get("/api/settings")).json()
        # Either {} or an engine-default blob; at minimum gm_node_soft_limit
        # from the previous PUT must not be 600 any more (replaced/cleared).
        assert body.get("gm_node_soft_limit") != 600

    async def test_put_with_multiple_keys_all_reflected(
        self, tmp_path, monkeypatch
    ):
        """All keys in PUT payload appear in the subsequent GET response."""
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path, config={})
        app = create_memory_app(engine)
        payload = {
            "gm_node_soft_limit": 750,
            "some_other_tunable": "high",
            "numeric_flag": 3,
        }
        async with _client(app) as c:
            await c.put("/api/settings", json=payload)
            body = (await c.get("/api/settings")).json()
        for key, value in payload.items():
            assert body.get(key) == value, (
                f"Key '{key}': expected {value!r}, got {body.get(key)!r}"
            )

    # --- negative ---

    async def test_put_non_dict_body_returns_400_or_422(
        self, tmp_path, monkeypatch
    ):
        """
        PUT a non-dict body (e.g. a JSON array or scalar) must return 400
        or 422 — not 200 and not 500.
        """
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path, config={})
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.put("/api/settings", content=b"[1, 2, 3]",
                            headers={"Content-Type": "application/json"})
        assert r.status_code in (400, 422), (
            f"Expected 400 or 422 for non-dict body, got {r.status_code}"
        )

    async def test_put_string_body_returns_400_or_422(
        self, tmp_path, monkeypatch
    ):
        """PUT a plain JSON string (not an object) must return 400 or 422."""
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path, config={})
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.put("/api/settings", content=b'"just a string"',
                            headers={"Content-Type": "application/json"})
        assert r.status_code in (400, 422)

    async def test_put_does_not_500(self, tmp_path, monkeypatch):
        """PUT with a valid dict body must never return 500."""
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path, config={})
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.put("/api/settings", json={"gm_node_soft_limit": 100})
        assert r.status_code != 500

    async def test_get_does_not_500(self, tmp_path, monkeypatch):
        """GET /api/settings must never return 500."""
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path, config={})
        app = create_memory_app(engine)
        async with _client(app) as c:
            r = await c.get("/api/settings")
        assert r.status_code != 500


# ===========================================================================
# 5. State transition: full settings round-trip (PUT → GET → PUT → GET)
# ===========================================================================


class TestSettingsRoundTrip:
    """
    Multi-step state transitions across GET/PUT /api/settings in a single
    ASGI client session.
    """

    async def test_initial_get_then_put_then_get(self, tmp_path, monkeypatch):
        """
        Full round-trip:
          1. GET → initial state (empty or dict)
          2. PUT {gm_node_soft_limit: 500}
          3. GET → reflects 500
          4. PUT {gm_node_soft_limit: 999}
          5. GET → reflects 999 (not 500)
        """
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path, config={})
        app = create_memory_app(engine)

        async with _client(app) as c:
            # Step 1: initial GET
            r = await c.get("/api/settings")
            assert r.status_code == 200

            # Step 2: PUT first value
            r = await c.put("/api/settings", json={"gm_node_soft_limit": 500})
            assert r.status_code == 200

            # Step 3: GET reflects 500
            body = (await c.get("/api/settings")).json()
            assert body.get("gm_node_soft_limit") == 500, (
                f"Expected 500 after first PUT, got {body}"
            )

            # Step 4: PUT second value
            r = await c.put("/api/settings", json={"gm_node_soft_limit": 999})
            assert r.status_code == 200

            # Step 5: GET reflects 999
            body = (await c.get("/api/settings")).json()
            assert body.get("gm_node_soft_limit") == 999, (
                f"Expected 999 after second PUT, got {body}"
            )
            # And definitely not 500 any more
            assert body.get("gm_node_soft_limit") != 500

    async def test_settings_persist_across_new_client_instances(
        self, tmp_path, monkeypatch
    ):
        """
        After PUT with one client session, a freshly created client on the
        same app sees the written values in GET.

        This verifies the write hits durable storage (the settings file via
        FileEngineConfigStore), not just in-memory state.
        """
        monkeypatch.chdir(tmp_path)
        engine = await _make_engine_with_config(tmp_path, config={})
        app = create_memory_app(engine)

        # Write settings
        async with _client(app) as c1:
            r = await c1.put("/api/settings", json={"gm_node_soft_limit": 777})
            assert r.status_code == 200

        # New client session — reads from same app / same engine
        async with _client(app) as c2:
            body = (await c2.get("/api/settings")).json()
        assert body.get("gm_node_soft_limit") == 777, (
            f"Settings not persisted: expected 777, got {body}"
        )
