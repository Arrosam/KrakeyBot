"""GM-latency benchmark — engine-internal (replaces deleted krakey/tools/perf_bench.py).

Pure, importable module; ``measure_at`` is async but has no side-effects on
the live engine — it creates a THROWAWAY in-memory ``GraphMemory`` instance.
``recommend_soft_limit`` is synchronous.
"""
from __future__ import annotations

import hashlib
import random
import statistics
import time
from typing import Any


def _p95(values: list[float]) -> float:
    """Return the 95th-percentile of *values* (ms). Returns 0.0 for empty."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, int(len(sorted_vals) * 0.95) - 1)
    return sorted_vals[idx]


def _dummy_embedding(text: str, dim: int) -> list[float]:
    """Deterministic, seedable unit-length embedding for benchmarking."""
    seed = int(hashlib.md5(text.encode(), usedforsecurity=False).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(seed)
    vec = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    magnitude = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / magnitude for x in vec]


async def measure_at(
    n_nodes: int,
    *,
    dim: int = 384,
    query_repeats: int = 10,
    db_path: str = ":memory:",
) -> dict[str, Any]:
    """Bench a single GM size on a THROWAWAY in-memory GraphMemory.

    Returns latency stats (ms): keys ``n``, ``insert_per_node_ms``,
    ``vec_search_ms_p50``, ``vec_search_ms_p95``, ``fts_search_ms_p50``.
    ``n_nodes == 0`` returns all-zero stats without touching the DB.
    """
    _zero: dict[str, Any] = {
        "n": 0,
        "insert_per_node_ms": 0.0,
        "vec_search_ms_p50": 0.0,
        "vec_search_ms_p95": 0.0,
        "fts_search_ms_p50": 0.0,
    }

    if n_nodes == 0:
        return _zero

    from krakey.engines.memory._internal.graph_memory import GraphMemory

    class _DummyEmbedder:
        """Sync-wrapped deterministic embedder satisfying the AsyncEmbedder protocol."""
        def __init__(self, dim: int) -> None:
            self._dim = dim

        async def embed(self, text: str) -> list[float]:
            return _dummy_embedding(text, self._dim)

    embedder = _DummyEmbedder(dim)
    gm = GraphMemory(db_path, embedder=embedder)
    await gm.initialize()

    try:
        # ---- insert phase ------------------------------------------------
        t_insert_start = time.perf_counter()
        for i in range(n_nodes):
            text = f"bench_node_{i}"
            embedding = _dummy_embedding(text, dim)
            await gm.insert_node(
                name=text,
                category="FACT",
                description=f"Benchmark node {i}",
                embedding=embedding,
            )
        t_insert_end = time.perf_counter()
        insert_total_ms = (t_insert_end - t_insert_start) * 1_000
        insert_per_node_ms = insert_total_ms / n_nodes if n_nodes else 0.0

        # ---- vec_search timings ------------------------------------------
        query_vec = _dummy_embedding("benchmark query", dim)
        vec_times: list[float] = []
        for _ in range(query_repeats):
            t0 = time.perf_counter()
            await gm.vec_search(query_vec, top_k=5)
            vec_times.append((time.perf_counter() - t0) * 1_000)

        # ---- fts_search timings ------------------------------------------
        fts_times: list[float] = []
        for _ in range(query_repeats):
            t0 = time.perf_counter()
            await gm.fts_search("bench", top_k=5)
            fts_times.append((time.perf_counter() - t0) * 1_000)

        return {
            "n": n_nodes,
            "insert_per_node_ms": round(insert_per_node_ms, 3),
            "vec_search_ms_p50": round(statistics.median(vec_times), 3),
            "vec_search_ms_p95": round(_p95(vec_times), 3),
            "fts_search_ms_p50": round(statistics.median(fts_times), 3),
        }

    finally:
        await gm.close()


def recommend_soft_limit(
    measurements: list[dict],
    *,
    target_p95_ms: float = 200.0,
) -> int | None:
    """Return the largest N whose ``vec_search_ms_p95 <= target_p95_ms``.

    Returns ``None`` if *measurements* is empty or no measurement qualifies.
    """
    if not measurements:
        return None

    sorted_m = sorted(measurements, key=lambda m: m["n"])
    best: int | None = None
    for m in sorted_m:
        if m["vec_search_ms_p95"] <= target_p95_ms:
            best = m["n"]
        else:
            # Monotone assumption: first overshoot → stop scanning
            break
    return best
