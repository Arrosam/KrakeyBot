"""Phase 3 / G: GM performance benchmark.

Measures per-operation latency at various GM sizes and recommends
`gm_node_soft_limit` based on a target p95 vector-search latency.

Used by `scripts/bench_gm.py`. Pure-function `measure_at` + `recommend_soft_limit`
are extracted here so they can be unit-tested without invoking the script.
"""
from __future__ import annotations

import asyncio
import random
import statistics
import time
from typing import Any

from krakey.memory.graph_memory import GraphMemory


class _DeterministicEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    async def __call__(self, text: str) -> list[float]:
        rng = random.Random(hash(text) & 0xFFFFFFFF)
        return [rng.gauss(0.0, 1.0) for _ in range(self.dim)]


async def measure_at(n_nodes: int, *, dim: int = 384,
                       query_repeats: int = 10,
                       db_path: str = ":memory:") -> dict[str, float]:
    """Bench a single GM size. Returns latency stats in ms."""
    embed = _DeterministicEmbedder(dim)
    gm = GraphMemory(db_path, embedder=embed)
    await gm.initialize()

    if n_nodes == 0:
        await gm.close()
        return {"n": 0, "insert_per_node_ms": 0.0,
                "vec_search_ms_p50": 0.0, "vec_search_ms_p95": 0.0,
                "fts_search_ms_p50": 0.0}

    # ------ Insert ------
    t0 = time.perf_counter()
    for i in range(n_nodes):
        await gm.insert_node(
            name=f"n{i}", category="FACT", description=f"node {i}",
            embedding=await embed(f"n{i}"),
        )
    insert_secs = time.perf_counter() - t0

    # ------ Vec search ------
    vec_times: list[float] = []
    for i in range(query_repeats):
        q = await embed(f"q{i}")
        t0 = time.perf_counter()
        await gm.vec_search(q, top_k=10)
        vec_times.append(time.perf_counter() - t0)

    # ------ FTS search ------
    fts_times: list[float] = []
    for i in range(query_repeats):
        t0 = time.perf_counter()
        await gm.fts_search(f"node {i % max(1, n_nodes)}", top_k=10)
        fts_times.append(time.perf_counter() - t0)

    await gm.close()
    return {
        "n": n_nodes,
        "insert_per_node_ms": (insert_secs / n_nodes) * 1000,
        "vec_search_ms_p50": statistics.median(vec_times) * 1000,
        "vec_search_ms_p95": _p95(vec_times) * 1000,
        "fts_search_ms_p50": statistics.median(fts_times) * 1000,
    }


def recommend_soft_limit(measurements: list[dict[str, Any]], *,
                            target_p95_ms: float = 200.0) -> int | None:
    """Pick the largest N whose vec_search p95 stays under the target.
    Returns None if no measured N satisfies the target."""
    if not measurements:
        return None
    sorted_ms = sorted(measurements, key=lambda r: r["n"])
    best: int | None = None
    for r in sorted_ms:
        if r.get("vec_search_ms_p95", float("inf")) <= target_p95_ms:
            best = int(r["n"])
        else:
            break  # results are monotone-ish; stop at first overshoot
    return best


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, int(len(s) * 0.95) - 1)
    return s[min(idx, len(s) - 1)]
