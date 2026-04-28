"""Run a GM perf benchmark and print a recommended `gm_node_soft_limit`.

Usage:
    python scripts/bench_gm.py
    python scripts/bench_gm.py --sizes 100 200 500 1000 --target-ms 150
"""
import argparse
import asyncio
import io
import sys
from pathlib import Path

# Make `python scripts/bench_gm.py` work from any cwd by adding the repo
# root to sys.path before importing src.*
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from krakey.tools.perf_bench import measure_at, recommend_soft_limit  # noqa: E402


# Force UTF-8 stdout for nice rendering on Windows GBK consoles.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                errors="replace")


async def main(args):
    print(f"benchmarking GM at sizes={args.sizes}, dim={args.dim}, "
          f"repeats={args.repeats}\n")
    print(f"{'N':>6}  {'insert(ms/n)':>14}  {'vec p50(ms)':>13}  "
          f"{'vec p95(ms)':>13}  {'fts p50(ms)':>13}")
    results = []
    for n in args.sizes:
        r = await measure_at(n, dim=args.dim, query_repeats=args.repeats)
        results.append(r)
        print(f"{r['n']:>6}  {r['insert_per_node_ms']:>14.2f}  "
              f"{r['vec_search_ms_p50']:>13.2f}  "
              f"{r['vec_search_ms_p95']:>13.2f}  "
              f"{r['fts_search_ms_p50']:>13.2f}")

    rec = recommend_soft_limit(results, target_p95_ms=args.target_ms)
    print()
    if rec is None:
        print(f"⚠ Even smallest size exceeds target p95 = {args.target_ms}ms.")
        print(f"  Either set a much smaller soft_limit (< {args.sizes[0]}) "
              "or relax the target.")
    else:
        print(f"→ Recommended  gm_node_soft_limit = {rec}  "
              f"(keeps vec_search p95 ≤ {args.target_ms}ms)")
        print(f"  Edit config.yaml under fatigue.gm_node_soft_limit.")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[50, 100, 200, 500, 1000, 2000])
    p.add_argument("--dim", type=int, default=384,
                   help="embedding dimension (default 384, like bge-m3 base)")
    p.add_argument("--repeats", type=int, default=10,
                   help="query repeats per size for percentile stability")
    p.add_argument("--target-ms", type=float, default=200.0,
                   help="target vec_search p95 latency")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(_parse()))
