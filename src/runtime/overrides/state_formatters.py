"""Render runtime state as human-readable strings.

These are the bodies of the ``/status`` and ``/memory_stats`` CLI
override commands, but the formatting itself is independent of the
override-dispatch system — it queries Runtime + GM + KB registry and
returns a one-line summary suitable for printing.

Kept under ``overrides/`` because the only callers today are the
override commands; if a dashboard endpoint or another CLI starts
needing the same output, this module is the natural import target.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.runtime.runtime import Runtime


async def format_status(runtime: "Runtime") -> str:
    nodes = await runtime.gm.count_nodes()
    edges = await runtime.gm.count_edges()
    pct = int(nodes / runtime.config.fatigue.gm_node_soft_limit * 100) \
        if runtime.config.fatigue.gm_node_soft_limit else 0
    # `_sleep_cycles` is a runtime-lifetime counter (per-process, not
    # persisted) since the 2026-04-25 self-model slim — see
    # docs/design/reflects-and-self-model.md Part 1.
    cycles = getattr(runtime, "_sleep_cycles", 0)
    name = runtime.self_model.get("identity", {}).get("name", "(unnamed)")
    return (
        f"name={name} "
        f"heartbeats={runtime.heartbeat_count} "
        f"gm_nodes={nodes} gm_edges={edges} fatigue={pct}% "
        f"sleep_cycles={cycles} "
        f"bootstrap_complete={not runtime.is_bootstrap}"
    )


async def format_memory_stats(runtime: "Runtime") -> str:
    nodes = await runtime.gm.count_nodes()
    edges = await runtime.gm.count_edges()
    db = runtime.gm._require()  # noqa: SLF001
    async with db.execute(
        "SELECT category, COUNT(*) FROM gm_nodes GROUP BY category"
    ) as cur:
        cat_rows = await cur.fetchall()
    async with db.execute(
        "SELECT source_type, COUNT(*) FROM gm_nodes GROUP BY source_type"
    ) as cur:
        src_rows = await cur.fetchall()
    kbs = await runtime.kb_registry.list_kbs()

    by_cat = ", ".join(f"{r[0]}={r[1]}" for r in cat_rows) or "(none)"
    by_src = ", ".join(f"{r[0]}={r[1]}" for r in src_rows) or "(none)"
    kb_line = f"{len(kbs)} KB(s)" + (
        ": " + ", ".join(f"{k['kb_id']}({k['entry_count']})" for k in kbs)
        if kbs else ""
    )
    return (f"gm: {nodes} nodes, {edges} edges  |  by_cat: {by_cat}  |  "
            f"by_src: {by_src}  |  {kb_line}")
