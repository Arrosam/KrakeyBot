"""Phase 2.3d: 7-phase Sleep orchestrator (DevSpec §11.3).

Orchestration only — each phase lives in its own module. Sleep is a
significant state mutation: GM nodes are deleted, KBs are created/grown,
the index graph is rebuilt. Run only when the runtime decides to (Self
DECISION 'enter sleep mode' OR fatigue ≥ force_sleep_threshold).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from src.interfaces.sensory import SensoryRegistry
from src.memory.graph_memory import GraphMemory
from src.memory.knowledge_base import KBRegistry
from src.sleep.clustering import run_leiden_clustering
from src.sleep.index_rebuild import rebuild_index_graph
from src.sleep.migration import migrate_gm_to_kb


class AsyncChatLLM(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...


class AsyncEmbedder(Protocol):
    async def __call__(self, text: str) -> list[float]: ...


async def enter_sleep_mode(
    gm: GraphMemory, reg: KBRegistry, sensories: SensoryRegistry,
    *, llm: AsyncChatLLM, embedder: AsyncEmbedder,
    log_dir: str | Path = "workspace/logs",
) -> dict[str, Any]:
    """Run all 7 phases. Returns summary stats."""

    started_at = datetime.now()

    # Phase 1: pause non-urgent sensories
    await sensories.pause_non_urgent()

    try:
        # Phase 2: cluster + summarize
        communities = await run_leiden_clustering(
            gm, llm=llm, embedder=embedder,
        )

        # Phase 3 (+ implicit Phase 4): migrate FACT/RELATION/KNOWLEDGE
        # (incl. completed-TARGETs-now-FACT) into KB.
        before_targets = await _count(gm, "TARGET")
        migrate_stats = await migrate_gm_to_kb(gm, reg)

        # Phase 4: explicit no-op (completed targets already migrated; open
        # targets stay in GM untouched).
        targets_preserved = await _count(gm, "TARGET")

        # Phase 5: clear FOCUS
        focus_cleared = await _delete_category(gm, "FOCUS")

        # Phase 6: rebuild Index Graph
        idx_stats = await rebuild_index_graph(
            gm, reg, llm=llm, embedder=embedder,
        )

        # Phase 7: daily log
        result: dict[str, Any] = {
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now().isoformat(),
            "communities": len(communities),
            "facts_migrated": migrate_stats["migrated_nodes"],
            "edges_migrated": migrate_stats["migrated_edges"],
            "kbs_created": migrate_stats["kbs_created"],
            "skipped_no_community": migrate_stats["skipped_no_community"],
            "focus_cleared": focus_cleared,
            "targets_preserved": targets_preserved,
            "targets_before": before_targets,
            "index_nodes": idx_stats["index_nodes"],
            "index_edges_added": idx_stats["edges_added"],
        }
        await _write_daily_log(log_dir, result)
        return result
    finally:
        await sensories.resume_all(_buffer_for_resume(sensories))


# ---------------- helpers ----------------


async def _count(gm: GraphMemory, category: str) -> int:
    db = gm._require()  # noqa: SLF001
    async with db.execute(
        "SELECT COUNT(*) FROM gm_nodes WHERE category = ?", (category,),
    ) as cur:
        row = await cur.fetchone()
        return int(row[0])


async def _delete_category(gm: GraphMemory, category: str) -> int:
    db = gm._require()  # noqa: SLF001
    async with db.execute(
        "SELECT COUNT(*) FROM gm_nodes WHERE category = ?", (category,),
    ) as cur:
        row = await cur.fetchone()
        n = int(row[0])
    await db.execute("DELETE FROM gm_nodes WHERE category = ?", (category,))
    await db.commit()
    return n


def _buffer_for_resume(sensories: SensoryRegistry):
    """SensoryRegistry.resume_all needs a buffer arg. We pass through the
    same buffer used at startup if any sensory is currently running, else
    a fresh one (resume_all is a no-op when nothing was paused)."""
    # Find any currently-running sensory's stored buffer; else None.
    for s in sensories._sensories.values():  # noqa: SLF001
        buf = getattr(s, "_buffer", None)
        if buf is not None:
            return buf
    # Fallback: an empty buffer; resume_all loops only over previously
    # paused, so this only matters when there were paused sensories
    # but no active ones to borrow a buffer from.
    from src.runtime.stimulus_buffer import StimulusBuffer
    return StimulusBuffer()


async def _write_daily_log(log_dir: str | Path, record: dict[str, Any]) -> None:
    p = Path(log_dir)
    p.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y-%m-%d")
    log_file = p / f"sleep-{day}.jsonl"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
