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
from typing import TYPE_CHECKING, Any, Protocol

from krakey.memory.graph_memory import GraphMemory
from krakey.memory.knowledge_base import KBRegistry
from krakey.memory.recall import Reranker
from krakey.memory.sleep.clustering import run_leiden_clustering
from krakey.memory.sleep.index_rebuild import rebuild_index_graph
from krakey.memory.sleep.kb_lifecycle import archive_excess_kbs, consolidate_kbs
from krakey.memory.sleep.migration import migrate_gm_to_kb

if TYPE_CHECKING:
    from krakey.runtime.stimuli.stimulus_buffer import StimulusBuffer


class AsyncChatLLM(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...


class AsyncEmbedder(Protocol):
    async def __call__(self, text: str) -> list[float]: ...


async def enter_sleep_mode(
    gm: GraphMemory, reg: KBRegistry, channels: "StimulusBuffer",
    *, llm: AsyncChatLLM, embedder: AsyncEmbedder,
    reranker: Reranker | None = None,
    log_dir: str | Path = "workspace/logs",
    min_community_size: int = 1,
    kb_consolidation_threshold: float = 0.85,
    kb_index_max: int = 30,
    kb_archive_pct: int = 10,
    kb_revive_threshold: float = 0.80,
    kb_dedup_top_k: int = 5,
) -> dict[str, Any]:
    """Run all 7 phases. Returns summary stats."""

    started_at = datetime.now()

    # Phase 1: pause non-urgent channels
    await channels.pause_non_urgent()

    try:
        # Phase 2: cluster + summarize
        communities = await run_leiden_clustering(
            gm, llm=llm, embedder=embedder,
            min_size=min_community_size,
        )

        # Phase 3 (+ implicit Phase 4): migrate FACT/RELATION/KNOWLEDGE
        # (incl. completed-TARGETs-now-FACT) into KB.
        before_targets = await gm.count_by_category("TARGET")
        migrate_stats = await migrate_gm_to_kb(
            gm, reg,
            llm=llm, reranker=reranker,
            dedup_top_k=kb_dedup_top_k,
            min_community_size=min_community_size,
            revive_threshold=kb_revive_threshold,
        )

        # Phase 3.5: consolidate KBs whose index vectors are cosine-close.
        # Done after migration so newly-added entries inform the merge.
        consolidate_stats = await consolidate_kbs(
            reg, threshold=kb_consolidation_threshold,
        )

        # Phase 3.6: archive excess KBs (drop GM index node, keep file).
        archive_stats = await archive_excess_kbs(
            reg, gm, max_count=kb_index_max,
            archive_pct=kb_archive_pct,
        )

        # Phase 4: explicit no-op (completed targets already migrated; open
        # targets stay in GM untouched).
        targets_preserved = await gm.count_by_category("TARGET")

        # Phase 5: clear FOCUS
        focus_cleared = await gm.delete_by_category("FOCUS")

        # Phase 6: rebuild Index Graph (only active KBs get GM nodes)
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
            "kbs_revived": migrate_stats["kbs_revived"],
            "skipped_no_community": migrate_stats["skipped_no_community"],
            "skipped_small_community": migrate_stats["skipped_small_community"],
            "kbs_merged": consolidate_stats["merged"],
            "kbs_archived": archive_stats["archived"],
            "kbs_active_after": archive_stats["active_after"],
            "focus_cleared": focus_cleared,
            "targets_preserved": targets_preserved,
            "targets_before": before_targets,
            "index_nodes": idx_stats["index_nodes"],
            "index_edges_added": idx_stats["edges_added"],
        }
        await _write_daily_log(log_dir, result)
        return result
    finally:
        # The buffer owns the channels AND the queue, so resume_all
        # needs no arg — it just hands each paused channel its push
        # callback again. The legacy ``active_buffer()`` workaround
        # disappeared with the ChannelRegistry merge.
        await channels.resume_all()


# ---------------- helpers ----------------


async def _write_daily_log(log_dir: str | Path, record: dict[str, Any]) -> None:
    p = Path(log_dir)
    p.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y-%m-%d")
    log_file = p / f"sleep-{day}.jsonl"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
