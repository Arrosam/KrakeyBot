"""Memory-related config sections: GraphMemory + KnowledgeBase + Sleep
+ Safety bounds.

Grouped together because they share the "memory subsystem tuning"
theme — GM is the working memory, KB is the long-term store, Sleep
moves data between them, Safety caps the upper bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphMemorySection:
    db_path: str = "workspace/data/graph_memory.sqlite"
    auto_ingest_similarity_threshold: float = 0.92
    # Hard upper bound on vec_search top_k per stimulus. The actual
    # top_k is computed dynamically from the screening-token target
    # (see ``recall_screening_token_multiplier``) and capped by this.
    # Keep generous: too small starves the oversampling pool that the
    # multiplier exists to fill.
    recall_per_stimulus_k: int = 50
    # Per-stimulus screening pool size, expressed as a MULTIPLIER of
    # the role's ``recall_token_budget``. With multiplier=3.0 and a
    # budget of 600 tokens, vec_search aims to surface ~1800 tokens
    # worth of candidate nodes per stimulus — enough that the dedup +
    # weight-merge across stimuli leaves the final ``finalize()``
    # token-budget cut with real selection choice instead of just
    # admitting everything. Multiplier=1.0 disables oversampling
    # (screening pool ≈ final cut). Capped by ``recall_per_stimulus_k``
    # in either direction.
    recall_screening_token_multiplier: float = 3.0
    neighbor_expand_depth: int = 1


@dataclass
class KnowledgeBaseSection:
    dir: str = "workspace/data/knowledge_bases"


@dataclass
class SleepSection:
    max_duration_seconds: int = 7200
    # Communities below this size stay in GM (don't get migrated to a KB).
    # Default 2 = skip pure singletons.
    min_community_size: int = 2
    # KB consolidation: pairwise-merge active KBs whose index vectors
    # (mean of member entry embeddings) are at least this cosine-close.
    kb_consolidation_threshold: float = 0.85
    # When active KB count exceeds this, archive the least-important
    # `kb_archive_pct` percent (importance = entry_count * mean importance).
    # Archived KBs keep their files + entries on disk and their index
    # vector in kb_registry — they just lose their GM index node so they
    # stop bloating recall.
    kb_index_max: int = 30
    kb_archive_pct: int = 10
    # When sleep would create a fresh KB for a new community, first compare
    # the community summary embedding against archived KBs' index vectors;
    # if the cosine similarity to one is at least this, revive that
    # archived KB and write the new entries into it instead. Models the
    # "forgot a topic, then re-encountered it" relearning shortcut.
    kb_revive_threshold: float = 0.80


@dataclass
class SafetySection:
    gm_node_hard_limit: int = 500
    max_consecutive_no_action: int = 50


def _build_graph_memory(raw: dict[str, Any]) -> GraphMemorySection:
    d = GraphMemorySection()
    return GraphMemorySection(
        db_path=str(raw.get("db_path", d.db_path)),
        auto_ingest_similarity_threshold=float(
            raw.get("auto_ingest_similarity_threshold",
                     d.auto_ingest_similarity_threshold)
        ),
        recall_per_stimulus_k=int(raw.get("recall_per_stimulus_k",
                                              d.recall_per_stimulus_k)),
        recall_screening_token_multiplier=float(
            raw.get("recall_screening_token_multiplier",
                     d.recall_screening_token_multiplier)
        ),
        neighbor_expand_depth=int(raw.get("neighbor_expand_depth",
                                              d.neighbor_expand_depth)),
    )


def _build_kb(raw: dict[str, Any]) -> KnowledgeBaseSection:
    d = KnowledgeBaseSection()
    return KnowledgeBaseSection(dir=str(raw.get("dir", d.dir)))


def _build_sleep(raw: dict[str, Any]) -> SleepSection:
    d = SleepSection()
    return SleepSection(
        max_duration_seconds=int(raw.get("max_duration_seconds",
                                              d.max_duration_seconds)),
        min_community_size=int(raw.get("min_community_size",
                                           d.min_community_size)),
        kb_consolidation_threshold=float(
            raw.get("kb_consolidation_threshold",
                     d.kb_consolidation_threshold)
        ),
        kb_index_max=int(raw.get("kb_index_max", d.kb_index_max)),
        kb_archive_pct=int(raw.get("kb_archive_pct", d.kb_archive_pct)),
        kb_revive_threshold=float(raw.get("kb_revive_threshold",
                                               d.kb_revive_threshold)),
    )


def _build_safety(raw: dict[str, Any]) -> SafetySection:
    d = SafetySection()
    return SafetySection(
        gm_node_hard_limit=int(raw.get("gm_node_hard_limit",
                                           d.gm_node_hard_limit)),
        max_consecutive_no_action=int(
            raw.get("max_consecutive_no_action", d.max_consecutive_no_action)
        ),
    )
