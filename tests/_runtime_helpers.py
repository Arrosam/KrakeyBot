"""Test-only Runtime builder + minimal LLM/embedder fakes.

Lives under `tests/` so the production package never pulls in test doubles.
Import from tests as: `from tests._runtime_helpers import build_runtime_with_fakes`.
"""
from __future__ import annotations

import tempfile
from typing import Protocol

from src.main import Runtime, RuntimeDeps
from src.memory.recall import Reranker


class ChatLike(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...


class AsyncEmbedder(Protocol):
    async def __call__(self, text: str) -> list[float]: ...


class NullEmbedder:
    async def __call__(self, text: str) -> list[float]:
        return [0.0]


class ScriptedLLM:
    def __init__(self, responses=None):
        self._responses = list(responses or [])

    async def chat(self, messages, **kwargs):
        if not self._responses:
            return ""
        return self._responses.pop(0)


def build_runtime_with_fakes(*, self_llm: ChatLike, hypo_llm: ChatLike,
                              compact_llm: ChatLike | None = None,
                              classify_llm: ChatLike | None = None,
                              embedder: AsyncEmbedder | None = None,
                              reranker: Reranker | None = None,
                              hibernate_min: float = 0.01,
                              hibernate_max: float = 5.0,
                              gm_path: str = ":memory:",
                              kb_dir: str | None = None,
                              skip_bootstrap: bool = True) -> Runtime:
    """In-memory Runtime with injectable doubles. CLI sensory disabled.

    `kb_dir` defaults to a fresh `tempfile.mkdtemp()` so KB files written
    during sleep migration never touch the production workspace.
    """
    if kb_dir is None:
        kb_dir = tempfile.mkdtemp(prefix="krakey_test_kb_")
    # Same isolation pattern for the web chat JSONL: without overriding,
    # main.py falls back to "workspace/data/web_chat.jsonl" — the production
    # path — and pytest writes test fixture messages into the real chat log.
    chat_dir = tempfile.mkdtemp(prefix="krakey_test_chat_")
    from src.models.config import (
        Config, DashboardSection, FatigueSection, GraphMemorySection,
        HibernateSection, KnowledgeBaseSection, LLMSection, SafetySection,
        SleepSection, SlidingWindowSection,
    )

    cfg = Config(
        llm=LLMSection(providers={}, roles={}),
        hibernate=HibernateSection(min_interval=1, max_interval=60,
                                    default_interval=1),
        fatigue=FatigueSection(gm_node_soft_limit=200,
                                force_sleep_threshold=120,
                                thresholds={}),
        sliding_window=SlidingWindowSection(max_tokens=4096),
        graph_memory=GraphMemorySection(
            db_path=gm_path, auto_ingest_similarity_threshold=0.9,
            recall_per_stimulus_k=5, max_recall_nodes=20,
            neighbor_expand_depth=1,
        ),
        knowledge_base=KnowledgeBaseSection(dir=kb_dir),
        plugins={
            # `enabled` is loader-owned and defaults to False — must be
            # set explicitly for tests that expect the plugin to load.
            # Mirror the "default-on" set the manifests used to declare
            # so existing tests keep their assumptions.
            "web_chat": {
                "enabled": True,
                # Keep web chat history inside the test tmpdir so it
                # doesn't bleed into workspace/data/web_chat.jsonl.
                "history_path": f"{chat_dir}/chat.jsonl",
            },
            "memory_recall": {"enabled": True},
            # `search` stays OFF in the helper: it hits the real network
            # (DuckDuckGo) and tests should opt in deliberately.
        },
        sleep=SleepSection(max_duration_seconds=7200,
                              min_community_size=1),
        safety=SafetySection(gm_node_hard_limit=500,
                               max_consecutive_no_action=50),
        dashboard=DashboardSection(enabled=False),
    )
    deps = RuntimeDeps(
        config=cfg, self_llm=self_llm, hypo_llm=hypo_llm,
        compact_llm=compact_llm or ScriptedLLM(),
        classify_llm=classify_llm or ScriptedLLM(),
        embedder=embedder or NullEmbedder(),
        reranker=reranker,
    )
    return Runtime(
        deps, hibernate_min=hibernate_min, hibernate_max=hibernate_max,
        is_bootstrap_override=False if skip_bootstrap else None,
    )
