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
                              skip_bootstrap: bool = True,
                              reflects: list[str] | None = None) -> Runtime:
    """In-memory Runtime with injectable doubles. CLI sensory disabled.

    `kb_dir` defaults to a fresh `tempfile.mkdtemp()` so KB files written
    during sleep migration never touch the production workspace.

    `reflects` is the ordered list of Reflect names to register at
    startup. ``None`` (the default) opts the test helper into the two
    in-tree Reflects that almost every existing test expects to be
    available — preserves the historical "Hypothalamus translates,
    recall populates [GRAPH MEMORY]" assumption without forcing every
    test to opt in. Tests exercising the zero-plugin path pass
    ``reflects=[]`` explicitly. This default is a TEST convenience
    only; production config defaults to registering nothing per the
    "all plugins default off" architectural rule.
    """
    if kb_dir is None:
        kb_dir = tempfile.mkdtemp(prefix="krakey_test_kb_")
    # Same isolation pattern for the web chat JSONL: without overriding,
    # main.py falls back to "workspace/data/web_chat.jsonl" — the production
    # path — and pytest writes test fixture messages into the real chat log.
    chat_dir = tempfile.mkdtemp(prefix="krakey_test_chat_")
    # Per-plugin config YAMLs under workspace/plugin-configs/ shadow
    # the legacy dict below when a file exists. Point the store at a
    # fresh empty tmpdir so the helper's plugin overrides actually
    # take effect (otherwise prod's web_chat.yaml wins → test
    # messages like "Hi there!" leak into the real user chat log).
    plugin_configs_dir = tempfile.mkdtemp(prefix="krakey_test_plugcfg_")
    # Ditto for self_model: it's a mutable file, and bootstrap tests
    # rewrite it. Without an override, concurrent test writes trample
    # the production workspace/self_model.yaml.
    self_model_path = f"{tempfile.mkdtemp(prefix='krakey_test_sm_')}/self_model.yaml"
    from src.models.config import (
        Config, DashboardSection, FatigueSection, GraphMemorySection,
        HibernateSection, KnowledgeBaseSection, LLMParams, LLMSection,
        RoleBinding, SafetySection, SleepSection,
    )

    # The Runtime reads self.config.llm.roles["self"].params for
    # window/recall budgets; without a "self" role, construction
    # KeyErrors. We stamp a minimal RoleBinding with LLMParams
    # defaults so tests get the standard 40%/3000-token split without
    # having to spell it out.
    default_self_params = LLMParams(max_input_tokens=16_000)
    cfg = Config(
        llm=LLMSection(
            providers={},
            roles={"self": RoleBinding(provider="", model="",
                                          params=default_self_params)},
        ),
        hibernate=HibernateSection(min_interval=1, max_interval=60,
                                    default_interval=1),
        fatigue=FatigueSection(gm_node_soft_limit=200,
                                force_sleep_threshold=120,
                                thresholds={}),
        # Test convenience: opt into both built-in Reflects when the
        # caller didn't say otherwise. Production config default is
        # zero plugins, but most existing tests assume the historical
        # "Hypothalamus + recall both registered" shape and shouldn't
        # have to re-state it. Tests on the zero-plugin path pass
        # ``reflects=[]`` to opt out explicitly.
        reflects=(
            list(reflects)
            if reflects is not None
            else ["default_hypothalamus", "default_recall_anchor"]
        ),
        graph_memory=GraphMemorySection(
            db_path=gm_path, auto_ingest_similarity_threshold=0.9,
            recall_per_stimulus_k=5, neighbor_expand_depth=1,
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
        plugin_configs_root=plugin_configs_dir,
        self_model_path=self_model_path,
    )
    return Runtime(
        deps, hibernate_min=hibernate_min, hibernate_max=hibernate_max,
        is_bootstrap_override=False if skip_bootstrap else None,
    )
