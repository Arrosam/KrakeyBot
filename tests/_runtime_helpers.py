"""Test-only Runtime builder + minimal LLM/embedder fakes.

Lives under `tests/` so the production package never pulls in test doubles.
Import from tests as: `from tests._runtime_helpers import build_runtime_with_fakes`.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
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
    # ONE tmpdir holds every per-plugin config.yaml for this test.
    # Same shape as the production workspace/plugins/ root: each
    # plugin gets its own subfolder with a config.yaml inside.
    plugin_configs_dir = tempfile.mkdtemp(prefix="krakey_test_plugcfg_")
    # Ditto for self_model: it's a mutable file, and bootstrap tests
    # rewrite it. Without an override, concurrent test writes trample
    # the production workspace/self_model.yaml.
    self_model_path = f"{tempfile.mkdtemp(prefix='krakey_test_sm_')}/self_model.yaml"
    # And the in_mind Reflect's state file. Tests that enable
    # in_mind_note would otherwise dispatch update_in_mind into
    # the production workspace/data/in_mind.json — same class of
    # leak as the web_chat history bug.
    in_mind_state_path = (
        f"{tempfile.mkdtemp(prefix='krakey_test_im_')}/in_mind.json"
    )
    from src.models.config import (
        Config, FatigueSection, GraphMemorySection,
        HibernateSection, KnowledgeBaseSection, LLMParams, LLMSection,
        Provider, SafetySection, SleepSection, TagBinding,
    )

    # The Runtime reads `core_params("self_thinking")` for window /
    # recall / overall-prompt budgets; without it the budget math
    # falls back to LLMParams defaults. Tests stamp an explicit
    # max_input_tokens=16000 tag so budget enforcement triggers
    # predictably.
    default_self_params = LLMParams(max_input_tokens=16_000)
    cfg = Config(
        llm=LLMSection(
            providers={"_test_fake": Provider(
                type="openai_compatible", base_url="http://test",
                api_key="test",
            )},
            tags={"_test_default": TagBinding(
                provider="_test_fake/_test_model",
                params=default_self_params,
            )},
            core_purposes={"self_thinking": "_test_default"},
        ),
        hibernate=HibernateSection(min_interval=1, max_interval=60,
                                    default_interval=1),
        fatigue=FatigueSection(gm_node_soft_limit=200,
                                force_sleep_threshold=120,
                                thresholds={}),
        # Test convenience: opt into the historical default plugin set
        # when the caller didn't say otherwise. Tests on the zero-plugin
        # path pass ``reflects=[]`` to opt out explicitly.
        #
        # The dashboard plugin is included so the web_chat_reply
        # tentacle is registered (existing tests dispatch to it). Its
        # per-plugin config (planted below) sets port=0 so the Web UI
        # server never binds — only the sensory + tentacle live.
        plugins=(
            list(reflects)
            if reflects is not None
            else [
                "hypothalamus",
                "recall",
                "dashboard",
            ]
        ),
        graph_memory=GraphMemorySection(
            db_path=gm_path, auto_ingest_similarity_threshold=0.9,
            recall_per_stimulus_k=5, neighbor_expand_depth=1,
        ),
        knowledge_base=KnowledgeBaseSection(dir=kb_dir),
        sleep=SleepSection(max_duration_seconds=7200,
                              min_community_size=1),
        safety=SafetySection(gm_node_hard_limit=500,
                               max_consecutive_no_action=50),
    )
    # Pre-cache the LLMClient slots that plugins will resolve through
    # ctx.get_llm_for_tag. We point the helper's "_test_default" tag at
    # the caller's hypo_llm (ScriptedLLM in most tests) so the
    # hypothalamus Reflect's `translator` purpose lands on the
    # scripted fake instead of trying to make a real HTTP call.
    llm_clients_by_tag: dict = {"_test_default": hypo_llm}

    # Plant a per-plugin config for hypothalamus that binds
    # its `translator` purpose to "_test_default" — without this, the
    # plugin's factory reads no purposes from ctx.config,
    # ctx.get_llm_for_tag(None) returns None, and the factory skips
    # registration (correct behavior, but would break tests that
    # expect the Reflect to be registered).
    Path(plugin_configs_dir, "hypothalamus").mkdir(
        parents=True, exist_ok=True,
    )
    Path(plugin_configs_dir, "hypothalamus", "config.yaml").write_text(
        "llm_purposes:\n  translator: _test_default\n", encoding="utf-8",
    )
    # Dashboard plugin (which now owns the embedded web chat). Plant
    # config that:
    #   * points history at a tmpdir so tests don't pollute
    #     workspace/data/web_chat.jsonl
    #   * sets port=0 so the dashboard's sensory.start() short-circuits
    #     before binding a port (no flaky port conflicts in tests)
    Path(plugin_configs_dir, "dashboard").mkdir(
        parents=True, exist_ok=True,
    )
    Path(plugin_configs_dir, "dashboard", "config.yaml").write_text(
        f"port: 0\nhistory_path: {chat_dir}/chat.jsonl\n",
        encoding="utf-8",
    )

    deps = RuntimeDeps(
        config=cfg, self_llm=self_llm, hypo_llm=hypo_llm,
        compact_llm=compact_llm or ScriptedLLM(),
        classify_llm=classify_llm or ScriptedLLM(),
        embedder=embedder or NullEmbedder(),
        reranker=reranker,
        plugin_configs_root=plugin_configs_dir,
        self_model_path=self_model_path,
        in_mind_state_path=in_mind_state_path,
        llm_clients_by_tag=llm_clients_by_tag,
    )
    runtime = Runtime(
        deps, hibernate_min=hibernate_min, hibernate_max=hibernate_max,
        is_bootstrap_override=False if skip_bootstrap else None,
    )
    # Stash a copy of the resolved test paths + LLM cache so test
    # helpers (e.g. _minimal_deps_for_runtime) can reconstruct an
    # equivalent deps when re-invoking _register_reflects_from_config.
    runtime._test_reflect_configs_root = plugin_configs_dir
    runtime._test_llm_clients_by_tag = llm_clients_by_tag
    return runtime
