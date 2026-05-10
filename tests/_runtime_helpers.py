"""Test-only Runtime builder + minimal LLM/embedder fakes.

Lives under `tests/` so the production package never pulls in test doubles.
Import from tests as: `from tests._runtime_helpers import build_runtime_with_fakes`.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Protocol

from krakey.main import Runtime, RuntimeDeps


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


def build_runtime_with_fakes(*, self_llm: ChatLike,
                              decision_translator_llm: ChatLike | None = None,
                              hypo_llm: ChatLike | None = None,
                              compact_llm: ChatLike | None = None,
                              classify_llm: ChatLike | None = None,
                              embedder: AsyncEmbedder | None = None,
                              idle_min: float = 0.01,
                              idle_max: float = 5.0,
                              gm_path: str = ":memory:",
                              kb_dir: str | None = None,
                              skip_bootstrap: bool = True,
                              self_model_path: str | None = None,
                              modifiers: list[str] | None = None) -> Runtime:
    """In-memory Runtime with injectable doubles. CLI channel disabled.

    `kb_dir` defaults to a fresh `tempfile.mkdtemp()` so KB files written
    during sleep migration never touch the production workspace.

    `modifiers` is the ordered list of Modifier names to register at
    startup. ``None`` (the default) opts the test helper into the two
    in-tree Modifiers that almost every existing test expects to be
    available — preserves the historical "Hypothalamus translates,
    recall populates [GRAPH MEMORY]" assumption without forcing every
    test to opt in. Tests exercising the zero-plugin path pass
    ``modifiers=[]`` explicitly. This default is a TEST convenience
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
    if self_model_path is None:
        self_model_path = (
            f"{tempfile.mkdtemp(prefix='krakey_test_sm_')}/self_model.yaml"
        )
    # And the in_mind Modifier's state file. Tests that enable
    # in_mind_note would otherwise dispatch update_in_mind into
    # the production workspace/data/in_mind.json — same class of
    # leak as the web_chat history bug.
    in_mind_state_path = (
        f"{tempfile.mkdtemp(prefix='krakey_test_im_')}/in_mind.json"
    )
    from krakey.models.config import (
        Config, FatigueSection, GraphMemorySection,
        IdleSection, KnowledgeBaseSection, LLMParams, LLMSection,
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
        idle=IdleSection(min_interval=1, max_interval=60,
                                    default_interval=1),
        fatigue=FatigueSection(gm_node_soft_limit=200,
                                force_sleep_threshold=120,
                                thresholds={}),
        # Test convenience: opt into the historical default plugin set
        # when the caller didn't say otherwise. Tests on the zero-plugin
        # path pass ``modifiers=[]`` to opt out explicitly.
        #
        # The dashboard plugin is included so the web_chat_reply
        # tool is registered (existing tests dispatch to it). Its
        # per-plugin config (planted below) sets port=0 so the Web UI
        # server never binds — only the channel + tool live.
        plugins=(
            list(modifiers)
            if modifiers is not None
            else [
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
    # ctx.get_llm_for_tag (which routes through
    # llm_factory.client_for_tag). The "_test_default" tag is used by
    # any plugin (or Engine) that calls ``client_for_core_purpose`` /
    # ``ctx.get_llm_for_tag`` during a test — point it at the scripted
    # hypo_llm fake when the caller supplied one, otherwise at an empty
    # ScriptedLLM so callers don't have to wire one up for tests that
    # never touch the LLM-translator path. Seeded straight into the
    # factory's cache below.
    seed_clients: dict = {
        "_test_default": hypo_llm if hypo_llm is not None else ScriptedLLM(),
    }
    # Dashboard plugin (which now owns the embedded web chat). Plant
    # config that:
    #   * points history at a tmpdir so tests don't pollute
    #     workspace/data/web_chat.jsonl
    #   * sets port=0 so the dashboard's channel.start() short-circuits
    #     before binding a port (no flaky port conflicts in tests)
    Path(plugin_configs_dir, "dashboard").mkdir(
        parents=True, exist_ok=True,
    )
    Path(plugin_configs_dir, "dashboard", "config.yaml").write_text(
        f"port: 0\nhistory_path: {chat_dir}/chat.jsonl\n",
        encoding="utf-8",
    )

    # Sliding window state file: per-test tmpdir so heartbeat
    # rounds don't bleed across runs. Tests that want to verify
    # cross-restart persistence (test_sliding_window.py) pass an
    # explicit path via build_runtime_with_fakes' kwargs and
    # construct two runtimes pointing at the same file.
    sliding_window_state_path = (
        f"{tempfile.mkdtemp(prefix='krakey_test_sw_')}/sliding_window.json"
    )
    # Provide an LLMClientFactoryEngine so Runtime can resolve the
    # reranker Engine slot (the default reranker constructor takes a
    # factory). Even if cfg has no reranker tag bound, the factory's
    # rerank_client() returns None and the Engine falls back to
    # preserve-order scoring.
    from krakey.engines.llm_factory.default import (
        DefaultLLMClientFactoryEngine,
    )
    llm_factory = DefaultLLMClientFactoryEngine(cfg)
    # Seed the test fakes into the factory's cache so plugin lookups
    # via ctx.get_llm_for_tag (which routes through
    # llm_factory.client_for_tag) return them.
    llm_factory._cache.update(seed_clients)

    deps = RuntimeDeps(
        config=cfg, self_llm=self_llm,
        compact_llm=compact_llm or ScriptedLLM(),
        classify_llm=classify_llm or ScriptedLLM(),
        embedder=embedder or NullEmbedder(),
        llm_factory=llm_factory,
        plugin_configs_root=plugin_configs_dir,
        self_model_path=self_model_path,
        in_mind_state_path=in_mind_state_path,
        sliding_window_state_path=sliding_window_state_path,
    )
    runtime = Runtime(
        deps, idle_min=idle_min, idle_max=idle_max,
    )
    if skip_bootstrap:
        anchor = runtime.modifiers.by_role("bootstrap")
        if anchor is not None and hasattr(anchor, "force_active"):
            anchor.force_active(False)
    # When the caller wants the LLM-translator decision path, pass
    # ``decision_translator_llm=ScriptedLLM([...json...])``: we swap
    # the resolved DecisionEngine to ``HypothalamusDecisionEngine``
    # and bind its ``hypothalamus`` core-purpose tag to the supplied
    # fake. Otherwise the default ``ToolCallParserDecisionEngine``
    # (scripted ``<tool_call>`` parser) stays in place.
    if decision_translator_llm is not None:
        from krakey.engines.decision.hypothalamus import (
            HypothalamusDecisionEngine,
        )
        # Bind the fake under the ``hypothalamus`` core-purpose tag
        # so the engine's ``factory.client_for_core_purpose("hypothalamus")``
        # call returns it without needing a real cfg entry.
        cfg.llm.core_purposes["hypothalamus"] = "_test_translator"
        llm_factory._cache["_test_translator"] = decision_translator_llm
        runtime.decision = HypothalamusDecisionEngine(
            cfg=cfg, factory=llm_factory,
        )
    # Stash the resolved test paths + factory so test helpers
    # (e.g. _minimal_deps_for_runtime) can reconstruct an equivalent
    # deps when re-invoking _register_modifiers_from_config.
    runtime._test_modifier_configs_root = plugin_configs_dir
    runtime._test_llm_factory = llm_factory
    return runtime
