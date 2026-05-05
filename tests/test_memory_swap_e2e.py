"""End-to-end memory slot swap test.

Verifies the memory + kb_registry slots actually replace the built-in
GraphMemory + KBRegistry — runtime construction succeeds, the resolved
instances are the user's classes, and a few representative methods on
each round-trip through the runtime.
"""
from __future__ import annotations

import textwrap

import pytest

from krakey.interfaces.services.memory import (
    KBRegistryService,
    KnowledgeBaseLike,
    MemoryService,
)
from krakey.main import build_runtime_from_config
from tests._fake_memory import (
    InMemoryKBRegistryService,
    InMemoryMemoryService,
)


def _config(tmp_path, *, memory_override="", kb_override=""):
    body = f"""
        llm:
          providers:
            P:
              type: openai_compatible
              base_url: "http://x"
              api_key: "k"
          tags:
            t:
              provider: "P/m"
              params: {{max_output_tokens: 100}}
          core_purposes:
            self_thinking: t
            compact: t
            classifier: t
          embedding: t
        core_implementations:
          memory: "{memory_override}"
          kb_registry: "{kb_override}"
        idle:
          min_interval: 1
          max_interval: 60
          default_interval: 1
        graph_memory:
          db_path: ":memory:"
          auto_ingest_similarity_threshold: 0.9
          recall_per_stimulus_k: 5
          neighbor_expand_depth: 1
        knowledge_base:
          dir: "kb"
        sleep:
          max_duration_seconds: 7200
        safety:
          gm_node_hard_limit: 500
          max_consecutive_no_action: 50
        fatigue:
          gm_node_soft_limit: 100
          force_sleep_threshold: 60
          thresholds: {{}}
    """
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_no_override_uses_default_memory(tmp_path):
    p = _config(tmp_path)
    runtime = build_runtime_from_config(str(p))
    # Built-in GraphMemory satisfies MemoryService (covered in
    # test_memory_protocols.py); name-check is the simplest signal
    # that we're not getting the fake.
    assert isinstance(runtime.gm, MemoryService)
    assert type(runtime.gm).__name__ == "GraphMemory"


def test_memory_override_uses_user_class(tmp_path):
    p = _config(
        tmp_path,
        memory_override="tests._fake_memory:InMemoryMemoryService",
        kb_override="tests._fake_memory:InMemoryKBRegistryService",
    )
    runtime = build_runtime_from_config(str(p))
    assert isinstance(runtime.gm, InMemoryMemoryService)
    assert isinstance(runtime.kb_registry, InMemoryKBRegistryService)


async def test_override_actually_serves_writes(tmp_path):
    """Sanity: the runtime really uses the fake — explicit_write goes
    through it (not the would-be GraphMemory)."""
    p = _config(
        tmp_path,
        memory_override="tests._fake_memory:InMemoryMemoryService",
        kb_override="tests._fake_memory:InMemoryKBRegistryService",
    )
    runtime = build_runtime_from_config(str(p))

    fake = runtime.gm
    assert isinstance(fake, InMemoryMemoryService)
    await fake.explicit_write("test memory write")
    assert fake.explicit_write_calls == ["test memory write"]
    assert await fake.count_nodes() == 1


async def test_override_kb_create_open_round_trip(tmp_path):
    """KB created via the override registry can be opened again."""
    p = _config(
        tmp_path,
        memory_override="tests._fake_memory:InMemoryMemoryService",
        kb_override="tests._fake_memory:InMemoryKBRegistryService",
    )
    runtime = build_runtime_from_config(str(p))
    registry = runtime.kb_registry
    kb = await registry.create_kb("test_kb", name="Test KB")
    assert isinstance(kb, KnowledgeBaseLike)
    reopened = await registry.open_kb("test_kb")
    assert reopened is kb


def test_only_memory_overridden_kb_default_breaks_at_kb_protocol(tmp_path):
    """If user overrides ONLY memory and the default KBRegistry expects
    a SQLite-backed gm (gm._require()), startup will fail at KB build.

    This test pins that observable behavior so users hit a clear error
    and the docs' "if you override memory, also override kb_registry"
    advice has teeth."""
    p = _config(
        tmp_path,
        memory_override="tests._fake_memory:InMemoryMemoryService",
        # No kb_override — default KBRegistry runs.
    )
    # Default KBRegistry's __init__ doesn't call gm._require() yet
    # (lazy); it'd happen on the first DB op. Construction succeeds,
    # but build_runtime_from_config doesn't crash here. We instead
    # observe that runtime.kb_registry is the default class — the
    # user is left to discover the breakage on first KB operation.
    #
    # If a future commit moves _require() into KBRegistry.__init__,
    # this test would fail loud — refactor it to assert the new error.
    runtime = build_runtime_from_config(str(p))
    assert type(runtime.kb_registry).__name__ == "KBRegistry"
    assert isinstance(runtime.gm, InMemoryMemoryService)
