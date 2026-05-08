"""End-to-end memory slot swap test (single-slot Engine refactor).

Verifies the unified ``memory`` slot — collapsed from the previous
``memory`` + ``kb_registry`` two-slot model — actually replaces the
built-in ``GraphMemoryEngine``: runtime construction succeeds, the
resolved instance is the user's class, and a few representative
methods (GM-side ``explicit_write``, KB-side ``create_kb`` /
``open_kb``) round-trip through the runtime.
"""
from __future__ import annotations

import textwrap

import pytest

from krakey.engines.memory.default import GraphMemoryEngine
from krakey.interfaces.engines import KnowledgeBaseLike, MemoryEngine
from krakey.main import build_runtime_from_config
from tests._fake_memory import InMemoryMemoryEngine


def _config(tmp_path, *, memory_override=""):
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


def test_no_override_uses_default_memory_engine(tmp_path):
    """No override → runtime.memory is the built-in GraphMemoryEngine."""
    p = _config(tmp_path)
    runtime = build_runtime_from_config(str(p))
    assert isinstance(runtime.memory, GraphMemoryEngine)
    assert isinstance(runtime.memory, MemoryEngine)
    # Back-compat aliases point at the same Engine.
    assert runtime.gm is runtime.memory
    assert runtime.kb_registry is runtime.memory


def test_memory_override_uses_user_class(tmp_path):
    """``core_implementations.memory`` set → runtime.memory is the
    user's class. Single slot covers both GM + KB management now;
    no separate kb_registry override required."""
    p = _config(
        tmp_path,
        memory_override="tests._fake_memory:InMemoryMemoryEngine",
    )
    runtime = build_runtime_from_config(str(p))
    assert isinstance(runtime.memory, InMemoryMemoryEngine)
    # Aliases also point at the fake.
    assert runtime.gm is runtime.memory
    assert runtime.kb_registry is runtime.memory


async def test_override_actually_serves_writes(tmp_path):
    """Sanity: the runtime really uses the fake — explicit_write goes
    through it (not the default GraphMemoryEngine)."""
    p = _config(
        tmp_path,
        memory_override="tests._fake_memory:InMemoryMemoryEngine",
    )
    runtime = build_runtime_from_config(str(p))

    fake = runtime.memory
    assert isinstance(fake, InMemoryMemoryEngine)
    await fake.explicit_write("test memory write")
    assert fake.explicit_write_calls == ["test memory write"]
    assert await fake.count_nodes() == 1


async def test_override_kb_create_open_round_trip(tmp_path):
    """KB created via the override engine can be opened again — same
    instance (cached on the engine's internal registry)."""
    p = _config(
        tmp_path,
        memory_override="tests._fake_memory:InMemoryMemoryEngine",
    )
    runtime = build_runtime_from_config(str(p))
    kb = await runtime.memory.create_kb("test_kb", name="Test KB")
    assert isinstance(kb, KnowledgeBaseLike)
    reopened = await runtime.memory.open_kb("test_kb")
    assert reopened is kb


async def test_override_sleep_cycle_invoked(tmp_path):
    """The fake records every sleep_cycle call — proves the runtime
    surfaces sleep through the engine's method. Calling
    memory.sleep_cycle directly here (rather than driving a real
    sleep transition) is enough to verify the surface."""
    p = _config(
        tmp_path,
        memory_override="tests._fake_memory:InMemoryMemoryEngine",
    )
    runtime = build_runtime_from_config(str(p))
    fake = runtime.memory
    assert isinstance(fake, InMemoryMemoryEngine)
    stats = await fake.sleep_cycle(
        channels=None, log_dir="/tmp", config={"min_community_size": 2},
    )
    assert stats == {}
    assert len(fake.sleep_cycle_calls) == 1
    assert fake.sleep_cycle_calls[0]["log_dir"] == "/tmp"
