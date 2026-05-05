"""Embedder slot — verify build_runtime_from_config respects the override.

Default path: when no override is set, the wrapped embed_client is used.
Override path: when ``core_implementations.embedder`` points at a class,
that class's instance is used (a real e2e heartbeat test would also
verify GraphMemory uses the override; that lives in the broader e2e
swap test in test_core_impl_swap_e2e.py).
"""
from __future__ import annotations

import textwrap

import pytest

from krakey.llm.resolve import AsyncEmbedder
from krakey.main import build_runtime_from_config


# A test-module-level embedder so the resolver's importlib path can find
# it via dotted reference. Must be at module top level (not inside a
# test fn) for importlib.import_module to resolve it.
class FakeUserEmbedder:
    """User-supplied embedder — returns a hand-rolled vector per call."""

    def __init__(self):
        self.calls: list[str] = []

    async def __call__(self, text: str) -> list[float]:
        self.calls.append(text)
        # Deterministic but non-trivial vector so consumers can tell it
        # apart from "all zeros" defaults.
        return [float(len(text))] * 8


class BadEmbedder:
    """Doesn't satisfy AsyncEmbedder — no __call__ method."""

    def shrug(self) -> str:
        return "no"


def _write_minimal_config(tmp_path, *, override: str = ""):
    """Minimal config the runtime accepts (must include the deprecated-
    section guards' required keys + a self_thinking tag binding so
    runtime exits pause mode)."""
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
          embedder: "{override}"
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


def test_no_override_uses_default_embedder(tmp_path):
    """No override → embedder is the closure-wrapped LLMClient."""
    p = _write_minimal_config(tmp_path, override="")
    runtime = build_runtime_from_config(str(p))
    assert isinstance(runtime.embedder, AsyncEmbedder)
    # The default embedder is an instance of the closure-defined
    # _DefaultEmbedder class — its class name leaks the slot identity.
    assert "Embedder" in type(runtime.embedder).__name__


def test_override_uses_user_embedder(tmp_path):
    """``core_implementations.embedder = "tests.test_embedder_slot:FakeUserEmbedder"``
    → runtime uses our user-supplied class."""
    p = _write_minimal_config(
        tmp_path,
        override="tests.test_embedder_slot:FakeUserEmbedder",
    )
    runtime = build_runtime_from_config(str(p))
    assert isinstance(runtime.embedder, FakeUserEmbedder)


async def test_override_actually_called_for_embedding(tmp_path):
    """Sanity: the runtime really calls the override (not just stores it)."""
    p = _write_minimal_config(
        tmp_path,
        override="tests.test_embedder_slot:FakeUserEmbedder",
    )
    runtime = build_runtime_from_config(str(p))
    # Reach through the embedder directly — the runtime forwards
    # arbitrary text to it. Do NOT call into GraphMemory here; we
    # only want to assert the slot is wired through.
    vec = await runtime.embedder("hello world")
    assert vec == [float(len("hello world"))] * 8
    assert runtime.embedder.calls == ["hello world"]


def test_bad_override_raises_typeerror_at_startup(tmp_path):
    """A class that doesn't satisfy AsyncEmbedder fails loud at
    build_runtime_from_config — not at first .embed() call mid-session."""
    p = _write_minimal_config(
        tmp_path,
        override="tests.test_embedder_slot:BadEmbedder",
    )
    with pytest.raises(TypeError, match="AsyncEmbedder"):
        build_runtime_from_config(str(p))
