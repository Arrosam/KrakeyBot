"""Verify the shipped ``examples/custom_embedder/`` example actually works.

Two responsibilities:
  1. Catch regressions in the docs/example: if the slot mechanism's
     contract changes, this test breaks loud and the docs need updating.
  2. Demonstrate the user-facing import path. The dotted reference
     ``examples.custom_embedder.hash_embedder:HashEmbedder`` is exactly
     what a user would write.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from krakey.llm.resolve import AsyncEmbedder
from krakey.main import build_runtime_from_config


# Ensure the examples/ directory is importable. It would be on a real
# user's PYTHONPATH because they'd `pip install` the example or have
# the cwd be the project root.
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(autouse=True)
def _add_examples_to_path():
    """Make the example importable as ``custom_embedder.hash_embedder``."""
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        yield
    finally:
        sys.path.remove(str(EXAMPLES_DIR))
        # Clear any cached import so other tests aren't affected.
        for mod_name in list(sys.modules):
            if mod_name.startswith("custom_embedder"):
                del sys.modules[mod_name]


def _config(tmp_path):
    body = """
        llm:
          providers:
            P:
              type: openai_compatible
              base_url: "http://x"
              api_key: "k"
          tags:
            t:
              provider: "P/m"
              params: {max_output_tokens: 100}
          core_purposes:
            self_thinking: t
            compact: t
            classifier: t
          embedding: t
        core_implementations:
          embedder: "custom_embedder.hash_embedder:HashEmbedder"
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
          thresholds: {}
    """
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_example_embedder_loads_and_satisfies_protocol(tmp_path):
    """The shipped example resolves through the full
    config → resolver → runtime path without any test-specific patching."""
    cfg_path = _config(tmp_path)
    runtime = build_runtime_from_config(str(cfg_path))
    assert isinstance(runtime.embedder, AsyncEmbedder)
    # Class name leaks the example identity — would fail loud if the
    # example file got renamed without updating this test (which is
    # deliberate; renaming the example IS a docs change).
    assert type(runtime.embedder).__name__ == "HashEmbedder"


async def test_example_embedder_produces_deterministic_vectors(tmp_path):
    """Sanity: the example actually does what its docstring says."""
    cfg_path = _config(tmp_path)
    runtime = build_runtime_from_config(str(cfg_path))

    v1 = await runtime.embedder("hello")
    v2 = await runtime.embedder("hello")
    v3 = await runtime.embedder("world")

    assert v1 == v2, "deterministic — same input must yield same vector"
    assert v1 != v3, "different inputs must yield different vectors"
    # Dimension is what the example advertises.
    from custom_embedder.hash_embedder import HashEmbedder
    assert len(v1) == HashEmbedder.DIM
