"""Reranker slot — verify build_runtime_from_config respects the override.

Three-state contract (see main.build_runtime_from_config):
  * no override + no tag bound  → runtime.reranker is None
  * no override + tag bound     → runtime.reranker wraps the LLMClient
  * override set                → runtime.reranker is the user's class

The "tag bound" cases require a real LLMClient setup which the
minimal config below provides (note: tag is only RESOLVED at startup
when the user actually overrides the reranker; for the override path
we explicitly avoid binding the reranker tag so we don't accidentally
test both paths in the same fixture).
"""
from __future__ import annotations

import textwrap

import pytest

from krakey.main import build_runtime_from_config
from krakey.memory.recall import Reranker


# Module-level so importlib can resolve via dotted path.
class FakeUserReranker:
    def __init__(self):
        self.calls = 0

    async def rerank(self, query: str, docs: list[str]) -> list[float]:
        self.calls += 1
        # Reverse order from input: simplest "did this run" signal.
        return list(range(len(docs), 0, -1))


class BadReranker:
    """Missing rerank() — fails Protocol."""
    def shrug(self) -> str: return "no"


def _write_config(tmp_path, *, override: str = "", reranker_tag: str = ""):
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
          reranker: "{reranker_tag}"
        core_implementations:
          reranker: "{override}"
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


def test_no_override_no_tag_yields_none(tmp_path):
    """No override + no `llm.reranker` tag → runtime.reranker is None."""
    p = _write_config(tmp_path, override="", reranker_tag="")
    runtime = build_runtime_from_config(str(p))
    assert runtime.reranker is None


def test_no_override_with_tag_yields_default_adapter(tmp_path):
    """`llm.reranker: t` + no override → runtime.reranker is the default
    adapter wrapping the tag-resolved LLMClient."""
    p = _write_config(tmp_path, override="", reranker_tag="t")
    runtime = build_runtime_from_config(str(p))
    assert runtime.reranker is not None
    assert isinstance(runtime.reranker, Reranker)
    assert "Reranker" in type(runtime.reranker).__name__


def test_override_yields_user_reranker(tmp_path):
    """``core_implementations.reranker = "tests.test_reranker_slot:FakeUserReranker"``
    → runtime.reranker is the user instance, regardless of tag."""
    p = _write_config(
        tmp_path,
        override="tests.test_reranker_slot:FakeUserReranker",
        reranker_tag="",
    )
    runtime = build_runtime_from_config(str(p))
    assert isinstance(runtime.reranker, FakeUserReranker)


async def test_override_actually_invoked(tmp_path):
    p = _write_config(
        tmp_path,
        override="tests.test_reranker_slot:FakeUserReranker",
        reranker_tag="",
    )
    runtime = build_runtime_from_config(str(p))
    scores = await runtime.reranker.rerank("q", ["a", "b", "c"])
    assert scores == [3, 2, 1]
    assert runtime.reranker.calls == 1


def test_bad_override_raises_typeerror_at_startup(tmp_path):
    p = _write_config(
        tmp_path,
        override="tests.test_reranker_slot:BadReranker",
        reranker_tag="",
    )
    with pytest.raises(TypeError, match="Reranker"):
        build_runtime_from_config(str(p))
