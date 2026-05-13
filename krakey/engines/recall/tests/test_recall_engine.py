"""IncrementalRecallEngine — Protocol conformance + new_session
returns a fresh per-beat session each call. Lifted from the
recall plugin into a RecallEngine impl."""
from __future__ import annotations

import pytest

from krakey.engines.recall.default import IncrementalRecallEngine
from krakey.interfaces.engines import RecallEngine, RecallSession
from krakey.models.config import (
    Config,
    GraphMemorySection,
    LLMSection,
    Provider,
    TagBinding,
)


async def _no_embed(text):
    return [0.0] * 8


class _FakeMemory:
    """Stub MemoryEngine — IncrementalRecall only touches a few
    methods during construction (none) so we don't need a real one."""

    async def vec_search(self, *args, **kwargs):
        return []

    async def fts_search(self, *args, **kwargs):
        return []

    async def get_neighbor_keywords(self, *args, **kwargs):
        return {}

    async def get_edges_among(self, *args, **kwargs):
        return []


def _make_cfg() -> Config:
    return Config(
        llm=LLMSection(
            providers={"P": Provider(
                type="openai_compatible",
                base_url="http://x", api_key="k",
            )},
            tags={"t": TagBinding(provider="P/m")},
            core_purposes={"self_thinking": "t"},
        ),
        graph_memory=GraphMemorySection(
            recall_per_stimulus_k=10,
            neighbor_expand_depth=1,
            recall_screening_token_multiplier=2.0,
        ),
    )


def test_satisfies_recall_engine_protocol():
    eng = IncrementalRecallEngine(
        cfg=_make_cfg(), memory=_FakeMemory(),
        embedder=_no_embed, reranker=None,
    )
    assert isinstance(eng, RecallEngine)


def test_new_session_returns_recall_session():
    eng = IncrementalRecallEngine(
        cfg=_make_cfg(), memory=_FakeMemory(),
        embedder=_no_embed, reranker=None,
    )
    session = eng.new_session()
    assert isinstance(session, RecallSession)


def test_new_session_returns_fresh_instance():
    """Each call returns a NEW IncrementalRecall — sessions are
    per-beat, not shared. Identity comparison."""
    eng = IncrementalRecallEngine(
        cfg=_make_cfg(), memory=_FakeMemory(),
        embedder=_no_embed, reranker=None,
    )
    s1 = eng.new_session()
    s2 = eng.new_session()
    assert s1 is not s2


def test_session_has_processed_stimuli_attribute():
    """RecallSession Protocol asks for ``processed_stimuli`` —
    IncrementalRecall must expose it as an empty list at session start."""
    eng = IncrementalRecallEngine(
        cfg=_make_cfg(), memory=_FakeMemory(),
        embedder=_no_embed, reranker=None,
    )
    session = eng.new_session()
    assert hasattr(session, "processed_stimuli")
    assert session.processed_stimuli == []
