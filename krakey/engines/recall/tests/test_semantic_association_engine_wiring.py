"""Part C — IncrementalRecallEngine factory/config gating tests.

Tests the new keyword-only params `factory` and `config` on
`IncrementalRecallEngine` and whether `new_session()` produces sessions
with enricher=None vs enricher=<enricher instance> depending on config.

Contracts under test (implementation does NOT exist yet):
  - IncrementalRecallEngine(..., factory=None, config=None)  →  enricher=None
  - IncrementalRecallEngine(..., factory=None, config={})    →  enricher=None
  - config={'semantic_association_enabled': False}           →  enricher=None
  - config={'semantic_association_enabled': True}, factory=None  →  enricher=None
  - config={'semantic_association_enabled': True},
      factory whose client_for_core_purpose() returns None   →  enricher=None (no crash)
  - config={'semantic_association_enabled': True},
      factory whose client_for_core_purpose() returns a ChatLike →
      session has a non-None enricher (SemanticAssociationEnricher).
  - new_session() still returns a RecallSession regardless of enricher path.

Mirrors the fixture style from test_recall_engine.py and applies the
zero-plugin graceful-degradation invariant from tests/test_zero_plugin_runtime.py.

Techniques applied:
  - Positive / equivalence-partition  (4 tests)
  - Boundary value analysis           (3 tests)
  - State transitions                 (2 tests)
  - Negative / error-guessing         (4 tests)
"""
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


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

async def _no_embed(text: str) -> list[float]:
    return [0.0] * 8


class _FakeMemory:
    """Stub MemoryEngine — no SQLite needed for wiring tests."""

    async def vec_search(self, *args, **kwargs):
        return []

    async def fts_search(self, *args, **kwargs):
        return []

    async def get_neighbor_keywords(self, *args, **kwargs):
        return {}

    async def get_edges_among(self, *args, **kwargs):
        return []


class _FakeChatClient:
    """Minimal ChatLike duck-type."""

    async def chat(self, messages, **kwargs) -> str:
        return "phrase one"


class _FactoryReturningClient:
    """Stub LLM factory that returns a real ChatLike for any purpose."""

    def __init__(self, client):
        self._client = client

    def client_for_core_purpose(self, purpose: str):
        return self._client


class _FactoryReturningNone:
    """Stub LLM factory whose client_for_core_purpose always returns None."""

    def client_for_core_purpose(self, purpose: str):
        return None


def _make_cfg() -> Config:
    return Config(
        llm=LLMSection(
            providers={"P": Provider(
                type="openai_compatible",
                base_url="http://x",
                api_key="k",
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


def _make_engine(*, factory=None, config=None) -> IncrementalRecallEngine:
    return IncrementalRecallEngine(
        cfg=_make_cfg(),
        memory=_FakeMemory(),
        embedder=_no_embed,
        reranker=None,
        factory=factory,
        config=config,
    )


def _session_enricher(engine: IncrementalRecallEngine):
    """Return the enricher stored on the session returned by new_session().
    The enricher is expected at session._enricher; this helper isolates
    that attribute access so only this file needs to know the internal name."""
    session = engine.new_session()
    return getattr(session, "_enricher", None)


# ---------------------------------------------------------------------------
# Positive tests
# ---------------------------------------------------------------------------

class TestEngineWiringPositive:

    def test_factory_none_config_none_session_is_recall_session(self):
        """Default constructor (no factory, no config) still returns a valid
        RecallSession — zero-plugin invariant."""
        eng = _make_engine(factory=None, config=None)
        assert isinstance(eng.new_session(), RecallSession)

    def test_semantic_enabled_with_real_client_enricher_is_not_none(self):
        """With semantic_association_enabled=True and a factory that returns
        a real client, the session should carry a non-None enricher."""
        client = _FakeChatClient()
        factory = _FactoryReturningClient(client)
        eng = _make_engine(
            factory=factory,
            config={"semantic_association_enabled": True},
        )
        enricher = _session_enricher(eng)
        assert enricher is not None

    def test_semantic_enabled_false_enricher_is_none(self):
        """semantic_association_enabled=False → enricher=None regardless of factory."""
        client = _FakeChatClient()
        factory = _FactoryReturningClient(client)
        eng = _make_engine(
            factory=factory,
            config={"semantic_association_enabled": False},
        )
        enricher = _session_enricher(eng)
        assert enricher is None

    def test_engine_satisfies_recall_engine_protocol(self):
        """IncrementalRecallEngine with new params still satisfies the
        RecallEngine Protocol."""
        eng = _make_engine(
            factory=_FactoryReturningClient(_FakeChatClient()),
            config={"semantic_association_enabled": True},
        )
        assert isinstance(eng, RecallEngine)


# ---------------------------------------------------------------------------
# Boundary value analysis
# ---------------------------------------------------------------------------

class TestEngineWiringBVA:

    def test_empty_config_dict_enricher_is_none(self):
        """config={} (key absent) → feature disabled → enricher=None."""
        eng = _make_engine(
            factory=_FactoryReturningClient(_FakeChatClient()),
            config={},
        )
        enricher = _session_enricher(eng)
        assert enricher is None

    def test_factory_none_semantic_enabled_enricher_is_none(self):
        """factory=None but semantic_association_enabled=True → enricher=None.
        No crash — graceful degradation."""
        eng = _make_engine(
            factory=None,
            config={"semantic_association_enabled": True},
        )
        enricher = _session_enricher(eng)
        assert enricher is None

    def test_factory_returns_none_client_semantic_enabled_enricher_is_none(self):
        """factory.client_for_core_purpose() returns None → enricher=None.
        No AttributeError or crash — graceful degradation."""
        eng = _make_engine(
            factory=_FactoryReturningNone(),
            config={"semantic_association_enabled": True},
        )
        # Must not raise; enricher must be None
        enricher = _session_enricher(eng)
        assert enricher is None


# ---------------------------------------------------------------------------
# State transition tests
# ---------------------------------------------------------------------------

class TestEngineWiringStateTransitions:

    def test_new_session_each_call_returns_fresh_instance(self):
        """Each new_session() call returns a distinct instance — sessions
        are per-beat and must not be shared."""
        eng = _make_engine(
            factory=_FactoryReturningClient(_FakeChatClient()),
            config={"semantic_association_enabled": True},
        )
        s1 = eng.new_session()
        s2 = eng.new_session()
        assert s1 is not s2

    def test_enricher_config_fixed_at_construction_time(self):
        """The enricher presence is determined at engine construction time;
        all sessions from the same engine instance share the same enricher
        policy (both None or both non-None)."""
        client = _FakeChatClient()
        factory = _FactoryReturningClient(client)
        eng = _make_engine(
            factory=factory,
            config={"semantic_association_enabled": True},
        )
        e1 = _session_enricher(eng)
        e2 = _session_enricher(eng)
        # Both must have the same enricher presence
        assert (e1 is None) == (e2 is None)


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------

class TestEngineWiringNegative:

    def test_factory_none_does_not_raise(self):
        """factory=None must not raise at construction or new_session() time."""
        eng = _make_engine(factory=None, config={"semantic_association_enabled": True})
        session = eng.new_session()
        assert isinstance(session, RecallSession)

    def test_factory_returning_none_does_not_raise(self):
        """factory.client_for_core_purpose() → None must not raise."""
        eng = _make_engine(
            factory=_FactoryReturningNone(),
            config={"semantic_association_enabled": True},
        )
        session = eng.new_session()
        assert isinstance(session, RecallSession)

    def test_config_none_does_not_raise(self):
        """config=None must be handled gracefully (treated as disabled)."""
        eng = _make_engine(factory=_FactoryReturningClient(_FakeChatClient()),
                           config=None)
        session = eng.new_session()
        assert isinstance(session, RecallSession)

    def test_existing_no_factory_constructor_still_works(self):
        """The existing 4-arg constructor signature (cfg, memory, embedder,
        reranker) without factory/config must keep working unchanged —
        regression guard for callers that don't pass the new params."""
        eng = IncrementalRecallEngine(
            cfg=_make_cfg(),
            memory=_FakeMemory(),
            embedder=_no_embed,
            reranker=None,
            # factory and config intentionally omitted
        )
        session = eng.new_session()
        assert isinstance(session, RecallSession)
        assert _session_enricher(eng) is None
