"""recall-result contract — semantic-association purpose-name edge tests.

SPEC UNDER TEST (development unit only — not a full contract suite):
  The `IncrementalRecallEngine` resolves a "core purpose" name and calls
  `factory.client_for_core_purpose(purpose)` to obtain the enricher LLM.

  Rule 1 — DEFAULT purpose:
    When `semantic_association_enabled` is True AND `semantic_association_purpose`
    is NOT set in the engine config, the purpose passed to
    `client_for_core_purpose` MUST be "compact".
    (Previously it was "recall_enrichment" — this unit changes that default.)

  Rule 2 — EXPLICIT purpose:
    When `semantic_association_purpose` IS explicitly set in the engine config,
    that exact string value MUST be forwarded verbatim to
    `client_for_core_purpose`.

  Rule 3 — DISABLED feature:
    When `semantic_association_enabled` is False (or absent),
    `client_for_core_purpose` must NEVER be called, regardless of whether
    `semantic_association_purpose` is set.

  Rule 4 — NULL FACTORY:
    When `factory` is None and the feature is enabled, construction and
    `new_session()` must complete without raising — graceful degradation.

  Rule 5 — EMPTY / WHITESPACE-ONLY EXPLICIT PURPOSE (clarified spec):
    An explicitly-set `semantic_association_purpose` that is empty ("") or
    contains only whitespace ("   ") after stripping is treated as UNSET →
    resolved to the default "compact".  Only a non-empty (post-strip) value
    is forwarded verbatim.  A value with surrounding whitespace but a non-empty
    core is stripped and the core is forwarded (e.g. "  classifier  " → "classifier").

Contracts under test (LOCKED, unchanged by this unit):
  - engine-protocols: RecallEngine.new_session()
  - recall-result: RecallSession lifecycle

The factory's `client_for_core_purpose` call is the ONLY observable
output asserted here — all other behavior is covered by sibling tests.

Techniques applied per method:
  new_session():
    Positive / equivalence: 4 tests
    Boundary value analysis: 6 tests  (+3 from spec-clarification amendment)
    State transitions: 2 tests
    Negative / error cases: 3 tests
  Total: 15 tests
"""
from __future__ import annotations

import pytest

from krakey.engines.recall.default import IncrementalRecallEngine
from krakey.models.config import (
    Config,
    GraphMemorySection,
    LLMSection,
    Provider,
    TagBinding,
)


# ---------------------------------------------------------------------------
# Shared test doubles
# ---------------------------------------------------------------------------

async def _no_embed(text: str) -> list[float]:
    """Minimal async embedder — returns a fixed-length zero vector."""
    return [0.0] * 8


class _FakeMemory:
    """Stub MemoryEngine — only the vec/fts/neighbor surface used by
    IncrementalRecall.add_stimuli / finalize.  No SQLite needed."""

    async def vec_search(self, *args, **kwargs):
        return []

    async def fts_search(self, *args, **kwargs):
        return []

    async def get_neighbor_keywords(self, *args, **kwargs):
        return {}

    async def get_edges_among(self, *args, **kwargs):
        return []


class _FakeChatClient:
    """Minimal ChatLike duck-type.  Returns a single phrase so
    SemanticAssociationEnricher can parse it without crashing."""

    async def chat(self, messages, **kwargs) -> str:
        return "test phrase"


class _SpyFactory:
    """LLMClientFactoryEngine spy — records every `client_for_core_purpose`
    call and optionally returns a configured client or None."""

    def __init__(self, *, return_client=True):
        self.calls: list[str] = []
        self._client = _FakeChatClient() if return_client else None

    def client_for_core_purpose(self, purpose: str):
        self.calls.append(purpose)
        return self._client


def _make_cfg() -> Config:
    """Minimal Config that satisfies IncrementalRecallEngine construction."""
    return Config(
        llm=LLMSection(
            providers={"P": Provider(
                type="openai_compatible",
                base_url="http://test-host",
                api_key="test-key",
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


# ---------------------------------------------------------------------------
# Positive / equivalence tests
# ---------------------------------------------------------------------------

class TestPurposeNamePositive:
    """Rule 1 & 2 — purpose value delivered to client_for_core_purpose."""

    def test_default_purpose_is_compact_when_key_absent(self):
        """Enabled + no 'semantic_association_purpose' key → purpose='compact'.

        This is the primary regression guard for the default-value change.
        The config dict has semantic_association_enabled=True but no purpose
        key; the engine MUST call client_for_core_purpose('compact') and NOT
        'recall_enrichment' or any other string."""
        spy = _SpyFactory(return_client=True)
        eng = _make_engine(
            factory=spy,
            config={"semantic_association_enabled": True},
        )
        eng.new_session()
        assert spy.calls == ["compact"], (
            f"Expected purpose 'compact', got {spy.calls!r}.  "
            "This confirms the default was changed from 'recall_enrichment'."
        )

    def test_explicit_purpose_overrides_default(self):
        """Enabled + explicit 'semantic_association_purpose' → that value used verbatim."""
        spy = _SpyFactory(return_client=True)
        eng = _make_engine(
            factory=spy,
            config={
                "semantic_association_enabled": True,
                "semantic_association_purpose": "classifier",
            },
        )
        eng.new_session()
        assert spy.calls == ["classifier"], (
            f"Expected purpose 'classifier', got {spy.calls!r}"
        )

    def test_explicit_purpose_arbitrary_string_forwarded_verbatim(self):
        """Any non-empty explicit purpose is forwarded without mutation."""
        spy = _SpyFactory(return_client=True)
        purpose = "my_custom_enricher_v2"
        eng = _make_engine(
            factory=spy,
            config={
                "semantic_association_enabled": True,
                "semantic_association_purpose": purpose,
            },
        )
        eng.new_session()
        assert spy.calls == [purpose], (
            f"Expected purpose {purpose!r}, got {spy.calls!r}"
        )

    def test_disabled_feature_never_calls_client_for_core_purpose(self):
        """Rule 3: disabled feature → zero calls regardless of purpose key."""
        spy = _SpyFactory(return_client=True)
        eng = _make_engine(
            factory=spy,
            config={
                "semantic_association_enabled": False,
                "semantic_association_purpose": "compact",
            },
        )
        eng.new_session()
        assert spy.calls == [], (
            "client_for_core_purpose must not be called when feature is disabled"
        )


# ---------------------------------------------------------------------------
# Boundary value analysis
# ---------------------------------------------------------------------------

class TestPurposeNameBVA:
    """Edge cases for the config dict and purpose string boundaries."""

    def test_purpose_key_explicitly_set_to_compact_string(self):
        """Explicit purpose='compact' (same as default) is forwarded — not
        treated as 'absent' and not double-resolved.  One call expected."""
        spy = _SpyFactory(return_client=True)
        eng = _make_engine(
            factory=spy,
            config={
                "semantic_association_enabled": True,
                "semantic_association_purpose": "compact",
            },
        )
        eng.new_session()
        assert spy.calls == ["compact"]

    def test_purpose_key_explicitly_set_to_old_default(self):
        """Explicit purpose='recall_enrichment' (old default) must still be
        forwarded verbatim — explicit always wins, even for the old value."""
        spy = _SpyFactory(return_client=True)
        eng = _make_engine(
            factory=spy,
            config={
                "semantic_association_enabled": True,
                "semantic_association_purpose": "recall_enrichment",
            },
        )
        eng.new_session()
        assert spy.calls == ["recall_enrichment"], (
            "Explicit 'recall_enrichment' must be forwarded, not silently "
            "replaced by the new default 'compact'."
        )

    def test_feature_absent_from_config_dict_never_calls_factory(self):
        """config={} (semantic_association_enabled key entirely absent) → disabled.
        client_for_core_purpose must NOT be called — treat absent as False."""
        spy = _SpyFactory(return_client=True)
        eng = _make_engine(factory=spy, config={})
        eng.new_session()
        assert spy.calls == [], (
            "Absent semantic_association_enabled must be treated as disabled "
            "— factory must not be called."
        )

    # --- Rule 5: empty / whitespace-only purpose (spec-clarification amendment) ---

    def test_empty_string_purpose_resolves_to_compact(self):
        """BVA / negative: explicit semantic_association_purpose='' (empty string)
        must be treated as UNSET → resolved to default 'compact'.

        An empty string is the minimum-boundary value for the purpose field.
        The empty string carries no information, so the engine must fall back
        to the same default it would use if the key were absent entirely."""
        spy = _SpyFactory(return_client=True)
        eng = _make_engine(
            factory=spy,
            config={
                "semantic_association_enabled": True,
                "semantic_association_purpose": "",
            },
        )
        eng.new_session()
        assert spy.calls == ["compact"], (
            f"Expected purpose 'compact' for empty-string explicit value, "
            f"got {spy.calls!r}.  Empty string must be treated as unset."
        )

    def test_whitespace_only_purpose_resolves_to_compact(self):
        """BVA / negative: explicit semantic_association_purpose='   ' (spaces only)
        must be treated as UNSET → resolved to default 'compact'.

        Whitespace-only is the second degenerate boundary: after strip() the
        result is '', so the same fallback rule as the empty-string case applies.
        The engine must NOT forward the raw whitespace string."""
        spy = _SpyFactory(return_client=True)
        eng = _make_engine(
            factory=spy,
            config={
                "semantic_association_enabled": True,
                "semantic_association_purpose": "   ",
            },
        )
        eng.new_session()
        assert spy.calls == ["compact"], (
            f"Expected purpose 'compact' for whitespace-only explicit value, "
            f"got {spy.calls!r}.  Whitespace-only must be treated as unset."
        )

    def test_surrounding_whitespace_purpose_is_stripped_to_core(self):
        """BVA (regression guard): explicit semantic_association_purpose with
        surrounding whitespace and a non-empty core must be stripped.

        '  classifier  '.strip() == 'classifier' — the resolved value forwarded
        to client_for_core_purpose must have no leading or trailing whitespace
        and must equal 'classifier' exactly.  This ensures the strip logic does
        not over-trim (collapse to default) when the core is non-empty."""
        spy = _SpyFactory(return_client=True)
        eng = _make_engine(
            factory=spy,
            config={
                "semantic_association_enabled": True,
                "semantic_association_purpose": "  classifier  ",
            },
        )
        eng.new_session()
        assert len(spy.calls) == 1, (
            f"Expected exactly one client_for_core_purpose call, got {spy.calls!r}"
        )
        resolved = spy.calls[0]
        assert resolved == "classifier", (
            f"Expected stripped purpose 'classifier', got {resolved!r}.  "
            "Surrounding whitespace must be stripped, but the non-empty core "
            "must be forwarded verbatim."
        )
        # Extra guard: verify no leading/trailing whitespace survives
        assert resolved == resolved.strip(), (
            f"Resolved purpose {resolved!r} still has leading/trailing whitespace."
        )


# ---------------------------------------------------------------------------
# State transition tests
# ---------------------------------------------------------------------------

class TestPurposeNameStateTransitions:
    """Purpose resolution must be consistent across multiple new_session() calls."""

    def test_default_purpose_consistent_across_two_sessions(self):
        """Each new_session() call must use the same default purpose='compact'.
        The engine must not drift to a different purpose on the second call."""
        spy = _SpyFactory(return_client=True)
        eng = _make_engine(
            factory=spy,
            config={"semantic_association_enabled": True},
        )
        eng.new_session()
        eng.new_session()
        assert spy.calls == ["compact", "compact"], (
            f"Expected ['compact', 'compact'], got {spy.calls!r}"
        )

    def test_explicit_purpose_consistent_across_two_sessions(self):
        """Explicit purpose must be forwarded the same way on each session."""
        spy = _SpyFactory(return_client=True)
        eng = _make_engine(
            factory=spy,
            config={
                "semantic_association_enabled": True,
                "semantic_association_purpose": "my_purpose",
            },
        )
        eng.new_session()
        eng.new_session()
        assert spy.calls == ["my_purpose", "my_purpose"], (
            f"Expected ['my_purpose', 'my_purpose'], got {spy.calls!r}"
        )


# ---------------------------------------------------------------------------
# Negative / error cases
# ---------------------------------------------------------------------------

class TestPurposeNameNegative:
    """Rule 4 and robustness: graceful degradation when factory is None or
    client_for_core_purpose returns None."""

    def test_factory_none_does_not_raise_when_enabled(self):
        """Rule 4: factory=None + enabled=True must not raise on construction
        or new_session() — zero calls expected (no factory to call)."""
        eng = _make_engine(
            factory=None,
            config={"semantic_association_enabled": True},
        )
        # Must complete without raising
        session = eng.new_session()
        assert session is not None

    def test_factory_returning_none_client_does_not_raise(self):
        """Factory returns None from client_for_core_purpose → engine must
        degrade gracefully (no enricher) without raising AttributeError or
        similar.  Purpose is still passed — factory is called."""
        spy = _SpyFactory(return_client=False)
        eng = _make_engine(
            factory=spy,
            config={"semantic_association_enabled": True},
        )
        # Must complete without raising
        session = eng.new_session()
        assert session is not None
        # The purpose was still resolved correctly even though client=None
        assert spy.calls == ["compact"]

    def test_config_none_never_calls_factory(self):
        """config=None (engine config not supplied) → feature disabled.
        client_for_core_purpose must NOT be called — no AttributeError."""
        spy = _SpyFactory(return_client=True)
        eng = _make_engine(factory=spy, config=None)
        eng.new_session()
        assert spy.calls == [], (
            "config=None must disable the feature — factory must not be called"
        )
