"""DefaultLLMClientFactoryEngine — caching + tag/purpose/embed/rerank
resolution + Protocol conformance.

The Engine wraps ``resolve_llm_for_tag``; tests verify the wrapper
produces the same observable behavior plus the new
core-purpose / embed / rerank conveniences. Cache semantics
(per-tag at-most-once construction) are tested via instance identity.
"""
from __future__ import annotations

import pytest

from krakey.engines.llm_factory.default import DefaultLLMClientFactoryEngine
from krakey.interfaces.engines import LLMClientFactoryEngine
from krakey.models.config import (
    Config,
    LLMParams,
    LLMSection,
    Provider,
    TagBinding,
)


def _make_cfg() -> Config:
    return Config(llm=LLMSection(
        providers={"P": Provider(
            type="openai_compatible",
            base_url="http://x", api_key="k",
        )},
        tags={
            "chat_tag":   TagBinding(provider="P/chat-model"),
            "embed_tag":  TagBinding(provider="P/embed-model"),
            "rerank_tag": TagBinding(provider="P/rerank-model"),
        },
        core_purposes={
            "self_thinking": "chat_tag",
            "compact":       "chat_tag",
        },
        embedding="embed_tag",
        reranker="rerank_tag",
    ))


# --------------------------------------------------------------------
# Protocol conformance
# --------------------------------------------------------------------


def test_default_factory_satisfies_protocol():
    """The default impl must structurally satisfy
    ``LLMClientFactoryEngine`` so EngineRegistry's isinstance check
    passes at startup."""
    f = DefaultLLMClientFactoryEngine(_make_cfg())
    assert isinstance(f, LLMClientFactoryEngine)


# --------------------------------------------------------------------
# client_for_tag — happy path + cache + None cases
# --------------------------------------------------------------------


def test_client_for_tag_resolves_known_tag():
    f = DefaultLLMClientFactoryEngine(_make_cfg())
    c = f.client_for_tag("chat_tag")
    assert c is not None
    # The default LLMClient exposes ``model`` as an attribute.
    assert c.model == "chat-model"


def test_client_for_tag_caches_per_tag():
    """Repeat calls for the same tag return the same instance — keeps
    connection state + future rate-limit accounting consistent."""
    f = DefaultLLMClientFactoryEngine(_make_cfg())
    c1 = f.client_for_tag("chat_tag")
    c2 = f.client_for_tag("chat_tag")
    assert c1 is c2


def test_client_for_tag_returns_none_for_missing_tag():
    f = DefaultLLMClientFactoryEngine(_make_cfg())
    assert f.client_for_tag("nonexistent_tag") is None


def test_client_for_tag_returns_none_for_empty_or_none():
    f = DefaultLLMClientFactoryEngine(_make_cfg())
    assert f.client_for_tag(None) is None
    assert f.client_for_tag("") is None


# --------------------------------------------------------------------
# client_for_core_purpose
# --------------------------------------------------------------------


def test_client_for_core_purpose_resolves_via_core_purposes_map():
    f = DefaultLLMClientFactoryEngine(_make_cfg())
    c = f.client_for_core_purpose("self_thinking")
    assert c is not None
    assert c.model == "chat-model"


def test_client_for_core_purpose_unknown_returns_none():
    f = DefaultLLMClientFactoryEngine(_make_cfg())
    assert f.client_for_core_purpose("not_a_purpose") is None


def test_two_purposes_pointing_at_same_tag_share_client():
    """``self_thinking`` and ``compact`` both bind to ``chat_tag`` in
    the fixture — they should share one cached client instance."""
    f = DefaultLLMClientFactoryEngine(_make_cfg())
    c1 = f.client_for_core_purpose("self_thinking")
    c2 = f.client_for_core_purpose("compact")
    assert c1 is c2


# --------------------------------------------------------------------
# embed_client / rerank_client
# --------------------------------------------------------------------


def test_embed_client_resolves_embedding_tag():
    f = DefaultLLMClientFactoryEngine(_make_cfg())
    c = f.embed_client()
    assert c is not None
    assert c.model == "embed-model"


def test_rerank_client_resolves_reranker_tag():
    f = DefaultLLMClientFactoryEngine(_make_cfg())
    c = f.rerank_client()
    assert c is not None
    assert c.model == "rerank-model"


def test_embed_client_returns_none_when_tag_unset():
    cfg = _make_cfg()
    cfg.llm.embedding = None
    f = DefaultLLMClientFactoryEngine(cfg)
    assert f.embed_client() is None


def test_rerank_client_returns_none_when_tag_unset():
    cfg = _make_cfg()
    cfg.llm.reranker = None
    f = DefaultLLMClientFactoryEngine(cfg)
    assert f.rerank_client() is None
