"""LLM client factory slot — verify resolve_llm_for_tag respects the override.

Default path: resolve_llm_for_tag returns an LLMClient.
Override path: the user-supplied class is returned instead.
Caching contract: each unique tag triggers at most one factory call,
preserved unchanged from the pre-slot behavior.
"""
from __future__ import annotations

import pytest

from krakey.engines.llm_client_factory._client import LLMClient
from krakey.engines.llm_factory._resolve import resolve_llm_for_tag
from krakey.interfaces.duck import ChatLike
from krakey.models.config import (
    Config,
    LLMParams,
    LLMSection,
    Provider,
    TagBinding,
)
from krakey.models.config.core_impls import CoreImplementations


# Module-level fakes so importlib can resolve via dotted path.

class FakeUserClient:
    """Minimal ChatLike — accepts the slot's kwargs and tracks calls."""

    def __init__(self, *, provider: Provider, model: str,
                  params: LLMParams | None = None):
        self.provider = provider
        self.model = model
        self.params = params
        self.chat_calls: list[list] = []

    async def chat(self, messages, **kwargs) -> str:
        self.chat_calls.append(messages)
        return f"fake reply to {len(messages)} msgs"


class BadUserClient:
    """No chat() method — should fail Protocol validation at startup."""

    def __init__(self, *, provider, model, params=None):
        self.provider = provider
        self.model = model

    def shrug(self) -> str:
        return "no chat method"


class WrongKwargsClient:
    """Doesn't accept the slot's kwargs — should fail with annotated error."""

    def __init__(self, *, totally_unrelated_kwarg: str):
        ...

    async def chat(self, messages, **kwargs) -> str:
        return "ok"


def _make_config(*, override: str = "") -> Config:
    return Config(
        llm=LLMSection(
            providers={"P": Provider(
                type="openai_compatible",
                base_url="http://x", api_key="k",
            )},
            tags={"t": TagBinding(provider="P/m", params=LLMParams())},
            core_purposes={"self_thinking": "t"},
        ),
        core_implementations=CoreImplementations(llm_client_factory=override),
    )


# ---- happy paths ---------------------------------------------------


def test_no_override_returns_llmclient():
    """Empty slot → resolver builds a default LLMClient."""
    cfg = _make_config(override="")
    cache: dict = {}
    client = resolve_llm_for_tag(cfg, "t", cache)
    assert isinstance(client, LLMClient)
    assert client.model == "m"


def test_override_returns_user_class():
    """`core_implementations.llm_client_factory = ...` → resolver returns
    an instance of the user class."""
    cfg = _make_config(
        override="tests.test_llm_client_factory_slot:FakeUserClient",
    )
    cache: dict = {}
    client = resolve_llm_for_tag(cfg, "t", cache)
    assert isinstance(client, FakeUserClient)
    assert client.model == "m"
    assert client.provider.base_url == "http://x"


def test_caching_preserved_across_resolve_calls():
    """Same tag → same client instance (caching contract unchanged)."""
    cfg = _make_config(
        override="tests.test_llm_client_factory_slot:FakeUserClient",
    )
    cache: dict = {}
    a = resolve_llm_for_tag(cfg, "t", cache)
    b = resolve_llm_for_tag(cfg, "t", cache)
    assert a is b
    assert "t" in cache


async def test_override_actually_chats():
    """Sanity: the user class's chat() really gets invoked."""
    cfg = _make_config(
        override="tests.test_llm_client_factory_slot:FakeUserClient",
    )
    cache: dict = {}
    client = resolve_llm_for_tag(cfg, "t", cache)
    reply = await client.chat([{"role": "user", "content": "hi"}])
    assert "fake reply" in reply
    assert client.chat_calls == [[{"role": "user", "content": "hi"}]]


# ---- error paths --------------------------------------------------


def test_bad_override_fails_protocol_check():
    """User class missing chat() → loud TypeError at first resolve."""
    cfg = _make_config(
        override="tests.test_llm_client_factory_slot:BadUserClient",
    )
    with pytest.raises(TypeError, match="ChatLike"):
        resolve_llm_for_tag(cfg, "t", {})


def test_override_with_wrong_kwargs_fails_loud():
    """User class that doesn't accept (provider, model, params) → annotated TypeError."""
    cfg = _make_config(
        override="tests.test_llm_client_factory_slot:WrongKwargsClient",
    )
    with pytest.raises(TypeError, match="kwargs"):
        resolve_llm_for_tag(cfg, "t", {})


# ---- existing fail-soft semantics still hold ----------------------


def test_unknown_tag_still_returns_none(capsys):
    """Tag not in llm.tags → still returns None + warns (slot mechanism
    sits AFTER tag-resolution failures)."""
    cfg = _make_config()
    out = resolve_llm_for_tag(cfg, "nonexistent_tag", {})
    assert out is None
    err = capsys.readouterr().err
    assert "not defined in llm.tags" in err


def test_empty_tag_name_returns_none():
    """Empty tag name short-circuits before slot resolution."""
    cfg = _make_config()
    assert resolve_llm_for_tag(cfg, "", {}) is None
    assert resolve_llm_for_tag(cfg, None, {}) is None
