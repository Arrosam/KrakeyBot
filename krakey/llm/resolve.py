"""Tag → LLMClient resolution + the structural Protocols runtime uses
to talk about LLMs and embedders generically.

Two concerns in one small module — both about how the rest of the
system addresses an LLM without knowing the concrete implementation:

  * ``ChatLike`` / ``AsyncEmbedder`` — Protocols declaring the minimal
    shape the runtime depends on. Built-in clients (LLMClient,
    embedding clients) and test doubles (ScriptedLLM, NullEmbedder)
    both satisfy them structurally without inheritance.

  * ``resolve_llm_for_tag(cfg, tag_name, cache)`` — given a tag name
    and the central config, return a cached or freshly-built
    ``LLMClient``. Shared between the core-purpose loader (in
    ``krakey/main.py``) and per-plugin LLM resolution (via
    ``PluginContext.get_llm_for_tag``) so two purposes pointing at
    the same tag share one client instance — keeps connection state
    + future rate-limit accounting consistent.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from krakey.llm.client import LLMClient
    from krakey.models.config import Config


@runtime_checkable
class ChatLike(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...


@runtime_checkable
class AsyncEmbedder(Protocol):
    """Async-callable returning an embedding vector for one text.

    ``@runtime_checkable`` so ``ServiceResolver`` can isinstance-check
    user-supplied embedder slots at startup. Caveat: Python's
    runtime-checkable Protocol with ``__call__`` only verifies the
    method exists — it can't check the signature, so any callable
    technically passes. Documentation discipline beats type-system
    enforcement here.
    """
    async def __call__(self, text: str) -> list[float]: ...


def resolve_llm_for_tag(
    cfg: "Config", tag_name: str | None,
    cache: dict[str, "LLMClient"],
) -> "ChatLike | None":
    """Build (or fetch from cache) the LLM client for a tag name.

    Returns ``None`` for: empty ``tag_name``, missing tag in
    ``cfg.llm.tags``, malformed provider field, or provider name not
    in ``cfg.llm.providers``. Each failure mode logs a single stderr
    warning so the user can see what to fix; callers continue without
    that LLM (strictly additive plugin model — bad config doesn't
    crash startup).

    Construction goes through ``ServiceResolver`` so the
    ``llm_client_factory`` core slot can replace the built-in
    ``LLMClient`` with a user-supplied implementation. Per-tag caching
    is preserved: each unique tag triggers at most one factory call,
    same as before. Caveat for users who override the factory: the
    resolved client is also used for embedding/reranker tags, so a
    user class that lacks ``embed()`` / ``rerank()`` will work for
    chat tags but break those — keep the embedder / reranker slots
    in mind, or implement those methods on the same class.
    """
    from krakey.engines.registry import EngineRegistry
    from krakey.llm.client import LLMClient

    if not tag_name:
        return None
    cached = cache.get(tag_name)
    if cached is not None:
        return cached
    tag = cfg.llm.tags.get(tag_name)
    if tag is None:
        print(f"warning: tag {tag_name!r} not defined in llm.tags",
              file=sys.stderr)
        return None
    try:
        provider_name, model_name = tag.split_provider()
    except ValueError as e:
        print(f"warning: tag {tag_name!r} has bad provider field: {e}",
              file=sys.stderr)
        return None
    provider = cfg.llm.providers.get(provider_name)
    if provider is None:
        print(
            f"warning: tag {tag_name!r} references unknown provider "
            f"{provider_name!r}", file=sys.stderr,
        )
        return None
    # Transient registry: EngineRegistry is a thin wrapper around
    # cfg.core_implementations + importlib, so per-cache-miss
    # construction is cheap. The legacy ``llm_client_factory`` slot
    # still substitutes the LLMClient *class* (its old per-tag-class
    # semantics); the higher-level ``llm_factory`` slot substitutes
    # the entire factory Engine and is wired separately by the
    # composition root.
    registry = EngineRegistry(cfg)
    client = registry.resolve(
        "llm_client_factory",
        default_path="krakey.llm.client:LLMClient",
        expected_protocol=ChatLike,
        provider=provider,
        model=model_name,
        params=tag.params,
    )
    cache[tag_name] = client
    return client
