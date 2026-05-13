"""Tag → LLMClient resolution helper for the LLM factory engine.

Implementation detail of ``DefaultLLMClientFactoryEngine``: given a
tag name + the central config, return a cached or freshly-built
``LLMClient``. The factory's ``client_for_tag`` Protocol method
delegates here. Per-tag caching means two consumers pointing at the
same tag share one client instance — keeps connection state and
future rate-limit accounting consistent.

The duck Protocols (``ChatLike``, ``AsyncEmbedder``) used to live
alongside this code; they're now at ``krakey.interfaces.duck`` since
they're cross-cutting typing primitives rather than this Engine's
private concern.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from krakey.interfaces.duck import ChatLike

if TYPE_CHECKING:
    from krakey.engines.llm_client_factory._client import LLMClient
    from krakey.models.config import Config


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
    from krakey.engine_system.registry import EngineRegistry

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
        expected_protocol=ChatLike,
        provider=provider,
        model=model_name,
        params=tag.params,
    )
    cache[tag_name] = client
    return client
