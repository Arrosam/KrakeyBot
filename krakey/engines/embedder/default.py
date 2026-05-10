"""``TagBoundEmbedderEngine`` — default Embedder Engine.

Stateless wrapper that pulls the embedding tag's client from
``LLMClientFactoryEngine.embed_client()`` on each call and invokes
``client.embed(text)``. The factory caches the client per-tag, so
"on each call" is cheap — one dict lookup + one provider call.

Failure mode: when no embedding tag is bound (the user skipped
embedding configuration in onboarding / the dashboard),
``factory.embed_client()`` returns ``None`` and the Engine raises
``RuntimeError`` at call time. This is the established lazy-fail
pattern — recall + KB indexing degrade gracefully (callers handle
embed exceptions); the heartbeat itself doesn't crash on a missing
embedder unless someone tries to embed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from krakey.interfaces.engines.llm_factory import LLMClientFactoryEngine


class TagBoundEmbedderEngine:
    """Embedder that resolves to ``cfg.llm.embedding`` tag via the
    factory Engine. The Engine itself owns no client reference — the
    factory's per-tag cache handles reuse."""

    def __init__(self, *, factory: "LLMClientFactoryEngine"):
        self._factory = factory

    async def __call__(self, text: str) -> list[float]:
        client = self._factory.embed_client()
        if client is None:
            raise RuntimeError(
                "no embedding tag bound — set llm.embedding to a tag "
                "name in config.yaml (or use the dashboard's LLM "
                "section)"
            )
        return await client.embed(text)
