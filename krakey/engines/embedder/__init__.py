"""``embedder`` Engine — text → vector.

Default impl ``TagBoundEmbedderEngine`` walks
``LLMClientFactoryEngine.embed_client()`` to reach the configured
embedding tag's client and calls its ``embed(text)`` method. The
``EmbedderEngine`` Protocol lives at
``krakey.interfaces.engines.embedder``.
"""
from krakey.engines.catalog import EngineImpl
from krakey.engines.embedder.default import TagBoundEmbedderEngine

BUILTIN_ENGINES = {
    "tag_bound": EngineImpl(
        cls=TagBoundEmbedderEngine,
        description=(
            "Forwards to the LLMClientFactory's embed_client (tag "
            "from llm.embedding)."
        ),
    ),
}

DEFAULT_ENGINE = "tag_bound"

__all__ = ["BUILTIN_ENGINES", "DEFAULT_ENGINE", "TagBoundEmbedderEngine"]
