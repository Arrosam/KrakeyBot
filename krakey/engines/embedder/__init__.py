"""``embedder`` Engine — text → vector.

Default impl ``TagBoundEmbedderEngine`` walks
``LLMClientFactoryEngine.embed_client()`` to reach the configured
embedding tag's client and calls its ``embed(text)`` method. The
``EmbedderEngine`` Protocol lives at
``krakey.interfaces.engines.embedder``.
"""
from krakey.engines.embedder.default import TagBoundEmbedderEngine

__all__ = ["TagBoundEmbedderEngine"]
