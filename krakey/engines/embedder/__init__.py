"""``embedder`` Engine — text → vector.

The slot's catalog of impls lives in ``meta.yaml`` next to this file.
The EmbedderEngine Protocol lives at
``krakey.interfaces.engines.embedder``.
"""
from krakey.engines.embedder.default import TagBoundEmbedderEngine

__all__ = ["TagBoundEmbedderEngine"]
