"""``reranker`` Engine — score-based reordering for recall + KB dedup.

The slot's catalog of impls lives in ``meta.yaml`` next to this file.
The RerankerEngine Protocol lives at
``krakey.interfaces.engines.reranker``.
"""
from krakey.engines.reranker.default import DefaultRerankerEngine

__all__ = ["DefaultRerankerEngine"]
