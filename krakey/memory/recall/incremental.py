"""Embedder Protocol used across the memory subsystem.

The recall driver itself lives at ``krakey.engines.recall.incremental``.
Pure scoring helpers (``rerank``, ``scripted_score``, ``ScoringWeights``,
``Reranker``) live in ``krakey.memory.recall.scoring``. This module
exposes only the duck-typed embedder callable Protocol the GM/KB/Sleep
modules consume.
"""
from __future__ import annotations

from typing import Protocol


class AsyncEmbedder(Protocol):
    async def __call__(self, text: str) -> list[float]: ...
