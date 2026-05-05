"""Deterministic SHA-256 based embedder — dev-mode placeholder.

Produces stable but semantically-meaningless vectors. Useful for local
development and CI; **do not ship in production** — recall quality
will be terrible.

Drop this module anywhere on PYTHONPATH and reference it via
``core_implementations.embedder = "hash_embedder:HashEmbedder"`` in
your config.yaml. See ``examples/custom_embedder/README.md`` and
``docs/extending-core.md`` for the full walkthrough.
"""
from __future__ import annotations

import hashlib


class HashEmbedder:
    """Hashes text into a deterministic float vector.

    Constructor takes no args (per the embedder slot's contract). All
    state is per-instance and small — the digest call is sub-millisecond
    so there's no point caching across calls.
    """

    DIM = 1024

    async def __call__(self, text: str) -> list[float]:
        # SHA-256 → 32 bytes. Map each byte to float in roughly [-1, 1]
        # then repeat to reach the target dimension.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        floats = [(b - 127.5) / 127.5 for b in digest]
        repeats = self.DIM // len(floats)
        return (floats * repeats)[: self.DIM]
