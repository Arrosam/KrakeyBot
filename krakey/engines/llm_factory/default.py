"""``DefaultLLMClientFactoryEngine`` — built-in LLM factory Engine.

Wraps ``krakey.llm.resolve.resolve_llm_for_tag`` (which handles the
actual tag → provider → ``LLMClient`` instantiation, plus the
``llm_client_factory`` slot for per-tag class substitution). This
Engine is the higher layer that hides ``cfg.llm`` from upstream
callers and owns the per-tag client cache. Users replace the whole
Engine via ``cfg.core_implementations.llm_factory`` to swap the
resolution mechanism wholesale (offline-only factory for tests, an
HTTP-routing factory for multi-tenant deployment, a mock for replays).

Caching contract: each unique tag triggers at most one underlying
client construction; subsequent ``client_for_tag`` calls return the
cached instance. ``client_for_core_purpose`` / ``embed_client`` /
``rerank_client`` are convenience wrappers that look up the configured
tag name and call ``client_for_tag``.

The cache itself is private — every consumer (plugins via
``PluginContext.get_llm_for_tag``, other engines, runtime composition)
goes through the Protocol's ``client_for_tag`` method. This keeps the
Protocol the only surface third-party impls must satisfy.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from krakey.llm.resolve import resolve_llm_for_tag

if TYPE_CHECKING:
    from krakey.llm.resolve import ChatLike
    from krakey.models.config import Config


class DefaultLLMClientFactoryEngine:
    """LLM factory Engine — long-lived, owns a private per-tag client cache."""

    def __init__(self, cfg: "Config"):
        self._cfg = cfg
        # Mutable dict; ``resolve_llm_for_tag`` writes the resolved
        # client back into it on cache miss, reads on hit.
        self._cache: dict[str, "ChatLike"] = {}

    # ---- Protocol surface -----------------------------------------------

    def client_for_tag(self, tag_name: str | None) -> "ChatLike | None":
        return resolve_llm_for_tag(self._cfg, tag_name, self._cache)

    def client_for_core_purpose(self, purpose: str) -> "ChatLike | None":
        tag_name = self._cfg.llm.core_purposes.get(purpose)
        return self.client_for_tag(tag_name)

    def embed_client(self) -> "ChatLike | None":
        return self.client_for_tag(self._cfg.llm.embedding)

    def rerank_client(self) -> "ChatLike | None":
        return self.client_for_tag(self._cfg.llm.reranker)
