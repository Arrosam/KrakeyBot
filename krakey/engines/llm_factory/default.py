"""``DefaultLLMClientFactoryEngine`` — built-in LLM factory Engine.

Wraps the existing ``resolve_llm_for_tag`` util from
``krakey.llm.resolve``. Responsibility split during the migration:

  * ``resolve_llm_for_tag`` continues to do the actual tag → provider
    → ``LLMClient`` instantiation, including the
    ``llm_client_factory`` slot that lets a user substitute the
    ``LLMClient`` *class* per tag (old behavior, still active).
  * This Engine is the higher layer that hides ``cfg.llm`` from
    upstream callers and owns the per-tag client cache. The user can
    replace this whole Engine via ``cfg.core_implementations.llm_factory``
    to swap the resolution mechanism wholesale (e.g. an offline-only
    factory for tests, an HTTP-routing factory for multi-tenant
    deployment, a mock for replays).

Caching contract: each unique tag triggers at most one underlying
client construction; subsequent ``client_for_tag`` calls return the
cached instance. ``client_for_core_purpose`` / ``embed_client`` /
``rerank_client`` are convenience wrappers — they look up the
configured tag name and then call ``client_for_tag``.

The cache is exposed as a property (``client_cache``) during the
migration window so the composition root can mirror it onto
``RuntimeDeps.llm_clients_by_tag``. Plugin code currently reads from
``deps.llm_clients_by_tag`` via ``PluginContext.get_llm_for_tag``;
once plugins migrate to use the Engine directly through
``ctx.services``, that mirror goes away and the cache becomes a
true private internal.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from krakey.llm.resolve import resolve_llm_for_tag

if TYPE_CHECKING:
    from krakey.llm.resolve import ChatLike
    from krakey.models.config import Config


class DefaultLLMClientFactoryEngine:
    """LLM factory Engine — long-lived, owns the per-tag client cache."""

    def __init__(self, cfg: "Config"):
        self._cfg = cfg
        # Mutable dict; ``resolve_llm_for_tag`` writes the resolved
        # client back into it on cache miss, reads on hit.
        self._cache: dict[str, "ChatLike"] = {}

    @property
    def client_cache(self) -> dict[str, "ChatLike"]:
        """Internal cache dict, exposed for the migration window so
        the composition root can pass it through to
        ``RuntimeDeps.llm_clients_by_tag`` and keep plugin code that
        still reads from there pointed at the same instances. Not a
        long-term API — slated for removal once plugins move to
        ``ctx.services["llm_factory"]`` access."""
        return self._cache

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
