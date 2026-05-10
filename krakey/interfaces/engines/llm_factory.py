"""``LLMClientFactoryEngine`` — the only Engine that touches ``cfg.llm``.

Hides the tag → provider → model → client resolution from every other
Engine. Consumers (memory writers, decision engines, embedder,
reranker, hypothalamus translator, etc.) ask the factory for a client
by tag name or by core-purpose name and never see the providers or
their API keys.

Replaces the previous ``resolve_llm_for_tag`` free function plus the
ad-hoc client-cache passing on ``RuntimeDeps``. The Engine owns the
cache as a private implementation detail — one client instance per
tag, shared across all consumers via ``client_for_tag``. Third-party
factory impls only need to satisfy the methods declared below.

API-key isolation: the factory keeps providers + keys in its own
state; clients it returns satisfy ``ChatLike`` (defined in
``krakey/llm/resolve.py``) and expose only ``chat``/``embed``/
``rerank`` methods. Plugins and other Engines never see API keys.

A user replacing this Engine controls how clients are constructed —
swap in a mock factory for tests, an HTTP-routing factory for a
multi-tenant deployment, a local-only factory for offline mode, etc.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from krakey.llm.resolve import ChatLike


@runtime_checkable
class LLMClientFactoryEngine(Protocol):
    """The only Engine permitted to read ``cfg.llm``. Resolves tag
    names and core-purpose names into chat/embed/rerank clients.

    All four ``*_client`` methods return ``None`` when the requested
    binding is missing or malformed; callers MUST handle ``None``
    (typical fallback: skip the operation, surface a stimulus, or
    raise depending on the call site's contract — never silently
    pretend success).
    """

    def client_for_tag(self, tag_name: str | None) -> "ChatLike | None":
        """Resolve a tag name to a chat-capable client. ``None`` for
        empty/missing/malformed tags. Cached by tag name internally so
        repeat calls share one client.
        """
        ...

    def client_for_core_purpose(
        self, purpose: str,
    ) -> "ChatLike | None":
        """Resolve a core-purpose name (e.g. ``self_thinking``,
        ``compact``, ``classifier``) by looking up
        ``cfg.llm.core_purposes[purpose]`` → tag name → client.
        Convenience wrapper over ``client_for_tag``.
        """
        ...

    def embed_client(self) -> "ChatLike | None":
        """Resolve ``cfg.llm.embedding`` → client. Used by
        ``EmbedderEngine`` defaults. ``None`` when no embedding tag
        is bound.
        """
        ...

    def rerank_client(self) -> "ChatLike | None":
        """Resolve ``cfg.llm.reranker`` → client. Used by
        ``RerankerEngine`` defaults. ``None`` when no reranker tag is
        bound; the default ``RerankerEngine`` falls back to scripted
        scoring in that case.
        """
        ...
