"""``reranker`` Engine — score-based reordering for recall + KB dedup.

Default impl ``DefaultRerankerEngine`` embeds the no-LLM fallback so
the slot always has a working impl — there is no ``reranker = None``
tri-state any more. When the user has bound a reranker tag the Engine
forwards to that client; when unbound or the upstream call fails the
Engine returns preserve-order scores (decreasing floats) so callers'
stable sort leaves the input order intact.
"""
from krakey.engines.catalog import EngineImpl
from krakey.engines.reranker.default import DefaultRerankerEngine

BUILTIN_ENGINES = {
    "passthrough": EngineImpl(
        cls=DefaultRerankerEngine,
        description=(
            "Forwards to the rerank-tag client when bound; preserve-"
            "order fallback when unbound or upstream errors."
        ),
    ),
}

DEFAULT_ENGINE = "passthrough"

__all__ = ["BUILTIN_ENGINES", "DEFAULT_ENGINE", "DefaultRerankerEngine"]
