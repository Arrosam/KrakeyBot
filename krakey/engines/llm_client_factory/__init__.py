"""``llm_client_factory`` slot — per-tag LLMClient class substitution.

Distinct from ``llm_factory`` (which substitutes the entire factory
Engine): this slot lets the user swap the *class* the factory
instantiates per tag. The standard ``LLMClient`` handles
OpenAI-compatible + Anthropic providers; users who need a totally
different transport (mock, replay, gRPC, …) point this slot at their
own class with the same ``chat`` / ``embed`` / ``rerank`` surface.

Resolved per-tag by ``krakey.llm.resolve.resolve_llm_for_tag`` rather
than once at startup — that's why this slot doesn't show up on
Runtime as ``self.<slot>``. Cataloged here so the EngineRegistry's
short-name resolution + the dashboard's slot dropdown stay uniform
across all 11 ``core_implementations`` entries.
"""
from krakey.engines.catalog import EngineImpl
from krakey.llm.client import LLMClient

BUILTIN_ENGINES = {
    "default": EngineImpl(
        cls=LLMClient,
        description=(
            "Standard LLMClient — speaks OpenAI-compatible and "
            "Anthropic APIs. Used by every tag unless overridden."
        ),
    ),
}

DEFAULT_ENGINE = "default"

__all__ = ["BUILTIN_ENGINES", "DEFAULT_ENGINE"]
