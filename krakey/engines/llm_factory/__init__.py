"""``llm_factory`` Engine — long-lived factory for LLM clients.

Default impl ``DefaultLLMClientFactoryEngine`` owns the per-tag
client cache. The ``LLMClientFactoryEngine`` Protocol lives at
``krakey.interfaces.engines.llm_factory``.

The Engine is the only place that touches ``cfg.llm`` — every other
Engine and every plugin asks the factory for a client by tag name or
core-purpose name, never reads providers / api keys / models
themselves. This is the API-key isolation boundary.
"""
from krakey.engines.catalog import EngineImpl
from krakey.engines.llm_factory.default import DefaultLLMClientFactoryEngine

BUILTIN_ENGINES = {
    "default": EngineImpl(
        cls=DefaultLLMClientFactoryEngine,
        description=(
            "Per-tag client cache backed by resolve_llm_for_tag; "
            "honours the llm_client_factory class-substitution slot."
        ),
    ),
}

DEFAULT_ENGINE = "default"

__all__ = [
    "BUILTIN_ENGINES",
    "DEFAULT_ENGINE",
    "DefaultLLMClientFactoryEngine",
]
