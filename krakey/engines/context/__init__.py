"""``context`` Engine — prompt assembly.

Default impl ``DefaultContextEngine`` wraps the
``krakey.prompt.builder.PromptBuilder`` rendering logic. The Protocol
the runtime depends on lives at ``krakey.interfaces.engines.context``.
"""
from krakey.engines.catalog import EngineImpl
from krakey.engines.context.default import DefaultContextEngine

BUILTIN_ENGINES = {
    "default": EngineImpl(
        cls=DefaultContextEngine,
        description=(
            "Standard PromptBuilder — assembles the canonical "
            "DEFAULT_ELEMENT_KEYS prompt layers."
        ),
    ),
}

DEFAULT_ENGINE = "default"

__all__ = ["BUILTIN_ENGINES", "DEFAULT_ENGINE", "DefaultContextEngine"]
