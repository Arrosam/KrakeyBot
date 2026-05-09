"""``context`` Engine — prompt assembly.

Default impl ``DefaultContextEngine`` (in ``default.py``) wraps the
``krakey.prompt.builder.PromptBuilder`` rendering logic. The Protocol
the runtime depends on lives at ``krakey.interfaces.engines.context``.
"""
from krakey.engines.context.default import DefaultContextEngine

__all__ = ["DefaultContextEngine"]
