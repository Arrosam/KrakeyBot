"""``context`` Engine — prompt assembly.

The slot's catalog of impls (default + alternatives) lives in
``meta.yaml`` next to this file. The ContextEngine Protocol lives at
``krakey.interfaces.engines.context``.
"""
from krakey.engines.context.default import DefaultContextEngine

__all__ = ["DefaultContextEngine"]
