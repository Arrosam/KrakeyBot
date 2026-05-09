"""``DefaultContextEngine`` — default ContextEngine impl.

Alias for the long-standing ``krakey.prompt.builder.PromptBuilder``;
the class already implements both ``build_default_elements`` and
``render`` and structurally satisfies the new ``ContextEngine``
Protocol. Subclassing rather than re-exporting so callers can
``isinstance(x, DefaultContextEngine)`` to identify the default
impl in tests, and so future feature additions can land here
without disturbing the underlying ``PromptBuilder``.
"""
from __future__ import annotations

from krakey.prompt.builder import PromptBuilder


class DefaultContextEngine(PromptBuilder):
    """Default ContextEngine — same behavior as the long-standing
    PromptBuilder, exported under the Engine slot name."""
    pass
