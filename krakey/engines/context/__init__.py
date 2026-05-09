"""``context`` Engine — prompt assembly.

Default impl ``DefaultContextEngine`` (in ``default.py``) is the
existing ``krakey.prompt.builder.PromptBuilder`` exported under the
new Engine name. The behavioral surface (``build_default_elements`` +
``render``) is unchanged; the rename reflects that prompt assembly
is one of the swappable core engines, not just a builder utility.

The ``ContextEngine`` Protocol the runtime depends on lives at
``krakey.interfaces.engines.context``.

The ``prompt`` package (``krakey/prompt/``) stays where it is during
the migration window — moving it under ``engines/context/`` is a
purely cosmetic file move slated for the step 14 cleanup. Until
then the default Engine is a one-line alias.
"""
from krakey.engines.context.default import DefaultContextEngine

__all__ = ["DefaultContextEngine"]
