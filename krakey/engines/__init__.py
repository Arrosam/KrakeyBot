"""Engine layer — built-in default impls + the registry that wires them.

Two responsibilities live in this package:

  1. **Default impls** of every Engine slot — one subpackage per slot
     (``engines/memory/``, ``engines/context/``, ...).
  2. **EngineRegistry** — turns ``cfg.core_implementations.<slot>``
     dotted paths (or built-in defaults) into concrete instances,
     fail-fast on any malformed / missing / Protocol-violating impl.

Plugins NEVER import from this package. The runtime hands plugin code
the resolved Engine instances via ``PluginContext.services`` (typed
against the Protocols in ``krakey/interfaces/engines/``).
"""
from krakey.engines.registry import EngineRegistry

__all__ = ["EngineRegistry"]
