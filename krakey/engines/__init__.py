"""Engine layer — built-in default impls + the registry that wires them.

Three responsibilities live in this package:

  1. **Default impls** of every Engine slot — one subpackage per slot
     (``engines/memory/``, ``engines/context/``, ...). Steps 3-11 of
     the Engine refactor populate these.
  2. **EngineBundle** dataclass — carries the 10 resolved Engine
     instances that the runtime needs at every beat.
  3. **EngineRegistry** — turns ``cfg.core_implementations.<slot>``
     dotted paths (or built-in defaults) into concrete instances,
     fail-fast on any malformed / missing / Protocol-violating impl.

Plugins NEVER import from this package. The runtime hands plugin code
the resolved Engine instances via ``PluginContext.services`` (typed
against the Protocols in ``krakey/interfaces/engines/``).
"""
from krakey.engines.bundle import EngineBundle
from krakey.engines.registry import EngineRegistry

__all__ = ["EngineBundle", "EngineRegistry"]
