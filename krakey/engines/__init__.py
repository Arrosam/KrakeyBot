"""Engine layer — built-in default impls + the registry that wires them.

Two responsibilities live in this package:

  1. **Default impls** of every Engine slot — one subpackage per slot
     (``engines/memory/``, ``engines/context/``, ...).
  2. **EngineRegistry** — turns ``cfg.core_implementations.<slot>``
     dotted paths (or built-in defaults) into concrete instances,
     fail-fast on any malformed / missing / Protocol-violating impl.

Plugins reach resolved Engine instances via ``PluginContext.services``
(typed against the Protocols in ``krakey/interfaces/engines/``).
Plugins MAY import shared helpers from this package — e.g.
``engines/recall/gm_query.py`` is consumed by both the in-tree
recall Engine and the ``memory_recall`` Tool plugin — but they MUST
NOT depend on a specific Engine impl class.
"""
from krakey.engines.registry import EngineRegistry

__all__ = ["EngineRegistry"]
