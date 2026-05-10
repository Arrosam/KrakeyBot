"""Engine catalog primitives — short-name → impl-class lookup.

Each Engine slot ships a ``BUILTIN_ENGINES`` mapping in its own
``engines/<slot>/__init__.py``::

    BUILTIN_ENGINES = {
        "tool_call_parser": EngineImpl(
            cls=ToolCallParserDecisionEngine,
            description="Scripted <tool_call> parser. No LLM call.",
        ),
        "hypothalamus": EngineImpl(
            cls=HypothalamusDecisionEngine,
            description="LLM translator. Bind core_purposes.hypothalamus.",
        ),
    }
    DEFAULT_ENGINE = "tool_call_parser"

The user picks an impl by SHORT NAME in ``config.yaml``::

    core_implementations:
      decision: hypothalamus

Plugin-supplied engines extend the same catalog through their
``meta.yaml`` (``kind: engine``, ``slot: decision``); the registry
merges the two catalogs at resolve time. The legacy
``module.path:ClassName`` dotted-path form keeps working as a
power-user fallback when the value contains a colon.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngineImpl:
    """Catalog entry — points at one Engine impl class.

    ``cls`` is the concrete class the registry instantiates. The
    constructor must accept the runtime kwargs documented for the
    slot (or use ``**kwargs``); ``EngineRegistry._filter_kwargs``
    drops anything the constructor's signature doesn't declare.

    ``description`` is the one-line blurb the dashboard's slot
    dropdown shows next to the short name. Keep it under ~80 chars
    so it fits in a single row.
    """
    cls: type
    description: str
