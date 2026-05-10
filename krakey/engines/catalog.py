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
            config_schema=[
                {"field": "temperature", "type": "number_float",
                 "default": 0.7,
                 "help": "Sampling temperature for the translator LLM."},
            ],
        ),
    }
    DEFAULT_ENGINE = "tool_call_parser"

The user picks an impl by SHORT NAME in ``config.yaml``::

    core_implementations:
      decision: hypothalamus
    engine_configs:
      decision:
        hypothalamus:
          temperature: 0.5

Plugin-supplied engines extend the same catalog through their
``meta.yaml`` (``kind: engine``, ``slot: decision``); the plugin's
top-level ``config_schema:`` block becomes the engine's
``config_schema``. The registry merges built-in + plugin catalogs
at resolve time. The legacy ``module.path:ClassName`` dotted-path
form keeps working as a power-user fallback when the value
contains a colon.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

    ``config_schema`` describes the engine's user-tunable options.
    Each entry is ``{field, type, default?, help?, choices?}`` —
    same shape plugins use in ``meta.yaml``'s top-level
    ``config_schema``. The dashboard renders a schema-driven form
    under the slot's dropdown when a schema-bearing impl is
    selected; the user's values flow through to the engine
    constructor's ``config`` kwarg.
    """
    cls: type
    description: str
    config_schema: list[dict[str, Any]] = field(default_factory=list)
