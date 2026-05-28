"""Engine catalog primitives — short-name → impl-class lookup.

Each Engine slot ships a ``meta.yaml`` next to its ``__init__.py``::

    # krakey/engines/decision/meta.yaml
    slot: decision
    description: |
      Translate Self's [DECISION] text into structured tool calls.
    builtin_engines:
      - name: tool_call_parser
        factory_module: krakey.engines.decision.tool_call_parser
        factory_attr: ToolCallParserDecisionEngine
        default: true
        description: Scripted <tool_call> parser. No LLM call.
      - name: hypothalamus
        factory_module: krakey.engines.decision.hypothalamus
        factory_attr: HypothalamusDecisionEngine
        description: LLM translator. Bind core_purposes.hypothalamus.
        config_schema:
          - field: temperature
            type: number_float
            default: 0.7
            help: Sampling temperature for the translator LLM.
        dependencies:
          - "some-llm-client>=1.0"
        post_install:
          - args: ["{python}", "-m", "some_llm_client", "setup"]
            description: "Run one-time client setup."
            optional: false

The user picks an impl by SHORT NAME in ``config.yaml``::

    core_implementations:
      decision: hypothalamus
    engine_configs:
      decision:
        hypothalamus:
          temperature: 0.5

Plugin-supplied engines extend the same catalog through their own
``meta.yaml`` (``kind: engine``, ``slot: decision``); the plugin's
top-level ``config_schema:`` block becomes the engine's
``config_schema``. The registry merges built-in + plugin catalogs
at resolve time. The ``module.path:ClassName`` dotted-path form
keeps working as a power-user fallback when the value contains a
colon. ``engine_system/defaults.py`` carries an emergency
dotted-path table the loader falls back to when a slot's
``meta.yaml`` is missing or malformed.
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

    ``dependencies`` is a list of pip-installable spec strings
    (e.g. ``"some-package>=1.0"``) that must be present for this
    engine impl to run. The ``krakey install`` CLI is expected to
    collect and install these. Defaults to empty list.

    ``post_install`` is a list of secondary install commands run
    AFTER pip — for things pip can't drive (e.g. downloading binary
    assets). Each entry is
    ``{args: list[str], description: str, optional: bool}``.
    The token ``{python}`` inside ``args`` is replaced with
    ``sys.executable`` at run-time. Defaults to empty list.
    """
    cls: type
    description: str
    config_schema: list[dict[str, Any]] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    post_install: list[dict[str, Any]] = field(default_factory=list)
