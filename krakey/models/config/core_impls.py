"""Core service slot overrides — let users replace built-in core
Engine implementations (memory, context, embedder, ...) with their own.

The user writes a dotted path in config.yaml; the EngineRegistry
imports it at startup, instantiates with runtime-supplied kwargs,
and validates the result satisfies the slot's Engine Protocol.
Empty / missing slots fall back to the built-in default impl.

The actual import + validation lives in
``krakey.engines.registry.EngineRegistry``; the per-slot Protocols
live under ``krakey.interfaces.engines.<slot>``.

Slot fields are ALL declared optional with empty-string defaults so
omitting the section in config.yaml is the same as opting in to the
built-in defaults across the board.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CoreImplementations:
    """Dotted-path overrides for built-in core Engine impls.

    Format: ``module.path:ClassName`` (entry-point style). Empty
    string = use the built-in default.
    """

    # Engine slots — every entry maps to an Engine in
    # ``krakey.engines.<slot>`` resolved by EngineRegistry.
    embedder: str = ""
    reranker: str = ""
    memory: str = ""
    context: str = ""
    explicit_history: str = ""
    decision: str = ""
    recall: str = ""
    heartbeat: str = ""
    dispatch: str = ""

    # ``llm_factory`` substitutes the long-lived
    # LLMClientFactoryEngine — the only Engine that touches
    # ``cfg.llm``. ``llm_client_factory`` is one layer below: the
    # legacy per-tag LLMClient *class* substitution slot used
    # internally by ``resolve_llm_for_tag``. Both coexist by
    # design (factory Engine vs. per-tag class).
    llm_factory: str = ""
    llm_client_factory: str = ""

    def get(self, slot: str) -> str:
        """Return the override path for a slot, or '' if not set."""
        return getattr(self, slot, "") or ""


def _build_core_implementations(raw: Any) -> CoreImplementations:
    """Parse the ``core_implementations:`` section of config.yaml.

    Unknown keys are silently dropped — a typo here is a no-op (the
    resolver sees ``''`` for the slot and falls through to the
    default).
    """
    if not isinstance(raw, dict):
        return CoreImplementations()
    known = {f for f in CoreImplementations.__dataclass_fields__}
    return CoreImplementations(
        **{k: str(v or "") for k, v in raw.items() if k in known}
    )
