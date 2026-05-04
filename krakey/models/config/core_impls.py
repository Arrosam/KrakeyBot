"""Core service slot overrides — let users replace built-in core
implementations (memory, prompt builder, embedder, ...) with their own.

The user writes a dotted path in config.yaml; the runtime imports it
at startup, instantiates with runtime-supplied kwargs, and validates
the result satisfies the slot's Protocol. Empty / missing slots fall
back to the built-in default.

This file owns ONLY the config dataclass + its loader. The actual
import + validation lives in
``krakey.runtime.service_resolver.ServiceResolver``; the per-slot
Protocols live under ``krakey.interfaces.services.<slot>``.

Slot fields are ALL declared optional with empty-string defaults so
omitting the whole section in config.yaml is the same as opting in to
the built-in defaults across the board — the additive-plugin spirit
applied to core: nothing happens unless the user opts in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CoreImplementations:
    """Optional dotted-path overrides for built-in core services.

    Each field corresponds to a ``slot`` in
    ``ServiceResolver.resolve(slot=...)``. Format:
    ``module.path:ClassName`` (standard entry-point style). Empty
    string = use the built-in default.

    Phase 1 surfaces three slots — ``embedder``, ``reranker``,
    ``prompt_builder`` — because they already have Protocols (or
    are stateless enough to give one easily). The remaining fields
    are reserved for later phases and currently still ignored by
    the resolver; declared here so config files don't break when
    later phases ship.
    """

    # Phase 1 — wired through resolver:
    embedder: str = ""
    reranker: str = ""
    prompt_builder: str = ""

    # Phase 2+ — declared so configs round-trip; resolver does NOT
    # consult these yet. Setting them today is silently ignored
    # until the matching phase ships its Protocol + wiring.
    memory: str = ""
    kb_registry: str = ""
    sliding_window: str = ""
    sleep_manager: str = ""
    llm_client_factory: str = ""

    def get(self, slot: str) -> str:
        """Return the override path for a slot, or '' if not set.

        Used by ``ServiceResolver`` so it doesn't have to know which
        slots exist as fields — slots not on this dataclass simply
        return ''.
        """
        return getattr(self, slot, "") or ""


def _build_core_implementations(raw: Any) -> CoreImplementations:
    """Parse the ``core_implementations:`` section of config.yaml.

    Robust to:
      * missing section (raw == {} or None) → all defaults
      * non-dict input → log nothing, return defaults (loader-level
        warning would obscure real config errors)
      * unknown keys → silently dropped (a user typo here is a no-op,
        not a crash; the resolver will see ``''`` for the slot and
        fall through to the default)
    """
    if not isinstance(raw, dict):
        return CoreImplementations()
    known = {f for f in CoreImplementations.__dataclass_fields__}
    return CoreImplementations(
        **{k: str(v or "") for k, v in raw.items() if k in known}
    )
