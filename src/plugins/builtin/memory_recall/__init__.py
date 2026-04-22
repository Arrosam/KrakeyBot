"""Built-in `memory_recall` plugin — Self-driven GM + KB exploration."""
from __future__ import annotations

from src.interfaces.tentacle import Tentacle
from src.tentacles.memory_recall import MemoryRecallTentacle


MANIFEST = {
    "name": "memory_recall",
    "description": "Active recall of GM nodes (and, via KB index nodes, "
                   "their KB entries). Self dispatches this when she "
                   "wants to reflect or dig into past learning.",
    "is_internal": True,
    "config_schema": [
        {"field": "enabled",       "type": "bool",   "default": True},
        {"field": "default_top_k", "type": "number", "default": 8,
         "help": "Default number of top-K nodes to return when Self "
                 "doesn't specify."},
    ],
}


def create_tentacle(config: dict, deps: dict) -> Tentacle:
    return MemoryRecallTentacle(
        gm=deps["gm"],
        embedder=deps["embedder"],
        kb_registry=deps["kb_registry"],
        default_top_k=int(config.get("default_top_k", 8)),
    )
