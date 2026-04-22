"""Built-in `search` plugin — DuckDuckGo web search via ddgs."""
from __future__ import annotations

from src.interfaces.tentacle import Tentacle
from src.tentacles.search import DDGSBackend, SearchTentacle


MANIFEST = {
    "name": "search",
    "description": "Web search via DuckDuckGo. Inward: results are for "
                   "Self's own reading; she decides whether to relay.",
    "is_internal": True,
    "config_schema": [
        {"field": "enabled",     "type": "bool",   "default": True},
        {"field": "max_results", "type": "number", "default": 5,
         "help": "Upper bound on results per query."},
    ],
}


def create_tentacle(config: dict, deps: dict) -> Tentacle:
    return SearchTentacle(
        backend=DDGSBackend(),
        max_results=int(config.get("max_results", 5)),
    )
