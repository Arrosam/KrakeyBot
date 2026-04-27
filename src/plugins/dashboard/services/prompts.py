"""PromptsService — Protocol for the /api/prompts route."""
from __future__ import annotations

from typing import Any, Protocol


class PromptsService(Protocol):
    """Return the N most recent built prompts (ring buffer)."""

    def recent(self, *, limit: int) -> list[dict[str, Any]]: ...
