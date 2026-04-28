"""MemoryService — Protocol for GM / KB read-only browser routes."""
from __future__ import annotations

from typing import Any, Protocol


class MemoryService(Protocol):
    """Read-only window into the graph memory + knowledge bases.

    Keeps each route handler one line: call the service, shape the
    response. SQL / runtime internals are confined to the adapter.
    """

    async def list_gm_nodes(
        self, *, category: str | None, limit: int,
    ) -> list[dict[str, Any]]: ...

    async def list_gm_edges(
        self, *, limit: int,
    ) -> list[dict[str, Any]]: ...

    async def gm_stats(self) -> dict[str, Any]: ...

    async def list_kbs(self) -> list[dict[str, Any]]: ...

    async def kb_entries(
        self, *, kb_id: str, limit: int,
    ) -> list[dict[str, Any]]: ...
