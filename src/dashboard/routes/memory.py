"""Read-only browser for the graph memory + knowledge bases.

Every endpoint is a thin HTTP adapter over MemoryService; RuntimeError
\u2192 503, LookupError \u2192 404, everything else bubbles.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from src.dashboard.services.memory import MemoryService


def register(app: FastAPI, *, memory: MemoryService) -> None:

    @app.get("/api/gm/nodes")
    async def gm_nodes(
        category: str | None = None,
        limit: int = Query(default=200, ge=1, le=2000),
    ):  # noqa: ANN201
        try:
            nodes = await memory.list_gm_nodes(category=category,
                                                 limit=limit)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        return {"count": len(nodes), "nodes": nodes}

    @app.get("/api/gm/edges")
    async def gm_edges(
        limit: int = Query(default=500, ge=1, le=5000),
    ):  # noqa: ANN201
        try:
            edges = await memory.list_gm_edges(limit=limit)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        return {"count": len(edges), "edges": edges}

    @app.get("/api/gm/stats")
    async def gm_stats():  # noqa: ANN201
        try:
            return await memory.gm_stats()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    @app.get("/api/kbs")
    async def kbs():  # noqa: ANN201
        try:
            return {"kbs": await memory.list_kbs()}
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    @app.get("/api/kb/{kb_id}/entries")
    async def kb_entries(
        kb_id: str,
        limit: int = Query(default=200, ge=1, le=2000),
    ):  # noqa: ANN201
        try:
            entries = await memory.kb_entries(kb_id=kb_id, limit=limit)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except LookupError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"kb_id": kb_id, "count": len(entries), "entries": entries}
