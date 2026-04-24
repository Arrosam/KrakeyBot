"""GET /api/prompts \u2014 recent-prompts ring buffer."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from src.dashboard.services.prompts import PromptsService


def register(app: FastAPI, *, prompts: PromptsService) -> None:

    @app.get("/api/prompts")
    async def recent(
        limit: int = Query(default=50, ge=1, le=500),
    ):  # noqa: ANN201
        try:
            items = prompts.recent(limit=limit)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        return {"prompts": items, "count": len(items)}
