"""GET /api/plugins \u2014 unified tentacle + sensory + plugin report."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from src.dashboard.services.plugins import PluginsService


def register(app: FastAPI, *, plugins: PluginsService) -> None:

    @app.get("/api/plugins")
    async def report():  # noqa: ANN201
        try:
            return plugins.report()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
