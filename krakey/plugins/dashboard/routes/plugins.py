"""Plugin report + per-project config edit routes.

- ``GET /api/plugins``: snapshot of every known tool + channel +
  plugin project (config schema, current values, enabled flag).
- ``POST /api/plugins/{project}/config``: save dashboard edits into
  the project's per-plugin YAML file. Body: ``{enabled, values}``.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from krakey.plugins.dashboard.services.plugins import PluginsService


def register(app: FastAPI, *, plugins: PluginsService) -> None:

    @app.get("/api/plugins")
    async def report():  # noqa: ANN201
        try:
            return plugins.report()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    @app.post("/api/plugins/{project}/config")
    async def update_config(project: str, body: dict):  # noqa: ANN201
        try:
            return plugins.update_config(project, body)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
