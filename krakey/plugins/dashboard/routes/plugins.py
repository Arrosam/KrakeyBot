"""Plugin report + per-project config edit routes.

- ``GET  /api/plugins``                   — snapshot of every known
  tool + channel + plugin project (config schema, current values,
  enabled flag).
- ``POST /api/plugins/{project}/config``  — save dashboard edits
  into the project's per-plugin YAML file. Body: ``{enabled,
  values}``.
- ``GET  /api/plugins/deps_status``       — per-plugin install
  state (which plugins have unsatisfied pip deps / post_install
  hooks). Drives the plugin-list "needs install" badges.
- ``POST /api/plugins/install``           — kick off
  ``krakey install`` programmatically. Body: ``{upgrade?: bool}``.
  Synchronous: returns ``{rc, stdout, stderr}`` after pip +
  post_install finish. Long-running, but the user is staring at
  the dashboard and wants the result; SSE streaming is a
  future-stage refinement.
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

    @app.get("/api/plugins/deps_status")
    async def deps_status():  # noqa: ANN201
        try:
            return plugins.deps_status()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    @app.post("/api/plugins/install")
    async def install_plugins(body: dict | None = None):  # noqa: ANN201
        try:
            return plugins.install(body or {})
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    @app.post("/api/plugins/hot_reload")
    async def hot_reload_plugins():  # noqa: ANN201
        """Add newly-enabled plugins without a process restart.
        Returns the runtime's report; ``still_pending_remove`` is
        the operator-facing hint that one or more plugins still
        need a full restart to take effect (hot-add only)."""
        try:
            return await plugins.hot_reload()
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
