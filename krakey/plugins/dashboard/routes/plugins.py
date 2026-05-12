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
- ``GET  /api/plugins/stale_configs``     — list per-plugin config
  dirs that no longer back any installed plugin (leftovers from
  removed / renamed plugins). Drives the dashboard's "delete
  stale configs" panel.
- ``POST /api/plugins/stale_configs/delete``  — delete one stale
  config folder. Body: ``{name: <plugin>}``. ValueError → 400
  (unsafe name / plugin still active); LookupError → 404
  (already removed). The HTTP layer is intentionally thin — the
  ``RuntimePluginsService`` adapter owns every safety check so
  the same guarantees apply to direct callers.
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

    @app.get("/api/plugins/stale_configs")
    async def list_stale_configs():  # noqa: ANN201
        try:
            return {"stale": plugins.find_stale_configs()}
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    @app.post("/api/plugins/stale_configs/delete")
    async def delete_stale_config(body: dict | None = None):  # noqa: ANN201
        # Reject empty / non-string names at the HTTP boundary so the
        # service layer's ValueError is reserved for genuine policy
        # violations (unsafe chars, plugin still active, etc.) and we
        # don't have to disambiguate failure modes downstream.
        body = body or {}
        name = body.get("name")
        if not isinstance(name, str) or not name.strip():
            raise HTTPException(
                status_code=400,
                detail="`name` is required (non-empty string)",
            )
        try:
            return plugins.delete_stale_config(name.strip())
        except LookupError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    # NOTE — keep the ``{project}/config`` route registered LAST so
    # FastAPI's path matcher doesn't greedily catch
    # ``/api/plugins/stale_configs/...`` and route them through the
    # update-config handler. (Both have 3 segments today, but a future
    # refactor that drops the trailing ``/config`` would silently
    # break stale-config deletion if this ordering is reversed.)
    @app.post("/api/plugins/{project}/config")
    async def update_config(project: str, body: dict):  # noqa: ANN201
        try:
            return plugins.update_config(project, body)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
