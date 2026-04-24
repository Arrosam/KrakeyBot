"""Settings REST \u2014 read / write config + trigger restart.

GET  /api/settings        \u2192 raw + parsed yaml.
POST /api/settings        \u2192 write (either 'parsed' or raw 'raw' string).
POST /api/restart         \u2192 call the on_restart hook.
GET  /api/config/schema   \u2192 field descriptors for dynamic UI rendering.

The schema endpoint is the single source of truth the dashboard uses
to render LLM role params (and any future introspected config section).
Adding a field to ``LLMParams`` in ``src/models/config.py`` surfaces it
in the UI on the next reload \u2014 no JS edits required.

Every validation error is 400; missing backing config is 503; missing
file is 404; write-time serialization failures are 400.
"""
from __future__ import annotations

from fastapi import Body, FastAPI, HTTPException

from src.dashboard.services.config import ConfigService
from src.models.config import (
    _ROLE_DEFAULTS,
    llm_params_schema,
    role_default_params,
)


def register(app: FastAPI, *, config: ConfigService) -> None:

    @app.get("/api/settings")
    async def get_settings():  # noqa: ANN201
        try:
            raw, parsed = config.read()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"path": str(config.path), "raw": raw, "parsed": parsed}

    @app.get("/api/config/schema")
    async def get_config_schema():  # noqa: ANN201
        """Field descriptors for dynamic UI rendering.

        Current sections:
          * ``llm_params``   \u2014 per-role LLM parameters (max_tokens,
            temperature, reasoning_mode, ...). Returned as
            ``[{field, type, default, help, choices?}, ...]`` with
            shape matching the plugin ``config_schema`` contract the
            dashboard already knows how to render.
          * ``llm_role_defaults`` \u2014 role-specific overrides the UI
            can pre-fill when a role is created (so the Self role
            shows max_tokens=8192 out of the box).
        """
        return {
            "llm_params": llm_params_schema(),
            "llm_role_defaults": {
                name: role_default_params(name)
                for name in _ROLE_DEFAULTS
            },
        }

    @app.post("/api/settings")
    async def post_settings(payload: dict = Body(...)):  # noqa: ANN201
        try:
            backup_path = config.write(
                raw=payload.get("raw"),
                parsed=payload.get("parsed"),
                backup_dir=payload.get("backup_dir") or "workspace/backups",
            )
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {
            "status": "saved",
            "backup": str(backup_path) if backup_path else None,
            "restart_required": True,
        }

    @app.post("/api/restart")
    async def restart():  # noqa: ANN201
        try:
            config.restart()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500,
                                  detail=f"restart failed: {e}")
        return {"status": "restarting"}
