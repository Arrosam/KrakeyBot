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

from pathlib import Path

import yaml

from src.dashboard.services.config import ConfigService
from src.models.config import llm_params_schema
from src.reflects.discovery import discover_reflects


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
          * ``llm_params`` \u2014 LLMParams field descriptors (max_tokens,
            temperature, reasoning_mode, ...). Returned as
            ``[{field, type, default, help, choices?}, ...]``. The
            dashboard renders this under each tag's params editor;
            the same shape is reused by plugin ``config_schema``
            entries.
        """
        return {"llm_params": llm_params_schema()}

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

    @app.get("/api/reflects/available")
    async def get_available_reflects():  # noqa: ANN201
        """List Reflects discoverable on disk.

        Pure-text scan — never imports any plugin module
        (architectural invariant). The dashboard renders this list
        as the "Available Reflects" UI on the settings page so users
        see what they can enable + which LLM purposes each declares.
        """
        out = []
        for name, meta in discover_reflects().items():
            out.append({
                "name": meta.name,
                "kind": meta.kind,
                "description": meta.description,
                "config_schema": meta.config_schema,
                "llm_purposes": meta.llm_purposes,
            })
        out.sort(key=lambda r: r["name"])
        return {"reflects": out}

    @app.get("/api/reflects/{name}/config")
    async def get_reflect_config(name: str):  # noqa: ANN201
        """Read the per-Reflect config file
        (``workspace/reflects/<name>/config.yaml``).

        Returns ``{}`` if the file doesn't exist yet — that's the
        valid initial state, the plugin operates with whatever
        defaults its code defines.
        """
        path = Path("workspace") / "reflects" / name / "config.yaml"
        if not path.exists():
            return {"path": str(path), "config": {}}
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise HTTPException(
                status_code=400,
                detail=f"plugin config parse failed: {e}",
            )
        if not isinstance(raw, dict):
            raise HTTPException(
                status_code=400,
                detail="plugin config top-level must be a mapping",
            )
        return {"path": str(path), "config": raw}

    @app.post("/api/reflects/{name}/config")
    async def post_reflect_config(name: str, payload: dict = Body(...)):  # noqa: ANN201
        """Write the per-Reflect config file. Creates directory if
        needed; replaces existing content."""
        new_cfg = payload.get("config")
        if not isinstance(new_cfg, dict):
            raise HTTPException(
                status_code=400,
                detail="payload must include 'config' (mapping)",
            )
        path = Path("workspace") / "reflects" / name / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(new_cfg, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return {
            "status": "saved", "path": str(path),
            "restart_required": True,
        }
