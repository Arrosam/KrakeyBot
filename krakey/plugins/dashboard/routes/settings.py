"""Settings REST \u2014 read / write config + trigger restart.

GET  /api/settings        \u2192 raw + parsed yaml.
POST /api/settings        \u2192 write (either 'parsed' or raw 'raw' string).
POST /api/restart         \u2192 call the on_restart hook.
GET  /api/config/schema   \u2192 field descriptors for dynamic UI rendering.

The schema endpoint is the single source of truth the dashboard uses
to render LLM role params (and any future introspected config section).
Adding a field to ``LLMParams`` in ``krakey/models/config.py`` surfaces it
in the UI on the next reload \u2014 no JS edits required.

Every validation error is 400; missing backing config is 503; missing
file is 404; write-time serialization failures are 400.
"""
from __future__ import annotations

from fastapi import Body, FastAPI, HTTPException

from pathlib import Path

import yaml

from krakey.plugins.dashboard.services.config import ConfigService
from krakey.models.config import llm_params_schema
from krakey.plugin_system.catalogue import (
    list_available_plugins as _discover_unified,
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

    @app.get("/api/modifiers/available")
    async def get_available_modifiers():  # noqa: ANN201
        """List unified-format plugins discoverable on disk.

        Pure-text scan — never imports any plugin module
        (architectural invariant). Each plugin's ``components`` array
        carries modifier / tool / channel entries with their own
        ``llm_purposes`` declarations.
        """
        out = []
        for name, meta in _discover_unified().items():
            # Aggregate llm_purposes across components for the UI's
            # tag-binding form (the dashboard treats them at the
            # plugin level even though they're declared per-component).
            all_purposes: list = []
            for c in meta.components:
                all_purposes.extend(c.llm_purposes)
            out.append({
                "name": meta.name,
                "description": meta.description,
                "config_schema": meta.config_schema,
                "components": [
                    {"kind": c.kind, "role": c.role}
                    for c in meta.components
                ],
                "llm_purposes": all_purposes,
            })
        out.sort(key=lambda r: r["name"])
        return {"modifiers": out}

    @app.get("/api/modifiers/{name}/config")
    async def get_modifier_config(name: str):  # noqa: ANN201
        """Read the per-plugin config file
        (``workspace/plugins/<name>/config.yaml``).

        Returns ``{}`` if the file doesn't exist yet — valid initial
        state. (Endpoint name kept for back-compat with the dashboard
        JS; the underlying path moved from ``workspace/modifiers/`` to
        ``workspace/plugins/`` in the unification refactor.)
        """
        path = Path("workspace") / "plugins" / name / "config.yaml"
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

    @app.post("/api/modifiers/{name}/config")
    async def post_modifier_config(name: str, payload: dict = Body(...)):  # noqa: ANN201
        """Write the per-plugin config file. Creates directory if
        needed; replaces existing content."""
        new_cfg = payload.get("config")
        if not isinstance(new_cfg, dict):
            raise HTTPException(
                status_code=400,
                detail="payload must include 'config' (mapping)",
            )
        path = Path("workspace") / "plugins" / name / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(new_cfg, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return {
            "status": "saved", "path": str(path),
            "restart_required": True,
        }
