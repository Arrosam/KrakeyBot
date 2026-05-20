"""Runtime pause/resume control routes.

- ``GET  /api/runtime/state``   — return ``{"paused": bool}``
- ``POST /api/runtime/pause``   — request an indefinite pause;
                                  return ``{"paused": True, "applied": bool}``
- ``POST /api/runtime/resume``  — request a resume;
                                  return ``{"paused": False, "applied": bool}``

Routes are registered unconditionally. When ``runtime`` is None the
handlers return 503 so the test suite can assert the correct shape
regardless of whether a real runtime is present.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse


def register(app: FastAPI, *, runtime) -> None:

    @app.get("/api/runtime/state")
    async def runtime_state():  # noqa: ANN201
        if runtime is None:
            return JSONResponse(
                status_code=503,
                content={"error": "runtime not available"},
            )
        return {"paused": bool(runtime.paused)}

    @app.post("/api/runtime/pause")
    async def runtime_pause():  # noqa: ANN201
        if runtime is None:
            return JSONResponse(
                status_code=503,
                content={"error": "runtime not available"},
            )
        applied = bool(runtime.request_pause())
        return {"paused": True, "applied": applied}

    @app.post("/api/runtime/resume")
    async def runtime_resume():  # noqa: ANN201
        if runtime is None:
            return JSONResponse(
                status_code=503,
                content={"error": "runtime not available"},
            )
        applied = bool(runtime.request_resume())
        return {"paused": False, "applied": applied}
