"""GET /api/health \u2014 liveness probe (dependency-free).

Does NOT depend on a service. Present even when the app is booted
with `runtime=None` for narrow tests, so /api/health is always the
right answer to "is the web layer up."
"""
from __future__ import annotations

from fastapi import FastAPI


def register(app: FastAPI) -> None:

    @app.get("/api/health")
    async def health():  # noqa: ANN201
        return {"status": "ok"}
