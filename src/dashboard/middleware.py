"""HTTP middlewares."""
from __future__ import annotations

from fastapi import FastAPI


def attach_no_cache(app: FastAPI) -> None:
    """Force `Cache-Control: no-store` on the SPA shell + static assets.

    During development the dashboard is rebuilt constantly; a stale
    app.js / style.css cached by the browser is the #1 source of "why
    is my change not showing" bug reports. Cost is negligible (single-
    user tool, everything served from loopback).
    """

    @app.middleware("http")
    async def _no_cache_static(request, call_next):
        resp = await call_next(request)
        path = request.url.path
        if path.startswith("/static/") or path == "/" or path == "/style.css":
            resp.headers["Cache-Control"] = "no-store, must-revalidate"
            resp.headers["Pragma"] = "no-cache"
        return resp
