"""Bearer-token auth for the dashboard.

Why
---
The dashboard binds 127.0.0.1 by default, but on a multi-tenant box
or when a malicious local process / browser tab knows the port, that
loopback isn't a real boundary. We require a bearer token on
``/api/*`` and ``/ws/*``, leaving the static UI shell + favicon open
so the index page can render and the paste-token form can recover
when the browser doesn't have a token cached.

Token persistence
-----------------
On first start the runtime calls :func:`load_or_create_token` with
the dashboard's data dir; we either re-read an existing
``dashboard.token`` file or generate fresh 32 random bytes (hex) and
persist them. Re-using the file keeps already-open browser tabs
authenticated across restarts so the user doesn't have to copy the
URL each time. POSIX: ``chmod 600`` after write. Windows: relies on
the user directory's existing ACLs (``os.chmod`` is largely a no-op
there, so we don't pretend otherwise).
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse


def load_or_create_token(path: Path | str) -> str:
    """Read the token from ``path``; generate + persist if missing."""
    p = Path(path)
    if p.exists():
        existing = p.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    p.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(32)
    p.write_text(token, encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        # chmod is largely a no-op on Windows; don't fail startup
        # over a permission tweak we couldn't apply.
        pass
    return token


# Paths the auth gate lets through unconditionally. Just enough for
# the index page + its CSS + its static assets to render so the
# paste-token form can recover an unauthenticated browser tab.
_OPEN_PATHS = frozenset({"/", "/style.css"})
_OPEN_PREFIXES: tuple[str, ...] = ("/static/", "/favicon")


def _path_is_open(path: str) -> bool:
    if path in _OPEN_PATHS:
        return True
    return any(path.startswith(p) for p in _OPEN_PREFIXES)


def _extract_http_token(request) -> str | None:
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        candidate = auth[7:].strip()
        if candidate:
            return candidate
    qtok = request.query_params.get("token")
    return qtok or None


def attach_token_auth(app: FastAPI, token: str | None) -> None:
    """Install HTTP middleware that gates everything except the SPA
    shell on ``token``. Falsy ``token`` is a no-op (tests use this
    path to keep their existing fixtures working without baking in
    a token everywhere)."""
    if not token:
        return

    @app.middleware("http")
    async def _gate(request, call_next):
        if _path_is_open(request.url.path):
            return await call_next(request)
        provided = _extract_http_token(request)
        if not provided or not secrets.compare_digest(provided, token):
            return JSONResponse(
                status_code=401,
                content={
                    "error": "auth required",
                    "hint": (
                        "include the dashboard token as "
                        "'Authorization: Bearer <T>' or '?token=<T>'"
                    ),
                },
            )
        return await call_next(request)


async def ws_check_token(ws: WebSocket, token: str | None) -> bool:
    """Verify a WebSocket carries the right token before ``accept``.

    Returns True when the caller should proceed; on False the socket
    has already been closed with policy-violation (1008) and the
    caller MUST ``return`` immediately. ``token=None`` is the
    auth-disabled path (tests).
    """
    if not token:
        return True
    auth = ws.headers.get("authorization") or ""
    provided: str | None = None
    if auth.lower().startswith("bearer "):
        candidate = auth[7:].strip()
        if candidate:
            provided = candidate
    if provided is None:
        provided = ws.query_params.get("token") or None
    if not provided or not secrets.compare_digest(provided, token):
        await ws.close(code=1008)
        return False
    return True
