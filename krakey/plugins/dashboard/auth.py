"""Cookie-session auth for the dashboard.

Why
---
The dashboard binds 127.0.0.1 by default, but on a multi-tenant box
or when a malicious local process / browser tab knows the port, that
loopback isn't a real boundary. We gate the entire app surface on a
per-installation token. The first request carrying ``?token=<T>``
(or a Bearer header) gets an HttpOnly+SameSite=Strict session
cookie set; subsequent requests authenticate via the cookie and the
JS layer doesn't have to plumb the token anywhere — the browser
attaches the cookie automatically to fetches and to the WebSocket
handshake.

The index page itself is gated. Unauthenticated browsers get a
server-rendered "auth required" page with a paste-token form (it
GETs back to ``/?token=<value>`` so the existing one-click flow
works either way). API clients get a JSON 401. Static assets stay
open — they're just CSS / images / JS bundle, no data leaks.

Why this matters
----------------
A previous design rendered the SPA shell open and used a JS-driven
modal to "block" interaction. That gave the wrong impression of
security: a user could open DevTools, delete the modal, and still
see the UI chrome — even though every API call would 401 in the
background. The current design returns nothing from the server
without auth: no SPA, no UI, no shell. Removing DOM elements is no
longer a thing because there are no SPA DOM elements to remove
until the cookie is valid.

Token persistence
-----------------
On first start the runtime calls :func:`load_or_create_token` with
the dashboard's data dir; we either re-read an existing
``dashboard.token`` file or generate fresh 32 random bytes (hex) and
persist them. Re-using the file keeps already-open browser tabs
authenticated across restarts. POSIX: ``chmod 600`` after write.
Windows: relies on the user directory's existing ACLs.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse


# Cookie name. Underscores instead of hyphens so the cookie is a
# valid Python identifier in tooling and matches a typical session
# cookie shape.
COOKIE_NAME = "krakey_dash_session"


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


# Paths the auth gate lets through unconditionally. Just static
# assets — no SPA shell, no API, no WS. Even the index ``/`` is
# gated; unauthenticated browsers see a server-rendered auth-page
# instead of the SPA.
_OPEN_PREFIXES: tuple[str, ...] = ("/static/", "/favicon")
_OPEN_PATHS = frozenset({"/style.css"})


def _path_is_open(path: str) -> bool:
    if path in _OPEN_PATHS:
        return True
    return any(path.startswith(p) for p in _OPEN_PREFIXES)


def _request_provides_token(request, token: str) -> bool:
    """True when ``request`` carries a matching token in any of:
    cookie (preferred — set by us on first auth), ``?token=`` query
    (one-click URL or paste form), ``Authorization: Bearer`` header
    (non-browser API clients)."""
    cookie = request.cookies.get(COOKIE_NAME)
    if cookie and secrets.compare_digest(cookie, token):
        return True
    qtok = request.query_params.get("token")
    if qtok and secrets.compare_digest(qtok, token):
        return True
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        candidate = auth[7:].strip()
        if candidate and secrets.compare_digest(candidate, token):
            return True
    return False


_AUTH_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Krakey Dashboard — auth required</title>
  <link rel="icon" type="image/png" href="/static/logo.png">
  <link rel="stylesheet" href="/style.css">
</head>
<body class="auth-required">
  <main class="auth-page">
    <div class="auth-card">
      <h2>Dashboard token required</h2>
      <p>Paste the token from the Krakey runtime log, or open the
        one-click URL the runtime printed at startup.</p>
      <form method="get" action="/" autocomplete="off">
        <input type="password" name="token" autofocus
               spellcheck="false" placeholder="dashboard token">
        <button type="submit">Sign in</button>
      </form>
      <p class="hint">Stored as an HttpOnly cookie scoped to this
        origin only.</p>
    </div>
  </main>
</body>
</html>
"""


def attach_token_auth(app: FastAPI, token: str | None) -> None:
    """Install HTTP middleware that gates everything except a tiny
    set of static-asset paths on ``token``. Falsy ``token`` is a
    no-op (tests use this path to keep existing fixtures working)."""
    if not token:
        return

    @app.middleware("http")
    async def _gate(request, call_next):
        path = request.url.path
        if _path_is_open(path):
            return await call_next(request)
        if _request_provides_token(request, token):
            response = await call_next(request)
            # Set the session cookie if the request didn't already
            # carry it (i.e. they authenticated via query / header
            # this time). Subsequent requests use the cookie and
            # don't have to keep echoing the token in URLs / headers.
            if request.cookies.get(COOKIE_NAME) != token:
                response.set_cookie(
                    COOKIE_NAME, token,
                    httponly=True,
                    samesite="strict",
                    secure=request.url.scheme == "https",
                    max_age=60 * 60 * 24 * 365,
                    path="/",
                )
            return response
        # Unauthenticated. Browser navigations to "/" get the auth
        # page so the user can paste a token; everything else gets
        # JSON so API clients have a parseable failure.
        if path == "/":
            return HTMLResponse(_AUTH_PAGE_HTML, status_code=401)
        return JSONResponse(
            status_code=401,
            content={
                "error": "auth required",
                "hint": (
                    "include the dashboard token as "
                    "'Authorization: Bearer <T>', '?token=<T>', or "
                    "the krakey_dash_session cookie"
                ),
            },
        )


async def ws_check_token(ws: WebSocket, token: str | None) -> bool:
    """Verify a WebSocket carries the right token.

    Browsers attach cookies to the WS handshake automatically, so
    the cookie path is the typical one. Query-param + Authorization
    header are kept for non-browser clients.

    Returns True when the caller should proceed (and is responsible
    for ``await ws.accept()``); on False the helper has accepted +
    immediately closed with policy-violation (1008) and the caller
    MUST ``return`` immediately. ``token=None`` is the auth-disabled
    path (tests).
    """
    if not token:
        return True
    cookie = ws.cookies.get(COOKIE_NAME)
    if cookie and secrets.compare_digest(cookie, token):
        return True
    qtok = ws.query_params.get("token")
    if qtok and secrets.compare_digest(qtok, token):
        return True
    auth = ws.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        candidate = auth[7:].strip()
        if candidate and secrets.compare_digest(candidate, token):
            return True
    try:
        await ws.accept()
    except Exception:  # noqa: BLE001 — already torn down by client
        return False
    await ws.close(code=1008)
    return False
