"""Cookie-session auth on the dashboard.

Confirms:
  * the index '/' is GATED (returns 401 + auth-page HTML without
    valid auth — the SPA shell is NOT served until the cookie is
    set, so a "delete the modal in DevTools" attack gets you nothing),
  * static assets (/static/*, /style.css, /favicon*) stay open so the
    auth page can pull its CSS,
  * /api/* and /ws/* are gated,
  * '?token=<T>', 'Authorization: Bearer <T>', and the session cookie
    are all accepted forms,
  * the cookie is set on first valid auth and is HttpOnly+SameSite=Strict,
  * a wrong/missing token still 401s,
  * tests that don't pass auth_token (the existing suite's path) are
    unaffected — the gate is a no-op when token is None.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from krakey.plugins.dashboard.app_factory import create_app
from krakey.plugins.dashboard.auth import COOKIE_NAME, load_or_create_token


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ---- HTTP gate ----

@pytest.mark.asyncio
async def test_index_blocked_without_token_returns_auth_page():
    """Index without auth must NOT return the SPA — server-rendered
    auth page in its place. This is the security model: nothing the
    user could "click around" until a valid cookie is set."""
    app = create_app(runtime=None, auth_token="secret-abc")
    async with _client(app) as c:
        r = await c.get("/")
    assert r.status_code == 401
    assert "text/html" in r.headers.get("content-type", "")
    body = r.text
    # The auth page has the paste form; the SPA shell does not.
    assert "Sign in" in body
    assert "Krakey Dashboard" in body
    # And critically: the SPA's bundle is NOT pulled in by this page
    # so deleting elements in DevTools can't unmask a hidden SPA.
    assert "/static/app.js" not in body


@pytest.mark.asyncio
async def test_api_blocked_without_token():
    app = create_app(runtime=None, auth_token="secret-abc")
    async with _client(app) as c:
        r = await c.get("/api/health")
    assert r.status_code == 401
    body = r.json()
    assert body.get("error") == "auth required"


@pytest.mark.asyncio
async def test_api_blocked_with_wrong_token():
    app = create_app(runtime=None, auth_token="secret-abc")
    async with _client(app) as c:
        r = await c.get(
            "/api/health",
            headers={"Authorization": "Bearer wrong"},
        )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_api_passes_with_bearer_header():
    app = create_app(runtime=None, auth_token="secret-abc")
    async with _client(app) as c:
        r = await c.get(
            "/api/health",
            headers={"Authorization": "Bearer secret-abc"},
        )
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_api_passes_with_query_param():
    app = create_app(runtime=None, auth_token="secret-abc")
    async with _client(app) as c:
        r = await c.get("/api/health?token=secret-abc")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_one_click_url_sets_session_cookie():
    """`?token=` lands → cookie set → subsequent request needs only
    the cookie. This is the typical browser flow."""
    app = create_app(runtime=None, auth_token="secret-abc")
    async with _client(app) as c:
        r1 = await c.get("/?token=secret-abc")
        # Index now serves the SPA on success.
        assert r1.status_code == 200
        # Cookie attributes that the middleware sets.
        sc = r1.headers.get("set-cookie", "")
        assert COOKIE_NAME in sc
        assert "HttpOnly" in sc
        assert "SameSite=strict" in sc.lower() or "samesite=strict" in sc.lower()
        # Subsequent request with just the cookie works (no token in URL).
        r2 = await c.get("/api/health")
    assert r2.status_code == 200


@pytest.mark.asyncio
async def test_cookie_alone_authenticates_subsequent_requests():
    app = create_app(runtime=None, auth_token="secret-abc")
    async with _client(app) as c:
        r = await c.get(
            "/api/health",
            cookies={COOKIE_NAME: "secret-abc"},
        )
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_wrong_cookie_rejected():
    app = create_app(runtime=None, auth_token="secret-abc")
    async with _client(app) as c:
        r = await c.get(
            "/api/health",
            cookies={COOKIE_NAME: "wrong"},
        )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_static_assets_open_without_token():
    """CSS / favicon must still load on the auth page."""
    app = create_app(runtime=None, auth_token="secret-abc")
    async with _client(app) as c:
        r = await c.get("/style.css")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_no_token_disables_gate():
    """Tests / dev workflows that don't pass auth_token keep working
    without paying the auth tax."""
    app = create_app(runtime=None)
    async with _client(app) as c:
        r = await c.get("/api/health")
    assert r.status_code == 200


# ---- Token storage ----

def test_load_or_create_token_persists(tmp_path: Path):
    p = tmp_path / "dashboard.token"
    t1 = load_or_create_token(p)
    assert p.exists()
    assert len(t1) >= 32
    t2 = load_or_create_token(p)
    assert t1 == t2  # second call re-reads, doesn't regenerate


def test_load_or_create_token_regenerates_when_missing(tmp_path: Path):
    p = tmp_path / "dashboard.token"
    t1 = load_or_create_token(p)
    p.unlink()
    t2 = load_or_create_token(p)
    assert t1 != t2  # fresh secret on each absence


# ---- WS gate ----

class _FakeBroadcaster:
    def recent(self): return []
    def add_socket(self, _): pass
    def remove_socket(self, _): pass


def test_ws_events_blocked_without_token():
    """No token → handshake completes (accept first), then 1008 close.
    The disconnect surfaces on the first receive so the browser sees
    a real close frame with code 1008 (not abnormal-close 1006)."""
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    app = create_app(
        runtime=None,
        event_broadcaster=_FakeBroadcaster(),
        auth_token="ws-secret",
    )
    with TestClient(app) as client:
        with client.websocket_connect("/ws/events") as ws:
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
            assert exc.value.code == 1008


def test_ws_events_blocked_with_wrong_token():
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    app = create_app(
        runtime=None,
        event_broadcaster=_FakeBroadcaster(),
        auth_token="ws-secret",
    )
    with TestClient(app) as client:
        with client.websocket_connect("/ws/events?token=wrong") as ws:
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
            assert exc.value.code == 1008


def test_ws_events_accepts_with_correct_token_query():
    from starlette.testclient import TestClient

    app = create_app(
        runtime=None,
        event_broadcaster=_FakeBroadcaster(),
        auth_token="ws-secret",
    )
    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/events?token=ws-secret"
        ) as ws:
            msg = ws.receive_json()
            assert msg["kind"] == "history"


def test_ws_events_accepts_with_session_cookie():
    """Browsers attach cookies to the WS handshake automatically;
    confirm the cookie path works without any URL token."""
    from starlette.testclient import TestClient

    app = create_app(
        runtime=None,
        event_broadcaster=_FakeBroadcaster(),
        auth_token="ws-secret",
    )
    with TestClient(app, cookies={COOKIE_NAME: "ws-secret"}) as client:
        with client.websocket_connect("/ws/events") as ws:
            msg = ws.receive_json()
            assert msg["kind"] == "history"


def test_ws_events_no_token_means_no_gate():
    """auth_token=None → WS open is unguarded (test path)."""
    from starlette.testclient import TestClient

    app = create_app(
        runtime=None,
        event_broadcaster=_FakeBroadcaster(),
    )
    with TestClient(app) as client:
        with client.websocket_connect("/ws/events") as ws:
            msg = ws.receive_json()
            assert msg["kind"] == "history"
