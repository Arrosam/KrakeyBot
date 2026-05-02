"""Token-gate auth on /api/* and /ws/*.

Confirms:
  * static UI shell ('/' and /static/*) stays open so the paste-token
    form can render without a token,
  * /api/* returns 401 without a token,
  * Authorization: Bearer <T> and ?token=<T> both let a request through,
  * a wrong token still 401s,
  * tests that don't pass auth_token (the existing suite's path) are
    unaffected — the gate is a no-op when token is None.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from krakey.plugins.dashboard.app_factory import create_app
from krakey.plugins.dashboard.auth import load_or_create_token


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


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
async def test_static_shell_open_without_token():
    """Index page must render without a token so the paste-token
    modal can come up. Same for /style.css and /static/*."""
    app = create_app(runtime=None, auth_token="secret-abc")
    async with _client(app) as c:
        r1 = await c.get("/")
        r2 = await c.get("/style.css")
    assert r1.status_code == 200
    assert r2.status_code == 200


@pytest.mark.asyncio
async def test_no_token_disables_gate():
    """Tests / dev workflows that don't pass auth_token keep working
    without paying the auth tax."""
    app = create_app(runtime=None)
    async with _client(app) as c:
        r = await c.get("/api/health")
    assert r.status_code == 200


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


# ---- WS auth ----

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


def test_ws_events_accepts_with_correct_token():
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
