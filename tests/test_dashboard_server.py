"""Phase 3.F.2: FastAPI dashboard server skeleton + lifecycle."""
import asyncio

import httpx
import pytest

from src.dashboard.app_factory import create_app
from src.dashboard.server import DashboardServer


def test_app_factory_returns_fastapi_instance():
    app = create_app(runtime=None)
    # FastAPI quacks: has .openapi() and is an ASGI callable
    assert hasattr(app, "openapi")


async def test_health_endpoint_returns_ok():
    app = create_app(runtime=None)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                    base_url="http://test") as c:
        r = await c.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_app_serves_static_index_at_root():
    """The SPA should be served at GET / (HTML response, status 200)."""
    app = create_app(runtime=None)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                    base_url="http://test") as c:
        r = await c.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Krakey" in r.text


async def test_dashboard_server_start_stop_on_real_port():
    """Real-server smoke: bind to ephemeral port, hit /api/health, stop."""
    server = DashboardServer(create_app(runtime=None),
                                host="127.0.0.1", port=0)
    await server.start()
    try:
        # Server picks an ephemeral port; expose it for client
        port = server.port
        assert port and port > 0
        async with httpx.AsyncClient() as c:
            r = await c.get(f"http://127.0.0.1:{port}/api/health",
                              timeout=5.0)
        assert r.status_code == 200
    finally:
        await server.stop()


async def test_dashboard_server_port_in_use_fails_loud():
    """When the requested port is taken, .start() should raise; the
    runtime layer can then choose to continue without dashboard."""
    blocker = DashboardServer(create_app(runtime=None),
                                 host="127.0.0.1", port=0)
    await blocker.start()
    try:
        port = blocker.port
        clash = DashboardServer(create_app(runtime=None),
                                   host="127.0.0.1", port=port)
        with pytest.raises(OSError):
            await clash.start()
    finally:
        await blocker.stop()
