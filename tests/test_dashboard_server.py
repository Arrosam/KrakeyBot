"""Phase 3.F.2: FastAPI dashboard server skeleton + lifecycle."""
import asyncio

import httpx
import pytest

from src.plugins.dashboard.app_factory import create_app
from src.plugins.dashboard.server import DashboardServer


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


def test_threaded_server_stop_terminates_thread_and_frees_port():
    """The plugin uses ThreadedDashboardServer (not DashboardServer)
    because its server starts during synchronous plugin construction,
    before any asyncio loop is running. stop() must signal uvicorn to
    exit and join the daemon thread so WebChatSensory.stop() (called
    on runtime shutdown) can hand control back to the runtime once the
    server is actually down — not just rely on daemon-thread death at
    process exit."""
    import socket
    from src.plugins.dashboard.threaded_server import ThreadedDashboardServer

    server = ThreadedDashboardServer(
        create_app(runtime=None), host="127.0.0.1", port=0,
    )
    server.start()
    bound_port = server.port
    assert bound_port and bound_port > 0
    assert server._thread is not None and server._thread.is_alive()

    server.stop(timeout=5.0)

    # Thread joined → uvicorn finished its shutdown → port is free.
    assert server._thread is None
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    try:
        probe.bind(("127.0.0.1", bound_port))
    finally:
        probe.close()


def test_threaded_server_stop_idempotent_when_never_started():
    """stop() on a never-started server must be a no-op (safe to call
    from WebChatSensory.stop() in the port=0 / dashboard-disabled
    path, even though the factory already short-circuits there)."""
    from src.plugins.dashboard.threaded_server import ThreadedDashboardServer

    server = ThreadedDashboardServer(
        create_app(runtime=None), host="127.0.0.1", port=0,
    )
    # Never called start(): no thread, no uvicorn instance.
    server.stop()  # must not raise
