"""Edge tests for the Phase D Docker sandbox provider:

  * ``DockerSandboxEnvironment.run`` / ``.preflight`` — wire protocol
    identical to ``SandboxEnvironment`` (drives a live stdlib agent).
  * ``ensure_container_running`` — full auto-start branch matrix via
    injected probe/daemon/run helpers (no real Docker).
  * ``DockerSandboxEnvironment.__init__`` auto_start non-fatality.

Techniques: positive/equivalence (roundtrip, already-up), boundary
(timeout, wait-for-port deadline), state transition (down→run→up),
negative (no docker / daemon down / run fails → False, never raises).
"""
from __future__ import annotations

import threading

import pytest

from krakey.environment.sandbox.agent import AgentState, AgentHandler
from krakey.environment.sandbox import (
    SandboxUnavailableError,
    DockerSandboxConfig,
    DockerSandboxEnvironment,
)
from krakey.environment.sandbox import docker_environment as de


# ---------------- real agent, loopback (provider-agnostic) ----------------

@pytest.fixture
def live_agent(tmp_path):
    from http.server import ThreadingHTTPServer

    class Handler(AgentHandler):
        pass

    Handler.state = AgentState(token="test-token", workspace=tmp_path / "ws")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}", "test-token", tmp_path / "ws", port
    finally:
        srv.shutdown()
        srv.server_close()


def _cfg(url, token, *, host_port=18765, **kw):
    return DockerSandboxConfig(
        agent_url=url, agent_token=token, guest_os="linux",
        image="krakey/sandbox:latest", container_name="krakey-sandbox",
        host_port=host_port, **kw,
    )


# ---------------- run() / preflight() wire protocol ----------------

class TestWireProtocol:
    async def test_preflight_reaches_live_agent(self, live_agent):
        url, token, _ws, _port = live_agent
        env = DockerSandboxEnvironment(_cfg(url, token))
        info = await env.preflight()
        assert info["status"] == "ok"
        assert info["agent_version"] == "1"

    async def test_preflight_rejects_bad_token(self, live_agent):
        url, _token, _ws, _port = live_agent
        env = DockerSandboxEnvironment(_cfg(url, "WRONG"))
        with pytest.raises(SandboxUnavailableError) as ei:
            await env.preflight()
        assert "token" in str(ei.value).lower()

    async def test_run_exec_roundtrip(self, live_agent):
        url, token, ws, _port = live_agent
        env = DockerSandboxEnvironment(_cfg(url, token))
        import sys as _sys
        exit_code, out, _err = await env.run(
            [_sys.executable, "-c", "print('hi from docker agent')"],
            cwd=ws, timeout=10.0,
        )
        assert exit_code == 0
        assert "hi from docker agent" in out

    async def test_run_unreachable_raises(self):
        env = DockerSandboxEnvironment(_cfg("http://127.0.0.1:1", "x"))
        import sys as _sys
        with pytest.raises(SandboxUnavailableError):
            await env.run([_sys.executable, "-c", "pass"], cwd=None, timeout=2.0)

    def test_name_is_sandbox(self):
        env = DockerSandboxEnvironment(_cfg("http://x:8765", "t"))
        assert env.name == "sandbox"


# ---------------- ensure_container_running branch matrix ----------------

class TestEnsureContainerRunning:
    def _cfg(self, **kw):
        return _cfg("http://127.0.0.1:18765", "t", **kw)

    def test_already_up_short_circuits(self):
        calls = {"run": 0}
        ok = de.ensure_container_running(
            self._cfg(),
            probe=lambda h, p: True,  # already reachable
            run_container=lambda cfg: calls.__setitem__("run", calls["run"] + 1) or True,
        )
        assert ok is True
        assert calls["run"] == 0  # never tried to start

    def test_no_docker_on_path_returns_false(self):
        ok = de.ensure_container_running(
            self._cfg(),
            probe=lambda h, p: False,
            docker_on_path=lambda: False,
        )
        assert ok is False

    def test_daemon_down_returns_false(self):
        ok = de.ensure_container_running(
            self._cfg(),
            probe=lambda h, p: False,
            docker_on_path=lambda: True,
            docker_daemon_alive=lambda: False,
        )
        assert ok is False

    def test_container_already_running_waits_for_port(self):
        ok = de.ensure_container_running(
            self._cfg(),
            probe=lambda h, p: False,
            docker_on_path=lambda: True,
            docker_daemon_alive=lambda: True,
            container_running=lambda name: True,
            wait_for_port=lambda h, p, d: True,
        )
        assert ok is True

    def test_down_then_run_then_up(self):
        """State transition: not running → docker run → port comes up."""
        ran = {"v": False}

        def run_container(cfg):
            ran["v"] = True
            return True

        ok = de.ensure_container_running(
            self._cfg(),
            probe=lambda h, p: False,
            docker_on_path=lambda: True,
            docker_daemon_alive=lambda: True,
            container_running=lambda name: False,
            run_container=run_container,
            wait_for_port=lambda h, p, d: True,
        )
        assert ok is True
        assert ran["v"] is True

    def test_run_fails_returns_false(self):
        ok = de.ensure_container_running(
            self._cfg(),
            probe=lambda h, p: False,
            docker_on_path=lambda: True,
            docker_daemon_alive=lambda: True,
            container_running=lambda name: False,
            run_container=lambda cfg: False,  # docker run failed
        )
        assert ok is False

    def test_run_succeeds_but_port_never_comes_up(self):
        ok = de.ensure_container_running(
            self._cfg(),
            probe=lambda h, p: False,
            docker_on_path=lambda: True,
            docker_daemon_alive=lambda: True,
            container_running=lambda name: False,
            run_container=lambda cfg: True,
            wait_for_port=lambda h, p, d: False,  # boot timed out
        )
        assert ok is False


# ---------------- __init__ auto_start non-fatality ----------------

class TestInitAutoStart:
    def test_auto_start_failure_does_not_raise(self, monkeypatch):
        """auto_start=True + everything fails → __init__ must NOT raise
        (graceful disable; preflight de-registers later)."""
        monkeypatch.setattr(de, "_probe", lambda h, p, timeout=0.5: False)
        monkeypatch.setattr(de, "_has_docker", lambda: False)
        env = DockerSandboxEnvironment(_cfg(
            "http://127.0.0.1:18765", "t", auto_start=True,
        ))
        assert env.name == "sandbox"  # constructed despite failure

    def test_auto_start_false_skips_lifecycle(self, monkeypatch):
        """auto_start=False must never touch the docker lifecycle."""
        called = {"v": False}
        monkeypatch.setattr(
            de, "ensure_container_running",
            lambda cfg, **kw: called.__setitem__("v", True) or True,
        )
        DockerSandboxEnvironment(_cfg("http://x:8765", "t", auto_start=False))
        assert called["v"] is False
