"""Sandbox host-side backend + runtime preflight integration."""
from __future__ import annotations

import threading
import time

import pytest

from src.sandbox.agent import AgentState, AgentHandler
from src.sandbox.backend import (
    SandboxConfig, SandboxRunner, SandboxUnavailableError, preflight,
)


# ---------------- real agent, loopback ----------------


@pytest.fixture
def live_agent(tmp_path):
    """Spin the stdlib agent server on an ephemeral port in a background
    thread. Yields (url, token, workspace). Stops cleanly on teardown."""
    from http.server import ThreadingHTTPServer

    class Handler(AgentHandler):
        pass

    Handler.state = AgentState(token="test-token", workspace=tmp_path / "ws")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}", "test-token", tmp_path / "ws"
    finally:
        srv.shutdown()
        srv.server_close()


async def test_preflight_reaches_live_agent(live_agent):
    url, token, _ = live_agent
    info = await preflight(SandboxConfig(
        agent_url=url, agent_token=token, guest_os="linux",
    ))
    assert info["status"] == "ok"
    assert info["agent_version"] == "1"


async def test_preflight_rejects_bad_token(live_agent):
    url, _, _ = live_agent
    with pytest.raises(SandboxUnavailableError) as ei:
        await preflight(SandboxConfig(
            agent_url=url, agent_token="WRONG", guest_os="linux",
        ))
    assert "token" in str(ei.value).lower()


async def test_preflight_raises_on_unreachable():
    # Port 1 is privileged/closed; binding fails immediately client-side
    with pytest.raises(SandboxUnavailableError):
        await preflight(SandboxConfig(
            agent_url="http://127.0.0.1:1", agent_token="x",
            guest_os="linux",
        ))


async def test_runner_exec_roundtrip(live_agent):
    url, token, ws = live_agent
    runner = SandboxRunner(SandboxConfig(
        agent_url=url, agent_token=token, guest_os="linux",
    ))
    # Use python itself — guaranteed available wherever pytest runs.
    import sys as _sys
    exit_code, out, err = await runner.run(
        [_sys.executable, "-c", "print('hi from agent')"],
        cwd=ws, timeout=10.0,
    )
    assert exit_code == 0
    assert "hi from agent" in out


async def test_runner_reports_timeout(live_agent):
    url, token, ws = live_agent
    runner = SandboxRunner(SandboxConfig(
        agent_url=url, agent_token=token, guest_os="linux",
    ))
    import sys as _sys
    # Sleep longer than the timeout
    exit_code, _out, _err = await runner.run(
        [_sys.executable, "-c", "import time; time.sleep(5)"],
        cwd=ws, timeout=0.5,
    )
    # 124 is the convention the agent returns on TimeoutExpired
    assert exit_code == 124


# ---------------- runtime preflight integration ----------------


async def test_runtime_refuses_start_when_sandbox_required_but_missing(tmp_path):
    """If coding.enabled=true and coding.sandbox=true (default) but the
    sandbox block is empty, Runtime._build_code_runner must raise."""
    from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes

    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM(), hypo_llm=ScriptedLLM(),
    )
    # Enable coding with sandbox default (true), leave runtime.config.sandbox blank
    runtime.config.plugins["coding"] = {"enabled": True, "sandbox": True}
    with pytest.raises(RuntimeError) as ei:
        runtime._build_code_runner(runtime.config.plugins["coding"])
    msg = str(ei.value)
    assert "sandbox" in msg.lower()
    assert "guest_os" in msg or "agent" in msg


async def test_runtime_allows_subprocess_when_sandbox_false(tmp_path):
    """Opting OUT of sandbox (sandbox=false) must be explicitly allowed
    so users can run coding directly on the host if they want."""
    from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes
    from src.tentacles.coding import SubprocessRunner

    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM(), hypo_llm=ScriptedLLM(),
    )
    runtime.config.plugins["coding"] = {"enabled": True, "sandbox": False}
    runner = runtime._build_code_runner(runtime.config.plugins["coding"])
    assert isinstance(runner, SubprocessRunner)


async def test_runtime_builds_sandbox_runner_with_complete_config(tmp_path):
    """sandbox=true + complete sandbox config → SandboxRunner instance."""
    from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes

    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM(), hypo_llm=ScriptedLLM(),
    )
    runtime.config.plugins["coding"] = {"enabled": True, "sandbox": True}
    runtime.config.sandbox.guest_os = "linux"
    runtime.config.sandbox.agent.url = "http://10.0.2.10:8765"
    runtime.config.sandbox.agent.token = "tok"
    runner = runtime._build_code_runner(runtime.config.plugins["coding"])
    assert isinstance(runner, SandboxRunner)


# ---------------- display mode config ----------------

def test_sandbox_display_defaults_to_headed():
    from src.models.config import _build_sandbox
    sb = _build_sandbox({})
    assert sb.display == "headed"


def test_sandbox_display_honors_user_choice():
    from src.models.config import _build_sandbox
    assert _build_sandbox({"display": "headless"}).display == "headless"
    assert _build_sandbox({"display": "HEADED"}).display == "headed"


def test_sandbox_display_invalid_falls_back_to_headed(capsys):
    from src.models.config import _build_sandbox
    sb = _build_sandbox({"display": "weird"})
    assert sb.display == "headed"
    assert "display" in capsys.readouterr().err.lower()
