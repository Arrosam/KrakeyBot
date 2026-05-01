"""Sandbox host-side backend + runtime preflight integration."""
from __future__ import annotations

import threading
import time

import pytest

from krakey.environment.sandbox.agent import AgentState, AgentHandler
from krakey.environment.sandbox import (
    SandboxConfig, SandboxEnvironment, SandboxUnavailableError, preflight,
)

# Old name stayed at the import site; bind it locally for the
# legacy assertions until commit 5 rewrites them against the Router.
SandboxRunner = SandboxEnvironment


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


def test_router_build_refuses_when_sandbox_partially_configured(tmp_path):
    """Half-filled sandbox block (guest_os without agent fields, or
    vice versa) is the most common config typo — silently downgrading
    to "no sandbox" would leave the user wondering why their plugin
    can't reach the env. Construction-time refusal forces them to
    fix it on the spot."""
    from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes

    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM(), hypo_llm=ScriptedLLM(),
    )
    # Partial: guest_os set, agent fields still empty — Router build
    # should refuse on the next compose attempt.
    runtime.config.sandbox.guest_os = "linux"
    with pytest.raises(RuntimeError) as ei:
        runtime._build_environment_router()
    msg = str(ei.value)
    assert "sandbox" in msg.lower()
    assert "agent" in msg


async def test_router_local_always_present_even_with_no_sandbox(tmp_path):
    """LocalEnvironment has no config and no failure mode — the
    Router always exposes it so plugins that don't need isolation
    can run regardless of sandbox VM state."""
    from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes

    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM(), hypo_llm=ScriptedLLM(),
    )
    # Default: no sandbox config → only Local env registered.
    assert runtime.environment_router.env_names() == ["local"]


async def test_router_registers_sandbox_env_when_config_complete(tmp_path):
    """Complete sandbox config → Router exposes both 'local' and
    'sandbox'. Allow-list is empty until config.environments lands;
    that's fine — no plugin asks for either env yet."""
    from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes

    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM(), hypo_llm=ScriptedLLM(),
    )
    runtime.config.sandbox.guest_os = "linux"
    runtime.config.sandbox.agent.url = "http://10.0.2.10:8765"
    runtime.config.sandbox.agent.token = "tok"
    rebuilt = runtime._build_environment_router()
    assert set(rebuilt.env_names()) == {"local", "sandbox"}


# ---------------- display mode config ----------------

def test_sandbox_display_defaults_to_headed():
    from krakey.models.config import _build_sandbox
    sb = _build_sandbox({})
    assert sb.display == "headed"


def test_sandbox_display_honors_user_choice():
    from krakey.models.config import _build_sandbox
    assert _build_sandbox({"display": "headless"}).display == "headless"
    assert _build_sandbox({"display": "HEADED"}).display == "headed"


def test_sandbox_display_invalid_falls_back_to_headed(capsys):
    from krakey.models.config import _build_sandbox
    sb = _build_sandbox({"display": "weird"})
    assert sb.display == "headed"
    assert "display" in capsys.readouterr().err.lower()
