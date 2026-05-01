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
    """Half-filled environments.sandbox block (guest_os without
    agent fields, or vice versa) is the most common config typo —
    silently downgrading to "no sandbox" would leave the user
    wondering why their plugin can't reach the env. Construction-
    time refusal forces them to fix it on the spot."""
    from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes
    from krakey.models.config import SandboxEnvironmentConfig

    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM(), hypo_llm=ScriptedLLM(),
    )
    # Partial: guest_os set, agent fields still empty.
    runtime.config.environments.sandbox = SandboxEnvironmentConfig(
        guest_os="linux",
    )
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
    # Default: no environments.sandbox → only Local env registered.
    assert runtime.environment_router.env_names() == ["local"]


async def test_router_registers_sandbox_env_when_config_complete(tmp_path):
    """Complete environments.sandbox → Router exposes both 'local'
    and 'sandbox', and the allow-list reflects the configured
    plugin assignments."""
    from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes
    from krakey.models.config import SandboxAgentSection, SandboxEnvironmentConfig

    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM(), hypo_llm=ScriptedLLM(),
    )
    runtime.config.environments.sandbox = SandboxEnvironmentConfig(
        allowed_plugins=["coding"],
        guest_os="linux",
        agent=SandboxAgentSection(
            url="http://10.0.2.10:8765", token="tok",
        ),
    )
    rebuilt = runtime._build_environment_router()
    assert set(rebuilt.env_names()) == {"local", "sandbox"}
    # Allow-list flows through from config.environments.
    assert rebuilt.for_plugin("coding", "sandbox") is rebuilt._envs["sandbox"]


# ---------------- display mode config ----------------

def test_sandbox_display_defaults_to_headed():
    from krakey.models.config.environments import _build_sandbox_env
    sb = _build_sandbox_env({})
    assert sb.display == "headed"


def test_sandbox_display_honors_user_choice():
    from krakey.models.config.environments import _build_sandbox_env
    assert _build_sandbox_env({"display": "headless"}).display == "headless"
    assert _build_sandbox_env({"display": "HEADED"}).display == "headed"


def test_sandbox_display_invalid_falls_back_to_headed(capsys):
    from krakey.models.config.environments import _build_sandbox_env
    sb = _build_sandbox_env({"display": "weird"})
    assert sb.display == "headed"
    assert "display" in capsys.readouterr().err.lower()


# ---------------- environments: top-level shape guard ----------------

@pytest.mark.parametrize("bad", [
    "just_a_string",       # failed env-var template, etc.
    [1, 2, 3],             # user wrote a list by mistake
    ["local", "sandbox"],  # confused list-of-keys form
    42,                    # YAML scalar instead of mapping
])
def test_build_environments_tolerates_non_mapping_top_level(bad, capsys):
    """A typo in ``environments:`` shouldn't hard-fail boot. Warn,
    fall back to defaults — same forgiving pattern as
    environments.local / environments.sandbox at the inner level."""
    from krakey.models.config.environments import _build_environments

    out = _build_environments(bad)
    # Defaults — Local present with empty allow-list, no sandbox.
    assert out.sandbox is None
    assert out.local.allowed_plugins == []
    err = capsys.readouterr().err.lower()
    assert "environments" in err
    assert "mapping" in err


@pytest.mark.parametrize("bad_subblock", [
    {"resources": [1, 2]},   # list where dict expected
    {"agent": "oops"},       # string where dict expected
    {"resources": 42},       # scalar where dict expected
])
def test_build_sandbox_env_tolerates_non_mapping_subblocks(bad_subblock, capsys):
    """Same non-mapping-input class of bug as the top-level fix —
    apply consistently to environments.sandbox.resources +
    environments.sandbox.agent so a typo in either sub-block warns
    rather than crashing AttributeError at startup."""
    from krakey.models.config.environments import _build_sandbox_env

    sb = _build_sandbox_env(bad_subblock)
    # Defaults survive — resources/agent fall back to their dataclass
    # defaults rather than crashing.
    assert sb.resources.cpu == 2
    assert sb.agent.url == ""
    err = capsys.readouterr().err.lower()
    assert "mapping" in err


# ---------------- preflight error wrapping ----------------

async def test_preflight_wraps_aiohttp_timeout_as_sandbox_error(tmp_path):
    """ClientTimeout exhaustion against a slow-but-alive agent
    raises bare asyncio.TimeoutError out of aiohttp — NOT a
    subclass of aiohttp.ClientError. Without an explicit catch the
    Router's preflight_all aggregator sees an unexpected exception
    type and bypasses the "collect all failures" pattern.
    """
    import asyncio as _asyncio
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from krakey.environment.sandbox import (
        SandboxConfig, SandboxUnavailableError,
    )
    from krakey.environment.sandbox.preflight import preflight as _preflight

    class _SlowHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            time.sleep(20)  # > 5s ClientTimeout
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a, **k):  # silence
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        cfg = SandboxConfig(
            agent_url=f"http://127.0.0.1:{port}",
            agent_token="x", guest_os="linux",
        )
        with pytest.raises(SandboxUnavailableError) as ei:
            await _preflight(cfg)
        # Helpful message naming the timeout cause, not bare TimeoutError.
        assert "timeout" in str(ei.value).lower()
    finally:
        srv.shutdown()
        srv.server_close()


# ---------------- old top-level sandbox: deprecation ----------------

def test_old_top_level_sandbox_block_emits_deprecation(tmp_path, capsys):
    """A user upgrading carries the old `sandbox:` block in their
    config.yaml. We don't auto-translate (the new shape adds an
    allow-list concept the old shape doesn't have); just nudge with
    a single stderr line so the upgrade is visible."""
    from krakey.models.config import load_config

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "llm:\n  providers: {}\n  tags: {}\n  core_purposes: {}\n"
        "plugins: []\n"
        "graph_memory:\n  db_path: ':memory:'\n"
        "sandbox:\n"
        "  guest_os: linux\n"
        "  agent:\n    url: http://x\n    token: t\n",
        encoding="utf-8",
    )
    load_config(str(cfg_path))
    err = capsys.readouterr().err.lower()
    assert "deprecated" in err
    assert "sandbox" in err
    assert "environments.sandbox" in err
