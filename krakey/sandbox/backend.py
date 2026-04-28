"""Host-side sandbox backend: HTTP RPC client against the guest agent.

Provides a `SandboxRunner` with a `CodeRunner`-compatible interface so
the coding tentacle can swap between local Subprocess and Sandbox
transparently.

Protocol with agent (Phase S1):

    POST  <agent_url>/exec
    Headers: X-Krakey-Token: <shared secret>
    Body (json):
        {
          "cmd":     ["python3", "-c", "print('hi')"],
          "cwd":     "/home/krakey/work",      # optional
          "timeout": 30.0,
          "stdin":   null                      # optional str
        }
    Response (json):
        {"exit": 0, "stdout": "hi\\n", "stderr": ""}

    GET  <agent_url>/health
        {"status": "ok", "guest_os": "linux", "agent_version": "1"}

All RPC uses an HTTP client that does not pool connections persistently
(aiohttp session per call), since heartbeat cadence is low and we want
robust recovery from a restarted guest agent.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp


@dataclass
class SandboxConfig:
    agent_url: str
    agent_token: str
    guest_os: str          # informational: "linux" | "macos" | "windows"
    # Phase S1 does not manage the VM lifecycle itself; it just talks
    # to an already-running agent.


class SandboxUnavailableError(RuntimeError):
    """Raised when the guest agent cannot be reached / is misconfigured
    while a sandboxed tentacle is enabled. Runtime should refuse to
    start in that case."""


class SandboxRunner:
    """Drop-in replacement for tentacles.coding.SubprocessRunner.

    Matches the CodeRunner Protocol:
        async def run(cmd, *, cwd, timeout, stdin=None) -> (exit, out, err)
    """

    def __init__(self, cfg: SandboxConfig):
        self._cfg = cfg

    async def run(self, cmd: list[str], *, cwd: Path,
                    timeout: float, stdin: str | None = None
                    ) -> tuple[int, str, str]:
        body = {
            "cmd": list(cmd),
            # cwd inside the guest is path-encoded as string; the host's
            # sandbox_dir has no meaning inside the guest, so the agent
            # decides its own default workspace. We still pass cwd so
            # Krakey can request a sub-dir under the agent workspace.
            "cwd": str(cwd) if cwd else None,
            "timeout": timeout,
            "stdin": stdin,
        }
        headers = {"X-Krakey-Token": self._cfg.agent_token}
        url = self._cfg.agent_url.rstrip("/") + "/exec"
        timeout_obj = aiohttp.ClientTimeout(total=timeout + 10)
        async with aiohttp.ClientSession(timeout=timeout_obj) as s:
            async with s.post(url, json=body, headers=headers) as r:
                if r.status != 200:
                    text = await r.text()
                    raise SandboxUnavailableError(
                        f"agent returned {r.status}: {text[:200]}"
                    )
                data = await r.json()
        return int(data["exit"]), str(data.get("stdout", "")), str(data.get("stderr", ""))


async def preflight(cfg: SandboxConfig) -> dict[str, Any]:
    """Verify the guest agent is reachable and authenticated.

    Returns the agent's /health payload on success. Raises
    SandboxUnavailableError otherwise.
    """
    url = cfg.agent_url.rstrip("/") + "/health"
    headers = {"X-Krakey-Token": cfg.agent_token}
    timeout = aiohttp.ClientTimeout(total=5.0)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(url, headers=headers) as r:
                if r.status == 401:
                    raise SandboxUnavailableError(
                        "agent rejected the token; check sandbox.agent.token"
                    )
                if r.status != 200:
                    raise SandboxUnavailableError(
                        f"agent /health returned {r.status}"
                    )
                return await r.json()
    except aiohttp.ClientError as e:
        raise SandboxUnavailableError(
            f"agent unreachable at {cfg.agent_url}: {e}"
        ) from e
