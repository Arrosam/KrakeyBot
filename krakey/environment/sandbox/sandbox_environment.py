"""``SandboxEnvironment`` — host-side HTTP/RPC client against the
guest agent that lives inside the user's sandbox VM.

Implements the Environment Protocol; ``run`` forwards a CLI argv
to the agent's ``POST /exec`` and ``preflight`` pings ``/health``.

Wire protocol (Phase S1):

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

A fresh aiohttp session per call (no persistent pool). The
heartbeat cadence is low and we want robust recovery from a
restarted guest agent — keeping a session pinned over a VM reboot
buys nothing here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from krakey.interfaces.environment import EnvironmentUnavailableError


@dataclass
class SandboxConfig:
    agent_url: str
    agent_token: str
    guest_os: str          # informational: "linux" | "macos" | "windows"
    # Phase S1 does not manage the VM lifecycle itself; it just talks
    # to an already-running agent.


class SandboxUnavailableError(EnvironmentUnavailableError):
    """Guest agent unreachable / misconfigured. Subclass of the
    generic ``EnvironmentUnavailableError`` so callers that want to
    catch ANY env failure get this one too, while sandbox-specific
    handlers can still discriminate.
    """


class SandboxEnvironment:
    """Sandbox-VM Environment impl. Forwards CLI argv to the guest
    agent's HTTP RPC."""

    name = "sandbox"

    def __init__(self, cfg: SandboxConfig):
        self._cfg = cfg

    async def run(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        timeout: float,
        stdin: str | None = None,
    ) -> tuple[int, str, str]:
        body = {
            "cmd": list(cmd),
            # cwd inside the guest is path-encoded as a string. The
            # host's filesystem layout has no meaning inside the
            # guest, so the agent decides its own default workspace.
            # We still forward cwd so Krakey can request a sub-dir
            # under the agent workspace.
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
        return (
            int(data["exit"]),
            str(data.get("stdout", "")),
            str(data.get("stderr", "")),
        )

    async def preflight(self) -> dict[str, Any] | None:
        # Defer to the module-level preflight() helper so the same
        # logic is callable directly from tests / config validators
        # without having to construct a full SandboxEnvironment.
        from krakey.environment.sandbox.preflight import preflight
        return await preflight(self._cfg)
