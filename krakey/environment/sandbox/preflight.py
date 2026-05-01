"""Sandbox preflight — one-shot ``GET /health`` against the guest
agent to verify it is reachable and authenticated before the
runtime starts dispatching real work to it.

Lives next to ``SandboxEnvironment`` (rather than on it) so config
validators and tests can call it without instantiating the full
env. ``SandboxEnvironment.preflight`` delegates here.
"""
from __future__ import annotations

from typing import Any

import aiohttp

from krakey.environment.sandbox.sandbox_environment import (
    SandboxConfig,
    SandboxUnavailableError,
)


async def preflight(cfg: SandboxConfig) -> dict[str, Any]:
    """Verify the guest agent is reachable + authenticated.

    Returns the agent's ``/health`` payload on success. Raises
    ``SandboxUnavailableError`` (subclass of
    ``EnvironmentUnavailableError``) on any failure.
    """
    url = cfg.agent_url.rstrip("/") + "/health"
    headers = {"X-Krakey-Token": cfg.agent_token}
    timeout = aiohttp.ClientTimeout(total=5.0)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(url, headers=headers) as r:
                if r.status == 401:
                    raise SandboxUnavailableError(
                        "agent rejected the token; check "
                        "sandbox.agent.token"
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
