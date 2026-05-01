"""Sandbox preflight — one-shot ``GET /health`` against the guest
agent to verify it is reachable and authenticated before the
runtime starts dispatching real work to it.

Lives next to ``SandboxEnvironment`` (rather than on it) so config
validators and tests can call it without instantiating the full
env. ``SandboxEnvironment.preflight`` delegates here.
"""
from __future__ import annotations

import asyncio
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

    Both ``aiohttp.ClientError`` (connection refused, DNS, etc.)
    AND ``asyncio.TimeoutError`` (ClientTimeout exhaustion against
    a slow-but-alive agent) get wrapped — aiohttp surfaces the two
    families through different exception hierarchies. Without the
    explicit timeout catch the slow-agent path would escape with a
    bare TimeoutError and bypass the Router's
    "collect-all-failures" preflight aggregation.
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
    except asyncio.TimeoutError as e:
        raise SandboxUnavailableError(
            f"agent timeout at {cfg.agent_url} (>5s no response)"
        ) from e
