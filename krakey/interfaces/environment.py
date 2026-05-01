"""Environment plugin interface — Protocol + exception types.

Sibling to ``channel.py`` / ``tool.py`` / ``modifier.py``: declares
the contract an Environment impl must satisfy. Concrete impls
(`LocalEnvironment`, `SandboxEnvironment`) live under
``krakey/environment/<name>/``; the central ``EnvironmentRouter``
that owns the plugin → env allow-list lives under
``krakey/environment/router/``.

An Environment is the **transport** a plugin uses to push CLI
commands (and, in a future iteration, GUI-control commands) out of
the host runtime into a chosen execution location — host process
("local") or sandbox VM ("sandbox"). It does NOT pick a shell or
translate code; the caller hands a fully-formed argv and gets back
``(exit, stdout, stderr)``. Shell selection / language semantics
belong to the plugin that uses the environment, not to the
environment itself.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Environment(Protocol):
    """Transport for CLI commands out of the host runtime."""

    name: str
    """Stable identifier used in config + Router lookups
    (``"local"``, ``"sandbox"``, ...)."""

    async def run(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        timeout: float,
        stdin: str | None = None,
    ) -> tuple[int, str, str]:
        """Execute a fully-formed CLI argv. Returns
        ``(exit_code, stdout, stderr)``.

        Caller hands argv as-is — the Environment does not pick a
        shell or translate language. ``cwd`` may have a different
        meaning per impl (host path for Local, guest path for
        Sandbox); document on each impl.
        """
        ...

    async def preflight(self) -> dict[str, Any] | None:
        """Optional readiness check. Local impl returns ``None``
        (nothing to check). Sandbox impl pings the guest agent and
        returns its ``/health`` payload, or raises
        ``EnvironmentUnavailableError``.
        """
        ...


class EnvironmentDenied(RuntimeError):
    """Raised by ``EnvironmentRouter.for_plugin`` when a plugin
    asks for an env it isn't allow-listed for in
    ``config.environments``. Lazy-call-time: the plugin discovers
    the denial only when it tries to use the env, so a plugin that
    never reaches its env code path never trips this — preserves
    the zero-plugin-runtime invariant.
    """


class EnvironmentUnavailableError(RuntimeError):
    """An env was reachable in config but not in reality (guest
    agent down, bad token, etc). Concrete impls (e.g.
    ``SandboxEnvironment``) raise this from ``preflight()`` /
    ``run()`` when their backend is unreachable.
    """
