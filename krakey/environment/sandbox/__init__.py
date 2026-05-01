"""Sandbox environment — VM-based transport for CLI commands.

Two halves:

  * ``SandboxEnvironment`` (host-side) — implements the Environment
    Protocol; forwards ``run`` / ``preflight`` over HTTP to the
    guest agent.
  * ``agent.py`` (guest-side) — stdlib-only HTTP server that runs
    inside the user's VM. **Not for host import** — the host code
    base never imports this module; it's a deployment artefact the
    user copies onto the VM (see ``SANDBOX.md``). Kept in this
    package only so the two halves of the sandbox are co-located
    in one folder.
"""
from krakey.environment.sandbox.sandbox_environment import (
    SandboxConfig,
    SandboxEnvironment,
    SandboxUnavailableError,
)
from krakey.environment.sandbox.preflight import preflight

__all__ = [
    "SandboxConfig",
    "SandboxEnvironment",
    "SandboxUnavailableError",
    "preflight",
]
