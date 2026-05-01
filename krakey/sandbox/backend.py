"""Back-compat shim — host-side sandbox runner moved.

``SandboxRunner`` is now ``SandboxEnvironment`` under
``krakey/environment/sandbox/``. This module re-exports the new
class under the old name for the duration of the environment-
extraction series. Removed in the final commit at the same time
as the rest of ``krakey/sandbox/``.

New code should import from ``krakey.environment.sandbox`` directly.
"""
from __future__ import annotations

from krakey.environment.sandbox import (
    SandboxConfig,
    SandboxEnvironment,
    SandboxUnavailableError,
    preflight,
)

# Drop-in alias so policy.build_code_runner + tests still resolve
# ``SandboxRunner``. Same class, gained ``name`` + ``preflight()``.
SandboxRunner = SandboxEnvironment


__all__ = [
    "SandboxConfig",
    "SandboxRunner",
    "SandboxUnavailableError",
    "preflight",
]
