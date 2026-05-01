"""Back-compat shim — the host-local runner moved.

``SubprocessRunner`` is now ``LocalEnvironment`` under
``krakey/environment/local/``. This module re-exports the new class
under the old name for one release window so callers (mostly tests)
keep working while the refactor lands. Removed in the final commit
of the environment-extraction series — at that point this whole
``krakey/sandbox/`` package goes away.

The old ``CodeRunner`` Protocol is preserved here as a type alias
for the same reason; new code should import ``Environment`` from
``krakey.interfaces.environment`` instead.
"""
from __future__ import annotations

from typing import Protocol

from krakey.environment.local import LocalEnvironment


class CodeRunner(Protocol):
    """Deprecated — use ``krakey.interfaces.environment.Environment``."""

    async def run(self, cmd, *, cwd, timeout, stdin=None) -> tuple[int, str, str]: ...


# Drop-in alias. Tests + the still-extant policy.build_code_runner
# look up ``SubprocessRunner`` by name; making it the new class
# means callers transparently get the new shape (which is a
# superset — adds ``name`` + ``preflight`` — so it satisfies the
# old Protocol too).
SubprocessRunner = LocalEnvironment


__all__ = ["CodeRunner", "SubprocessRunner"]
