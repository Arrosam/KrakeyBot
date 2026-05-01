"""``LocalEnvironment`` — run CLI commands as a subprocess of the
host runtime, with the runtime user's privileges.

This is **NOT a sandbox**. The child inherits the parent process's
permissions, environment, and filesystem access. The Router's
allow-list is the only barrier between an enabled plugin and host
state, so reserve ``local`` for plugins that genuinely need host
access (or the user explicitly opted out of sandboxing for a
plugin that supports both).

Use ``SandboxEnvironment`` (krakey/environment/sandbox/) when the
plugin should actually be isolated.

Carried over verbatim from the deleted ``SubprocessRunner`` —
same ``asyncio.create_subprocess_exec`` + timeout-kill pattern.
The only contract change is the ``name`` attribute and the
``preflight`` no-op required by the Environment Protocol.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


class LocalEnvironment:
    """Host-local Environment impl. Runs the given argv directly via
    ``asyncio.create_subprocess_exec`` — no shell, no translation."""

    name = "local"

    async def run(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        timeout: float,
        stdin: str | None = None,
    ) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out_b, err_b = await asyncio.wait_for(
                proc.communicate(input=stdin.encode() if stdin else None),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await proc.communicate()
            except Exception:  # noqa: BLE001
                pass
            raise
        rc = proc.returncode if proc.returncode is not None else -1
        out = out_b.decode("utf-8", errors="replace")
        err = err_b.decode("utf-8", errors="replace")
        return rc, out, err

    async def preflight(self) -> dict[str, Any] | None:
        # Nothing to check: if the host process is alive enough to
        # call this, subprocess creation will work.
        return None
