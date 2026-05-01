"""Local subprocess runner — used when `coding.sandbox` is false.

Defines the `CodeRunner` Protocol (also implemented by
`sandbox.backend.SandboxRunner` for the VM case) and the plain
`SubprocessRunner` that shells out on the host. Lives in `krakey/sandbox/`
because that's where all code-execution backends live, even though
`SubprocessRunner` doesn't sandbox anything.

Runtime picks one of the two at plugin-load time via
`Runtime._build_code_runner(coding_cfg)` and hands the result to the
coding plugin's factory.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol


class CodeRunner(Protocol):
    async def run(self, cmd: list[str], *, cwd: Path,
                    timeout: float, stdin: str | None = None
                    ) -> tuple[int, str, str]: ...


class SubprocessRunner:
    """Host-local runner using `asyncio.create_subprocess_exec`.

    Honest disclosure: this is NOT a sandbox. The child inherits the
    parent's privileges. Use `SandboxRunner` (krakey/sandbox/backend.py)
    when you actually need isolation.
    """

    async def run(self, cmd: list[str], *, cwd: Path,
                    timeout: float, stdin: str | None = None
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
