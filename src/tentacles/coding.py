"""Phase 3 / C: Coding Tentacle — execute Python or shell code.

**Honest disclosure**: this is **NOT a real sandbox**. The subprocess
inherits the parent's privileges and can read/write anywhere the user
running Krakey can. The `sandbox_dir` is just the working directory the
process is spawned in, and `timeout_seconds` caps wall-clock duration.

Real isolation (cgroups, seccomp, separate user, container) is left for
future work — it's heavy and OS-specific. Until then, treat this as
"convenient code execution" and don't grant Krakey to people you don't
trust on this machine.
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from src.interfaces.tentacle import Tentacle
from src.models.stimulus import Stimulus


_DEFAULT_TIMEOUT = 30.0
_DEFAULT_MAX_OUTPUT = 4000


class CodeRunner(Protocol):
    async def run(self, cmd: list[str], *, cwd: Path,
                    timeout: float, stdin: str | None = None
                    ) -> tuple[int, str, str]: ...


class SubprocessRunner:
    """Real runner using asyncio.create_subprocess_exec."""

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


class CodingTentacle(Tentacle):
    def __init__(self, runner: CodeRunner, sandbox_dir: str | Path,
                  *, timeout_seconds: float = _DEFAULT_TIMEOUT,
                  python_executable: str | None = None,
                  max_output_chars: int = _DEFAULT_MAX_OUTPUT):
        self._runner = runner
        self._sandbox = Path(sandbox_dir)
        self._timeout = timeout_seconds
        self._python = python_executable or sys.executable
        self._max_output = max_output_chars

    @property
    def name(self) -> str:
        return "coding"

    @property
    def description(self) -> str:
        return ("Execute Python (default) or shell code in a sandbox dir. "
                "Returns (returncode, stdout, stderr). NOT a real "
                "security sandbox — code runs with Krakey's privileges.")

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "language": "'python' (default) or 'shell'",
            "code": "source to execute (defaults to the natural-language intent)",
        }

    @property
    def is_internal(self) -> bool:
        # Output goes to Self for inspection, not directly to the human.
        return True

    async def execute(self, intent: str,
                        params: dict[str, Any]) -> Stimulus:
        language = (params.get("language") or "python").lower()
        code = params.get("code") or intent or ""

        self._sandbox.mkdir(parents=True, exist_ok=True)

        if language == "python":
            cmd = [self._python, "-"]      # read code from stdin
            stdin = code
        elif language == "shell":
            shell_cmd = self._shell_invocation(code)
            cmd = shell_cmd
            stdin = None
        else:
            return self._stim(
                f"Unsupported language: {language!r}. "
                "Use 'python' or 'shell'.",
            )

        try:
            rc, out, err = await self._runner.run(
                cmd, cwd=self._sandbox, timeout=self._timeout, stdin=stdin,
            )
        except asyncio.TimeoutError:
            return self._stim(
                f"Timeout after {self._timeout}s.", adrenalin=True,
            )
        except Exception as e:  # noqa: BLE001
            return self._stim(f"Runner error: {e}", adrenalin=True)

        out_t, out_truncated = _truncate(out, self._max_output)
        err_t, err_truncated = _truncate(err, self._max_output)
        suffix = ""
        if out_truncated or err_truncated:
            suffix = "\n[output truncated]"
        body = (
            f"exit={rc}\n"
            f"--- stdout ---\n{out_t}\n"
            f"--- stderr ---\n{err_t}"
            f"{suffix}"
        )
        return self._stim(body)

    def _shell_invocation(self, code: str) -> list[str]:
        if os.name == "nt":
            return ["cmd.exe", "/c", code]
        return ["/bin/sh", "-c", code]

    def _stim(self, content: str, *, adrenalin: bool = False) -> Stimulus:
        return Stimulus(
            type="tentacle_feedback",
            source=f"tentacle:{self.name}",
            content=content,
            timestamp=datetime.now(),
            adrenalin=adrenalin,
        )


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if text is None:
        return "", False
    if len(text) <= max_chars:
        return text, False
    return text[: max_chars - 1] + "…", True
