"""Built-in `coding` plugin — run Python / shell via subprocess or sandbox.

Runner selection (local Subprocess vs sandbox-agent SandboxRunner) is
Runtime policy: the factory calls `deps["build_code_runner"](config)`
to get the right one. If sandbox is requested but unconfigured, that
call raises — loader captures it as the plugin error.

**Honest disclosure**: with `sandbox: false` this is **NOT a real
sandbox**. The subprocess inherits the parent's privileges and can
read/write anywhere the user running Krakey can. The `sandbox_dir` is
just the working directory the process is spawned in, and
`timeout_seconds` caps wall-clock duration. Real isolation (cgroups,
seccomp, separate user, container) is the job of the sandbox VM
runner — enable `sandbox: true` and configure the top-level
`sandbox:` block for that.
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.interfaces.tentacle import Tentacle
from src.models.stimulus import Stimulus
from src.sandbox.subprocess_runner import CodeRunner


_DEFAULT_TIMEOUT = 30.0
_DEFAULT_MAX_OUTPUT = 4000


MANIFEST = {
    "name": "coding",
    "description": "Execute a Python or shell command and return exit "
                   "code + stdout / stderr. Routes through the sandbox "
                   "VM when `sandbox: true` (default).",
    "is_internal": True,
    "config_schema": [
        {"field": "sandbox",          "type": "bool",   "default": True,
         "help": "When true, exec runs via the sandbox guest agent. "
                 "Set to false only on trusted hosts."},
        {"field": "sandbox_dir",      "type": "text",
         "default": "workspace/sandbox",
         "help": "Working directory hint passed to the runner."},
        {"field": "timeout_seconds",  "type": "number", "default": 30,
         "help": "Subprocess timeout. Exceeding it returns exit=124."},
        {"field": "max_output_chars", "type": "number", "default": 4000,
         "help": "stdout / stderr truncated past this many chars."},
    ],
}


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


def build_tentacle(ctx) -> Tentacle:
    """Unified-format factory (Phase 2). Pulls build_code_runner
    from ctx.services; pulls sandbox / timeout / output-cap
    from ctx.config."""
    build_runner = ctx.services.get("build_code_runner")
    if build_runner is None:
        raise RuntimeError(
            "coding plugin needs services['build_code_runner'] "
            "(Runtime._build_code_runner callable)."
        )
    runner = build_runner(ctx.config)
    return CodingTentacle(
        runner=runner,
        sandbox_dir=str(ctx.config.get("sandbox_dir", "workspace/sandbox")),
        timeout_seconds=int(ctx.config.get("timeout_seconds", 30)),
        max_output_chars=int(ctx.config.get("max_output_chars", 4000)),
    )
