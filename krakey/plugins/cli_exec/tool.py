"""``cli_exec`` Tool — run argv in a Self-chosen Environment.

The tool's factory captures the per-plugin ``ctx.environment``
accessor so ``execute()`` can resolve env names at call time without
holding a Runtime reference. Every failure mode (denial, timeout,
unavailability, generic subprocess error, malformed params) returns
an error ``Stimulus`` rather than raising — additive-plugin invariant
per CLAUDE.md.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from krakey.interfaces.environment import (
    EnvironmentDenied,
    EnvironmentUnavailableError,
)
from krakey.interfaces.tool import Tool
from krakey.models.stimulus import Stimulus

if TYPE_CHECKING:
    from krakey.interfaces.environment import Environment
    from krakey.interfaces.plugin_context import PluginContext


DEFAULT_TIMEOUT_S = 30.0
"""Wall-clock cap when Self omits ``timeout_s``. Same order of
magnitude as a typical interactive shell command; a long-running
build should pass an explicit larger value."""

OUTPUT_TRUNCATE_CHARS = 4000
"""Per-stream cap on stdout / stderr in the returned Stimulus. Bounds
prompt growth from a chatty subprocess; the truncation marker tells
Self the output was cut so it can re-run with redirection if it
needs the full body."""


def build_tool(ctx: "PluginContext") -> "CliExecTool":
    """Factory for the single ``tool`` component declared in
    ``meta.yaml``. Captures the per-plugin env resolver so the tool
    can dispatch by env name at call time."""
    return CliExecTool(env_resolver=ctx.environment)


class CliExecTool(Tool):
    """Self-facing tool that runs argv in the requested Environment."""

    def __init__(
        self, env_resolver: Callable[[str], "Environment"],
    ):
        self._env_resolver = env_resolver

    @property
    def name(self) -> str:
        return "cli_exec"

    @property
    def description(self) -> str:
        return (
            "Run a CLI command in a target Environment. `env` selects "
            "the Environment by name (e.g. \"local\" or \"sandbox\"); "
            "the plugin must be allow-listed for that env in config. "
            "`cmd` is the fully-formed argv list (no shell expansion "
            "— pass [\"bash\", \"-c\", \"...\"] yourself if you need a "
            "shell). `cwd` (optional, default \".\") is the working "
            "directory inside the env. `timeout_s` (optional, default "
            f"{int(DEFAULT_TIMEOUT_S)}) caps wall-clock time. `stdin` "
            "(optional) is piped to the process's stdin. Returns "
            "exit_code + stdout + stderr; long output is truncated."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["env", "cmd"],
            "properties": {
                "env": {
                    "type": "string",
                    "description": (
                        "Environment name; e.g. \"local\" or "
                        "\"sandbox\"."
                    ),
                },
                "cmd": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Fully-formed argv list. No shell expansion; "
                        "pass an explicit shell wrapper if you need "
                        "globs, pipes, or redirection."
                    ),
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        "Working directory inside the env "
                        "(default \".\")."
                    ),
                },
                "timeout_s": {
                    "type": "number",
                    "description": (
                        "Wall-clock timeout in seconds "
                        f"(default {int(DEFAULT_TIMEOUT_S)})."
                    ),
                },
                "stdin": {
                    "type": "string",
                    "description": (
                        "Optional text piped to the process's stdin."
                    ),
                },
            },
        }

    async def execute(
        self, intent: str, params: dict[str, Any],
    ) -> Stimulus:
        env_name = params.get("env")
        if not isinstance(env_name, str) or not env_name:
            return self._err("missing or invalid `env` parameter")

        cmd = params.get("cmd")
        if (
            not isinstance(cmd, list)
            or not cmd
            or not all(isinstance(c, str) for c in cmd)
        ):
            return self._err(
                "`cmd` must be a non-empty list of strings",
            )

        cwd_raw = params.get("cwd", ".")
        if not isinstance(cwd_raw, str) or not cwd_raw:
            return self._err("`cwd` must be a non-empty string")

        timeout_raw = params.get("timeout_s", DEFAULT_TIMEOUT_S)
        if (
            not isinstance(timeout_raw, (int, float))
            or isinstance(timeout_raw, bool)
            or timeout_raw <= 0
        ):
            return self._err(
                "`timeout_s` must be a positive number",
            )

        stdin = params.get("stdin")
        if stdin is not None and not isinstance(stdin, str):
            return self._err("`stdin` must be a string when provided")

        try:
            env = self._env_resolver(env_name)
        except EnvironmentDenied as e:
            return self._err(
                f"environment {env_name!r} denied: {e}",
            )
        except Exception as e:  # noqa: BLE001
            return self._err(
                f"env resolver error: {type(e).__name__}: {e}",
            )

        try:
            rc, out, err = await env.run(
                list(cmd),
                cwd=Path(cwd_raw),
                timeout=float(timeout_raw),
                stdin=stdin,
            )
        except asyncio.TimeoutError:
            return self._err(
                f"timed out after {float(timeout_raw)}s running "
                f"{list(cmd)!r} in env {env_name!r}",
            )
        except EnvironmentUnavailableError as e:
            return self._err(
                f"environment {env_name!r} unavailable: {e}",
            )
        except Exception as e:  # noqa: BLE001
            return self._err(
                f"env.run error: {type(e).__name__}: {e}",
            )

        out_t = _truncate(out, OUTPUT_TRUNCATE_CHARS)
        err_t = _truncate(err, OUTPUT_TRUNCATE_CHARS)
        content = (
            f"cli_exec env={env_name} rc={rc}\n"
            f"--- stdout ---\n{out_t}\n"
            f"--- stderr ---\n{err_t}"
        )
        return Stimulus(
            type="tool_feedback",
            source=f"tool:{self.name}",
            content=content,
            timestamp=datetime.now(),
            adrenalin=False,
        )

    def _err(self, msg: str) -> Stimulus:
        return Stimulus(
            type="tool_feedback",
            source=f"tool:{self.name}",
            content=f"cli_exec error: {msg}",
            timestamp=datetime.now(),
            adrenalin=False,
        )


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n...[truncated, total {len(s)} chars]"
