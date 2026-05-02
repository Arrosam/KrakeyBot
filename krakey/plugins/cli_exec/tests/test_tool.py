"""Unit tests for ``cli_exec`` Tool.

Run from repo root:

    pytest krakey/plugins/cli_exec

(``pytest.ini`` fixes ``testpaths = tests``, so the in-plugin tests
are not auto-discovered by a bare ``pytest`` invocation. Per-plugin
explicit path is the contract.)
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

import pytest

from krakey.interfaces.environment import (
    EnvironmentDenied,
    EnvironmentUnavailableError,
)
from krakey.plugins.cli_exec.tool import (
    CliExecTool,
    DEFAULT_TIMEOUT_S,
    OUTPUT_TRUNCATE_CHARS,
)


# --------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------


class FakeEnv:
    """Minimal Environment stub. Records the most recent call's
    arguments and returns a scripted ``(rc, out, err)`` triple — or
    raises a scripted exception."""

    name = "local"

    def __init__(
        self,
        result: tuple[int, str, str] | None = None,
        raises: BaseException | None = None,
    ):
        self._result = result if result is not None else (0, "", "")
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    async def run(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        timeout: float,
        stdin: str | None = None,
    ) -> tuple[int, str, str]:
        self.calls.append({
            "cmd": list(cmd), "cwd": cwd, "timeout": timeout,
            "stdin": stdin,
        })
        if self._raises is not None:
            raise self._raises
        return self._result

    async def preflight(self):  # noqa: D401
        return None


def _resolver_returning(env: FakeEnv) -> Callable[[str], FakeEnv]:
    return lambda _name: env


def _resolver_denying(msg: str = "not allow-listed") -> Callable[[str], Any]:
    def _raise(_name: str):
        raise EnvironmentDenied(msg)
    return _raise


# --------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------


async def test_happy_path_returns_tool_feedback_stimulus():
    env = FakeEnv(result=(0, "ok\n", ""))
    tool = CliExecTool(env_resolver=_resolver_returning(env))

    s = await tool.execute(
        "say hi", {"env": "local", "cmd": ["echo", "ok"]},
    )

    assert s.type == "tool_feedback"
    assert s.source == "tool:cli_exec"
    assert "rc=0" in s.content
    assert "env=local" in s.content
    assert "ok" in s.content
    assert s.adrenalin is False
    # Default cwd + timeout + stdin propagated:
    assert env.calls == [{
        "cmd": ["echo", "ok"],
        "cwd": Path("."),
        "timeout": DEFAULT_TIMEOUT_S,
        "stdin": None,
    }]


async def test_explicit_cwd_timeout_stdin_threaded_through():
    env = FakeEnv(result=(2, "", "boom"))
    tool = CliExecTool(env_resolver=_resolver_returning(env))

    s = await tool.execute(
        "do thing",
        {
            "env": "local",
            "cmd": ["bash", "-c", "exit 2"],
            "cwd": "/work",
            "timeout_s": 5.0,
            "stdin": "input data",
        },
    )

    assert "rc=2" in s.content
    assert "boom" in s.content  # stderr surfaced
    assert env.calls[0]["cwd"] == Path("/work")
    assert env.calls[0]["timeout"] == 5.0
    assert env.calls[0]["stdin"] == "input data"


# --------------------------------------------------------------------
# Bad params → error stimulus, fake env never called
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "params, expect",
    [
        ({"cmd": ["echo"]}, "missing or invalid `env`"),
        ({"env": "", "cmd": ["echo"]}, "missing or invalid `env`"),
        ({"env": 123, "cmd": ["echo"]}, "missing or invalid `env`"),
        ({"env": "local"}, "non-empty list of strings"),
        ({"env": "local", "cmd": []}, "non-empty list of strings"),
        ({"env": "local", "cmd": "echo"}, "non-empty list of strings"),
        ({"env": "local", "cmd": [1, 2]}, "non-empty list of strings"),
        ({"env": "local", "cmd": ["x"], "cwd": ""}, "`cwd` must be"),
        ({"env": "local", "cmd": ["x"], "cwd": 42}, "`cwd` must be"),
        ({"env": "local", "cmd": ["x"], "timeout_s": 0}, "`timeout_s`"),
        ({"env": "local", "cmd": ["x"], "timeout_s": -1}, "`timeout_s`"),
        ({"env": "local", "cmd": ["x"], "timeout_s": "5"}, "`timeout_s`"),
        # bool is a subclass of int — explicitly rejected
        ({"env": "local", "cmd": ["x"], "timeout_s": True}, "`timeout_s`"),
        ({"env": "local", "cmd": ["x"], "stdin": 9}, "`stdin` must be a string"),
    ],
)
async def test_bad_params_return_error_without_calling_env(params, expect):
    env = FakeEnv()
    tool = CliExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute("anything", params)
    assert s.type == "tool_feedback"
    assert s.content.startswith("cli_exec error:")
    assert expect in s.content
    assert env.calls == []


# --------------------------------------------------------------------
# Denial / unavailability / timeout / generic error → error stimulus
# --------------------------------------------------------------------


async def test_denied_env_returns_error_stimulus():
    tool = CliExecTool(env_resolver=_resolver_denying("nope"))
    s = await tool.execute(
        "x", {"env": "sandbox", "cmd": ["echo", "x"]},
    )
    assert s.content.startswith("cli_exec error:")
    assert "denied" in s.content
    assert "sandbox" in s.content


async def test_resolver_generic_error_returns_error_stimulus():
    def _resolver(_name):
        raise RuntimeError("bus on fire")

    tool = CliExecTool(env_resolver=_resolver)
    s = await tool.execute(
        "x", {"env": "local", "cmd": ["echo", "x"]},
    )
    assert "env resolver error" in s.content
    assert "RuntimeError" in s.content


async def test_run_timeout_returns_error_stimulus():
    env = FakeEnv(raises=asyncio.TimeoutError())
    tool = CliExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x",
        {"env": "local", "cmd": ["sleep", "100"], "timeout_s": 0.1},
    )
    assert "timed out" in s.content
    assert "0.1" in s.content
    assert "local" in s.content


async def test_run_unavailable_returns_error_stimulus():
    env = FakeEnv(raises=EnvironmentUnavailableError("guest down"))
    tool = CliExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x", {"env": "sandbox", "cmd": ["echo"]},
    )
    assert "unavailable" in s.content
    assert "guest down" in s.content


async def test_run_generic_error_returns_error_stimulus():
    env = FakeEnv(raises=OSError("EIO"))
    tool = CliExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x", {"env": "local", "cmd": ["echo"]},
    )
    assert "env.run error" in s.content
    assert "OSError" in s.content


# --------------------------------------------------------------------
# Output truncation
# --------------------------------------------------------------------


async def test_long_stdout_is_truncated_with_marker():
    big = "a" * (OUTPUT_TRUNCATE_CHARS * 3)
    env = FakeEnv(result=(0, big, ""))
    tool = CliExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x", {"env": "local", "cmd": ["echo", "huge"]},
    )
    assert "[truncated, total" in s.content
    # truncated content + marker is shorter than the original
    assert len(s.content) < len(big) + 500


async def test_short_output_not_truncated():
    env = FakeEnv(result=(0, "tiny", ""))
    tool = CliExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x", {"env": "local", "cmd": ["echo", "tiny"]},
    )
    assert "[truncated" not in s.content


# --------------------------------------------------------------------
# Tool ABC surface — describe()-equivalent introspection used by the
# capabilities layer.
# --------------------------------------------------------------------


def test_static_tool_metadata():
    tool = CliExecTool(env_resolver=_resolver_returning(FakeEnv()))
    assert tool.name == "cli_exec"
    schema = tool.parameters_schema
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"env", "cmd"}
    props = schema["properties"]
    for k in ("env", "cmd", "cwd", "timeout_s", "stdin"):
        assert k in props
    # Description mentions both env names so Self can pick.
    assert "local" in tool.description
    assert "sandbox" in tool.description
