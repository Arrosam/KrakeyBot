"""Unit tests for ``browser_exec`` plugin.

Run from repo root:

    pytest krakey/plugins/browser_exec

Pins the public surface and the ``execute`` dispatch path against
a FakeEnv (no real browser needed in CI). The plugin only ever
talks to the Environment via the resolver closure captured at
construction, so the FakeEnv stand-in is sufficient for full
behavioral coverage.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from krakey.interfaces.environment import (
    EnvironmentDenied,
    EnvironmentUnavailableError,
)
from krakey.plugins.browser_exec import tool as tool_mod
from krakey.plugins.browser_exec.tool import (
    ACTIONS,
    BROWSERS,
    DEFAULT_BROWSER,
    DEFAULT_HEADLESS,
    DEFAULT_PYTHON_CMD,
    DEFAULT_TIMEOUT_S,
    OUTPUT_FORMATS,
    OUTPUT_TRUNCATE_CHARS,
    SCREENSHOT_DIR,
    BrowserExecTool,
    build_tool,
)


# =====================================================================
# Plugin metadata sanity
# =====================================================================


def test_meta_yaml_parses_and_declares_one_tool():
    """Catches accidental meta.yaml drift (e.g. missing factory_attr,
    wrong kind enum). Uses the same parser the runtime uses so the
    failure mode here matches what an operator would see."""
    import yaml

    meta_path = Path(__file__).resolve().parent.parent / "meta.yaml"
    data = yaml.safe_load(meta_path.read_text(encoding="utf-8"))

    assert data["name"] == "browser_exec"
    assert isinstance(data.get("description"), str) and data["description"]

    components = data["components"]
    assert isinstance(components, list) and len(components) == 1
    comp = components[0]
    assert comp["kind"] == "tool"
    assert comp["factory_module"] == "krakey.plugins.browser_exec.tool"
    assert comp["factory_attr"] == "build_tool"

    # config_schema advertises the four documented fields.
    schema = data["config_schema"]
    field_names = {entry["field"] for entry in schema}
    assert field_names == {
        "python_cmd", "headless",
        "default_browser", "default_timeout_s",
    }


# =====================================================================
# Module-level constants — pinned so other modules / tests can
# import them safely
# =====================================================================


def test_default_constants_have_expected_shapes():
    assert DEFAULT_PYTHON_CMD == "python"
    assert DEFAULT_TIMEOUT_S == 30.0
    assert DEFAULT_BROWSER == "chromium"
    assert DEFAULT_HEADLESS is True
    assert set(BROWSERS) == {"chromium", "firefox", "webkit"}
    assert "a11y" in OUTPUT_FORMATS  # default format
    assert set(OUTPUT_FORMATS) == {"a11y", "text", "html"}
    assert set(ACTIONS) == {
        "navigate", "click", "type", "press",
        "scroll", "wait_for", "screenshot",
    }


# =====================================================================
# Factory + tool shape (skeleton-level, real dispatch in step 3)
# =====================================================================


class _FakeCtx:
    """Minimal duck-typed PluginContext stand-in (matches the pattern
    used in gui_exec's factory tests). The factory only touches
    ``.config`` and ``.environment`` so a small namespace suffices."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def environment(self, _name: str):  # pragma: no cover — stub
        raise AssertionError("env_resolver not exercised in skeleton tests")


def test_build_tool_returns_browser_exec_tool_instance():
    tool = build_tool(_FakeCtx())  # type: ignore[arg-type]
    assert isinstance(tool, BrowserExecTool)


def test_tool_static_metadata():
    tool = build_tool(_FakeCtx())  # type: ignore[arg-type]
    assert tool.name == "browser_exec"

    schema = tool.parameters_schema
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"env", "start_url", "actions"}

    props = schema["properties"]
    for k in (
        "env", "start_url", "actions",
        "timeout_s", "output", "return_screenshot",
        "headless", "browser",
    ):
        assert k in props, f"missing schema property: {k}"

    # Description names the env names + the python_cmd config field +
    # the playwright dep so Self can describe the failure modes.
    desc = tool.description
    assert "local" in desc
    assert "sandbox" in desc
    assert "playwright" in desc.lower()
    assert "python_cmd" in desc
    assert "a11y" in desc.lower()


def test_tool_default_constants_pin_against_browser_constants():
    # Convenience cross-check that the BROWSERS / OUTPUT_FORMATS
    # tuples in tool.py match what the schema and validators rely
    # on. If a future refactor splits these between modules they
    # MUST stay in sync.
    assert DEFAULT_BROWSER in BROWSERS
    assert "a11y" in OUTPUT_FORMATS


# =====================================================================
# FakeEnv — same shape as gui_exec's; records calls + canned result
# =====================================================================


class FakeEnv:
    name = "local"

    def __init__(
        self,
        result: tuple[int, str, str] = (0, "", ""),
        raises: BaseException | None = None,
    ):
        self._result = result
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

    async def preflight(self):
        return None


def _resolver_returning(env: FakeEnv):
    def resolver(_name: str) -> FakeEnv:
        return env
    return resolver


def _ok_payload(
    final_url: str = "https://example.com/",
    output: Any = {"role": "WebArea", "name": "Example Domain"},
    output_format: str = "a11y",
    actions_completed: int = 0,
    actions_total: int = 0,
    screenshot_path: str | None = None,
) -> str:
    return json.dumps({
        "final_url":         final_url,
        "output_format":     output_format,
        "output":            output,
        "screenshot_path":   screenshot_path,
        "actions_completed": actions_completed,
        "actions_total":     actions_total,
    })


# =====================================================================
# Happy path
# =====================================================================


async def test_happy_path_returns_success_stimulus_with_a11y_tree():
    env = FakeEnv(result=(0, _ok_payload(), ""))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))

    s = await tool.execute(
        "fetch the docs",
        {
            "env": "local",
            "start_url": "https://example.com",
            "actions": [],
        },
    )

    assert s.type == "tool_feedback"
    assert s.source == "tool:browser_exec"
    assert "browser_exec env=local" in s.content
    assert "final_url='https://example.com/'" in s.content
    assert "format=a11y" in s.content
    assert "actions=0/0" in s.content
    # The a11y tree (a dict) is JSON-serialized into the body so
    # the prompt sees structured data.
    assert "Example Domain" in s.content


async def test_happy_path_dispatches_python_minus_c_snippet():
    env = FakeEnv(result=(0, _ok_payload(), ""))
    tool = BrowserExecTool(
        env_resolver=_resolver_returning(env),
        python_cmd="python3",
    )
    await tool.execute(
        "x",
        {
            "env": "local",
            "start_url": "https://example.com",
            "actions": [],
        },
    )
    assert len(env.calls) == 1
    cmd = env.calls[0]["cmd"]
    assert cmd[0] == "python3"
    assert cmd[1] == "-c"
    # The snippet must be valid Python source; compile it.
    compile(cmd[2], "<dispatched_snippet>", "exec")


async def test_happy_path_threads_actions_into_snippet_via_json():
    env = FakeEnv(result=(0, _ok_payload(), ""))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    await tool.execute(
        "x",
        {
            "env": "local",
            "start_url": "https://example.com",
            "actions": [
                {"action": "type", "selector": "input[name='q']",
                 "text": "krakey"},
                {"action": "press", "key": "Enter"},
            ],
        },
    )
    snippet = env.calls[0]["cmd"][2]
    # Selector + text values appear inside the JSON literal, not as
    # bare Python tokens (they're string values inside SPEC =
    # json.loads(...)).
    assert "input[name='q']" in snippet
    assert "krakey" in snippet


async def test_browser_and_headless_overrides_threaded_into_spec():
    env = FakeEnv(result=(0, _ok_payload(), ""))
    tool = BrowserExecTool(
        env_resolver=_resolver_returning(env),
        headless=True, default_browser="chromium",
    )
    await tool.execute(
        "x",
        {
            "env": "local",
            "start_url": "https://example.com",
            "actions": [],
            "browser": "firefox",
            "headless": False,
        },
    )
    snippet = env.calls[0]["cmd"][2]
    # Pull the JSON literal out and decode it to confirm the
    # overrides made it through.
    import re
    m = re.search(r"SPEC = json\.loads\((.*)\)", snippet)
    spec = json.loads(json.loads(m.group(1)))
    assert spec["browser"] == "firefox"
    assert spec["headless"] is False


# =====================================================================
# return_screenshot=True appends a screenshot action
# =====================================================================


async def test_return_screenshot_appends_screenshot_action(monkeypatch):
    monkeypatch.setattr(tool_mod, "_now_ts", lambda: "FIXED")
    env = FakeEnv(result=(0, _ok_payload(
        screenshot_path=str(SCREENSHOT_DIR / "FIXED.png"),
    ), ""))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    await tool.execute(
        "x",
        {
            "env": "local",
            "start_url": "https://example.com",
            "actions": [],
            "return_screenshot": True,
        },
    )
    snippet = env.calls[0]["cmd"][2]
    import re
    m = re.search(r"SPEC = json\.loads\((.*)\)", snippet)
    spec = json.loads(json.loads(m.group(1)))
    # Screenshot action appended at the end of the chain.
    assert spec["actions"][-1] == {"action": "screenshot"}
    # Path is plumbed into the spec so the snippet writes to the
    # right place.
    assert spec["screenshot_path"] == "workspace/data/screenshots/FIXED.png"


async def test_return_screenshot_does_not_double_append(monkeypatch):
    """If Self already included a screenshot action in the chain,
    the tool should NOT append a second one — Self decides where
    in the chain to capture, the flag is just a convenience."""
    monkeypatch.setattr(tool_mod, "_now_ts", lambda: "FIXED")
    env = FakeEnv(result=(0, _ok_payload(), ""))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    await tool.execute(
        "x",
        {
            "env": "local",
            "start_url": "https://example.com",
            "actions": [{"action": "screenshot"}],
            "return_screenshot": True,
        },
    )
    snippet = env.calls[0]["cmd"][2]
    import re
    m = re.search(r"SPEC = json\.loads\((.*)\)", snippet)
    spec = json.loads(json.loads(m.group(1)))
    assert len(spec["actions"]) == 1


# =====================================================================
# Bad params — error Stimulus, env never called
# =====================================================================


@pytest.mark.parametrize("params, expect", [
    # missing env
    ({"start_url": "https://x", "actions": []},
     "missing or invalid `env`"),
    ({"env": "", "start_url": "https://x", "actions": []},
     "missing or invalid `env`"),
    ({"env": 5, "start_url": "https://x", "actions": []},
     "missing or invalid `env`"),
    # missing / bad start_url
    ({"env": "local", "actions": []},
     "non-empty string"),
    ({"env": "local", "start_url": "", "actions": []},
     "non-empty string"),
    ({"env": "local", "start_url": "file:///etc/passwd", "actions": []},
     "http://"),
    ({"env": "local", "start_url": "javascript:alert(1)", "actions": []},
     "http://"),
    ({"env": "local", "start_url": "data:text/html,<h1>x</h1>",
      "actions": []},
     "http://"),
    # missing / bad actions
    ({"env": "local", "start_url": "https://x"},
     "must be an array"),
    ({"env": "local", "start_url": "https://x", "actions": "click"},
     "must be an array"),
    # bad action shape
    ({"env": "local", "start_url": "https://x",
      "actions": [{"action": "evaluate"}]},
     "must be one of"),
    ({"env": "local", "start_url": "https://x",
      "actions": [{"action": "type", "selector": "#x"}]},
     "string `text`"),
    ({"env": "local", "start_url": "https://x",
      "actions": [{"action": "scroll", "direction": "diagonal",
                   "amount": 100}]},
     "direction"),
    # bad timeout_s
    ({"env": "local", "start_url": "https://x", "actions": [],
      "timeout_s": 0},
     "positive number"),
    ({"env": "local", "start_url": "https://x", "actions": [],
      "timeout_s": True},
     "positive number"),
    ({"env": "local", "start_url": "https://x", "actions": [],
      "timeout_s": "30"},
     "positive number"),
    # bad output
    ({"env": "local", "start_url": "https://x", "actions": [],
      "output": "markdown"},
     "must be one of"),
    # bad return_screenshot / headless / browser types
    ({"env": "local", "start_url": "https://x", "actions": [],
      "return_screenshot": "yes"},
     "must be a boolean"),
    ({"env": "local", "start_url": "https://x", "actions": [],
      "headless": "yes"},
     "must be a boolean"),
    ({"env": "local", "start_url": "https://x", "actions": [],
      "browser": "edge"},
     "must be one of"),
])
async def test_bad_params_return_error_without_calling_env(
    params, expect,
):
    env = FakeEnv()
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute("anything", params)
    assert s.content.startswith("browser_exec error:")
    assert expect in s.content
    assert env.calls == []


# =====================================================================
# Env errors → error Stimulus
# =====================================================================


async def test_env_denied_returns_error_stimulus():
    def deny(_name):
        raise EnvironmentDenied("plugin not allow-listed")

    tool = BrowserExecTool(env_resolver=deny)
    s = await tool.execute(
        "x",
        {"env": "local", "start_url": "https://x", "actions": []},
    )
    assert s.content.startswith("browser_exec error:")
    assert "denied" in s.content


async def test_env_resolver_unexpected_error_returns_error_stimulus():
    def boom(_name):
        raise RuntimeError("router exploded")

    tool = BrowserExecTool(env_resolver=boom)
    s = await tool.execute(
        "x",
        {"env": "local", "start_url": "https://x", "actions": []},
    )
    assert s.content.startswith("browser_exec error:")
    assert "RuntimeError" in s.content


async def test_env_run_timeout_returns_error_stimulus():
    env = FakeEnv(raises=asyncio.TimeoutError())
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x",
        {"env": "local", "start_url": "https://x", "actions": [],
         "timeout_s": 5},
    )
    assert "session timed out" in s.content


async def test_env_run_unavailable_returns_error_stimulus():
    env = FakeEnv(
        raises=EnvironmentUnavailableError("agent unreachable"),
    )
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x",
        {"env": "local", "start_url": "https://x", "actions": []},
    )
    assert "unavailable" in s.content


async def test_env_run_generic_exception_returns_error_stimulus():
    env = FakeEnv(raises=ValueError("something else"))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x",
        {"env": "local", "start_url": "https://x", "actions": []},
    )
    assert "ValueError" in s.content


# =====================================================================
# Subprocess result errors
# =====================================================================


async def test_nonzero_rc_returns_error_with_truncated_stderr():
    env = FakeEnv(result=(
        1, "", "ModuleNotFoundError: No module named 'playwright'",
    ))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x",
        {"env": "local", "start_url": "https://x", "actions": []},
    )
    assert "rc=1" in s.content
    assert "ModuleNotFoundError" in s.content


async def test_malformed_stdout_returns_error_stimulus():
    env = FakeEnv(result=(0, "not json at all", ""))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x",
        {"env": "local", "start_url": "https://x", "actions": []},
    )
    assert "could not parse session JSON" in s.content


async def test_stdout_missing_keys_returns_error_stimulus():
    """Even valid JSON must carry the expected keys; otherwise
    something inside the snippet went wrong (different version,
    truncated stdout). Surface a precise error rather than
    pretending it was a success."""
    env = FakeEnv(result=(
        0, json.dumps({"unexpected": "shape"}), "",
    ))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x",
        {"env": "local", "start_url": "https://x", "actions": []},
    )
    assert "missing expected keys" in s.content


# =====================================================================
# Stderr truncation is bounded
# =====================================================================


async def test_long_stderr_is_truncated():
    huge = "X" * (OUTPUT_TRUNCATE_CHARS + 500)
    env = FakeEnv(result=(1, "", huge))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x",
        {"env": "local", "start_url": "https://x", "actions": []},
    )
    # The stderr chunk in the body must be ≤ truncation cap (plus
    # the truncation marker line); never the full ``huge`` string.
    assert len(huge) > len(s.content)
    assert "truncated" in s.content
