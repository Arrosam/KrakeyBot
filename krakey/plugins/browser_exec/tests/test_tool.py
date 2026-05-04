"""Unit tests for ``browser_exec`` plugin (post-v2 redesign).

Run from repo root:

    pytest krakey/plugins/browser_exec

Pins the public surface (meta + schema) and the ``execute``
dispatch path against a FakeEnv. The plugin only ever talks to
the Environment via the resolver closure captured at construction,
and the env's stdout carries one JSON envelope from the running
browser server. The FakeEnv canned-response pattern is enough to
exercise the dispatch + envelope-parsing logic without launching
a real Playwright browser in CI.
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
    TOP_LEVEL_ACTIONS,
    BrowserExecTool,
    build_tool,
)


# =====================================================================
# Plugin metadata sanity
# =====================================================================


def test_meta_yaml_parses_and_declares_one_tool():
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
    schema = data["config_schema"]
    field_names = {entry["field"] for entry in schema}
    assert field_names == {
        "python_cmd", "headless",
        "default_browser", "default_timeout_s",
    }


# =====================================================================
# Module-level constants
# =====================================================================


def test_default_constants_have_expected_shapes():
    assert DEFAULT_PYTHON_CMD == "python"
    assert DEFAULT_TIMEOUT_S == 30.0
    assert DEFAULT_BROWSER == "chromium"
    assert DEFAULT_HEADLESS is True
    assert set(BROWSERS) == {"chromium", "firefox", "webkit"}
    assert set(OUTPUT_FORMATS) == {"a11y", "text", "html"}
    assert set(ACTIONS) == {
        "navigate", "click", "type", "press",
        "scroll", "wait_for", "screenshot",
    }
    assert set(TOP_LEVEL_ACTIONS) == {
        "list_tabs", "new_tab", "close_tab", "operate",
    }


# =====================================================================
# Schema + description pins
# =====================================================================


class _FakeCtx:
    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def environment(self, _name: str):  # pragma: no cover — stub
        raise AssertionError("env_resolver not exercised")


def test_build_tool_returns_browser_exec_tool_instance():
    tool = build_tool(_FakeCtx())  # type: ignore[arg-type]
    assert isinstance(tool, BrowserExecTool)


def test_tool_static_metadata():
    tool = build_tool(_FakeCtx())  # type: ignore[arg-type]
    assert tool.name == "browser_exec"

    schema = tool.parameters_schema
    assert schema["type"] == "object"
    # New top-level shape: required is just env + action.
    assert set(schema["required"]) == {"env", "action"}

    props = schema["properties"]
    for k in (
        "env", "action", "start_url", "label", "tab_id",
        "actions", "timeout_s", "output", "return_screenshot",
        "headless", "browser",
    ):
        assert k in props, f"missing schema property: {k}"

    # Action enum surface.
    assert set(props["action"]["enum"]) == set(TOP_LEVEL_ACTIONS)

    desc = tool.description
    assert "local" in desc
    assert "sandbox" in desc
    assert "playwright" in desc.lower()
    assert "python_cmd" in desc
    assert "tab" in desc.lower()
    assert "list_tabs" in desc
    assert "new_tab" in desc


# =====================================================================
# FakeEnv — canned RPC envelope
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


def _envelope(
    *,
    ok: bool = True,
    result: dict | None = None,
    error: str | None = None,
    tabs: list | None = None,
) -> str:
    """Build an RPC envelope JSON string the FakeEnv echoes from
    stdout. Matches what server.py emits."""
    body: dict = {
        "ok": ok,
        "tabs": tabs if tabs is not None else [],
    }
    if ok:
        body["result"] = result or {}
    else:
        body["error"] = error or "unspecified"
    return json.dumps(body)


# =====================================================================
# Bad params (top-level)
# =====================================================================


@pytest.mark.parametrize("params, expect", [
    # missing env
    ({"action": "list_tabs"},
     "missing or invalid `env`"),
    ({"env": "", "action": "list_tabs"},
     "missing or invalid `env`"),
    ({"env": 5, "action": "list_tabs"},
     "missing or invalid `env`"),
    # missing / bad action
    ({"env": "local"},
     "must be one of"),
    ({"env": "local", "action": "evaluate"},
     "must be one of"),
    ({"env": "local", "action": ""},
     "must be one of"),
    # bad timeout_s
    ({"env": "local", "action": "list_tabs", "timeout_s": 0},
     "positive number"),
    ({"env": "local", "action": "list_tabs", "timeout_s": True},
     "positive number"),
    ({"env": "local", "action": "list_tabs", "timeout_s": "30"},
     "positive number"),
    # bad headless / browser
    ({"env": "local", "action": "list_tabs", "headless": "yes"},
     "must be a boolean"),
    ({"env": "local", "action": "list_tabs", "browser": "edge"},
     "must be one of"),
    # ---- new_tab ----
    # missing start_url
    ({"env": "local", "action": "new_tab"},
     "non-empty string"),
    # bad start_url scheme
    ({"env": "local", "action": "new_tab",
      "start_url": "file:///etc/passwd"},
     "http://"),
    ({"env": "local", "action": "new_tab",
      "start_url": "javascript:alert(1)"},
     "http://"),
    # bad label type
    ({"env": "local", "action": "new_tab",
      "start_url": "https://x", "label": 7},
     "label"),
    # ---- close_tab ----
    ({"env": "local", "action": "close_tab"},
     "tab_id"),
    ({"env": "local", "action": "close_tab", "tab_id": ""},
     "tab_id"),
    ({"env": "local", "action": "close_tab", "tab_id": 7},
     "tab_id"),
    # ---- operate ----
    ({"env": "local", "action": "operate"},
     "tab_id"),
    ({"env": "local", "action": "operate", "tab_id": "t"},
     "must be an array"),
    # bad in-tab action
    ({"env": "local", "action": "operate", "tab_id": "t",
      "actions": [{"action": "evaluate"}]},
     "must be one of"),
    ({"env": "local", "action": "operate", "tab_id": "t",
      "actions": [{"action": "type", "selector": "#x"}]},
     "string `text`"),
    ({"env": "local", "action": "operate", "tab_id": "t",
      "actions": [{"action": "scroll", "direction": "diagonal",
                   "amount": 100}]},
     "direction"),
    # bad output format
    ({"env": "local", "action": "operate", "tab_id": "t",
      "actions": [], "output": "markdown"},
     "must be one of"),
    # bad return_screenshot
    ({"env": "local", "action": "operate", "tab_id": "t",
      "actions": [], "return_screenshot": "yes"},
     "must be a boolean"),
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
# Happy paths — one per top-level action
# =====================================================================


async def test_list_tabs_happy_path():
    env = FakeEnv(result=(0, _envelope(
        result={},
        tabs=[
            {"id": "tab_a1", "url": "https://x.com",
             "title": "X", "label": "search"},
        ],
    ), ""))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "what's open?",
        {"env": "local", "action": "list_tabs"},
    )
    assert "browser_exec env=local action=list_tabs ok" in s.content
    # Tab listing visible.
    assert "tab_a1" in s.content
    assert "https://x.com" in s.content
    assert "[search]" in s.content


async def test_new_tab_happy_path_returns_tab_id_and_lists_tabs():
    env = FakeEnv(result=(0, _envelope(
        result={
            "tab_id": "tab_xyz",
            "url": "https://example.com/",
            "title": "Example Domain",
        },
        tabs=[
            {"id": "tab_xyz", "url": "https://example.com/",
             "title": "Example Domain", "label": "docs"},
        ],
    ), ""))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "open the docs",
        {
            "env": "local", "action": "new_tab",
            "start_url": "https://example.com",
            "label": "docs",
        },
    )
    assert "action=new_tab ok" in s.content
    assert "tab_id='tab_xyz'" in s.content
    assert "Example Domain" in s.content
    # Tab list block always present.
    assert "--- tabs ---" in s.content


async def test_close_tab_happy_path():
    env = FakeEnv(result=(0, _envelope(
        result={}, tabs=[],
    ), ""))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x",
        {"env": "local", "action": "close_tab", "tab_id": "tab_xyz"},
    )
    assert "action=close_tab ok" in s.content
    assert "tab closed" in s.content
    assert "(no tabs open)" in s.content


async def test_operate_happy_path_returns_a11y_output():
    env = FakeEnv(result=(0, _envelope(
        result={
            "final_url": "https://example.com/post",
            "output_format": "a11y",
            "output": {"role": "WebArea", "name": "Example Post"},
            "screenshot_path": None,
            "actions_completed": 2,
            "actions_total": 2,
        },
        tabs=[
            {"id": "tab_xyz", "url": "https://example.com/post",
             "title": "Example Post", "label": ""},
        ],
    ), ""))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "click the link",
        {
            "env": "local", "action": "operate",
            "tab_id": "tab_xyz",
            "actions": [
                {"action": "click", "selector": "a.first"},
                {"action": "wait_for", "selector": "h1"},
            ],
        },
    )
    assert "action=operate ok" in s.content
    assert "final_url='https://example.com/post'" in s.content
    assert "format=a11y" in s.content
    assert "actions=2/2" in s.content
    # Output block carries the a11y tree.
    assert "Example Post" in s.content
    # Tab listing always present.
    assert "tab_xyz" in s.content


# =====================================================================
# Dispatch shape — one_dispatch_one_call, snippet is python -c
# =====================================================================


async def test_dispatch_uses_python_minus_c_with_snippet():
    env = FakeEnv(result=(0, _envelope(tabs=[]), ""))
    tool = BrowserExecTool(
        env_resolver=_resolver_returning(env),
        python_cmd="python3",
    )
    await tool.execute(
        "x", {"env": "local", "action": "list_tabs"},
    )
    assert len(env.calls) == 1
    cmd = env.calls[0]["cmd"]
    assert cmd[0] == "python3"
    assert cmd[1] == "-c"
    # Snippet compiles as Python.
    compile(cmd[2], "<dispatched>", "exec")
    # Carries the embedded server source.
    assert "SERVER_SOURCE = json.loads(" in cmd[2]


async def test_dispatch_threads_args_into_payload_via_json():
    env = FakeEnv(result=(0, _envelope(tabs=[]), ""))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    await tool.execute(
        "x",
        {
            "env": "local", "action": "operate",
            "tab_id": "tab_abc",
            "actions": [
                {"action": "type", "selector": "input[name='q']",
                 "text": "hello"},
            ],
        },
    )
    snippet = env.calls[0]["cmd"][2]
    # selector + text values appear inside the PAYLOAD JSON
    # literal (NOT as bare Python tokens).
    import re
    m = re.search(r"PAYLOAD = json\.loads\((.*)\)", snippet)
    payload = json.loads(json.loads(m.group(1)))
    assert payload["op"] == "operate"
    assert payload["args"]["tab_id"] == "tab_abc"
    assert payload["args"]["actions"][0]["selector"] == (
        "input[name='q']"
    )


async def test_operate_return_screenshot_appends_screenshot_action(
    monkeypatch,
):
    monkeypatch.setattr(tool_mod, "_now_ts", lambda: "FIXED")
    env = FakeEnv(result=(0, _envelope(result={
        "final_url": "https://x.com/",
        "output_format": "a11y",
        "output": {"role": "WebArea"},
        "screenshot_path": str(SCREENSHOT_DIR / "FIXED.png"),
        "actions_completed": 1,
        "actions_total": 1,
    }, tabs=[]), ""))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    await tool.execute(
        "x",
        {
            "env": "local", "action": "operate",
            "tab_id": "tab_abc", "actions": [],
            "return_screenshot": True,
        },
    )
    snippet = env.calls[0]["cmd"][2]
    import re
    m = re.search(r"PAYLOAD = json\.loads\((.*)\)", snippet)
    payload = json.loads(json.loads(m.group(1)))
    assert payload["args"]["actions"][-1] == {"action": "screenshot"}
    assert payload["args"]["screenshot_path"] == (
        "workspace/data/screenshots/FIXED.png"
    )


async def test_operate_return_screenshot_does_not_double_append(
    monkeypatch,
):
    monkeypatch.setattr(tool_mod, "_now_ts", lambda: "FIXED")
    env = FakeEnv(result=(0, _envelope(result={
        "final_url": "https://x/", "output_format": "a11y",
        "output": None, "screenshot_path": None,
        "actions_completed": 1, "actions_total": 1,
    }, tabs=[]), ""))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    await tool.execute(
        "x",
        {
            "env": "local", "action": "operate",
            "tab_id": "t",
            "actions": [{"action": "screenshot"}],
            "return_screenshot": True,
        },
    )
    snippet = env.calls[0]["cmd"][2]
    import re
    m = re.search(r"PAYLOAD = json\.loads\((.*)\)", snippet)
    payload = json.loads(json.loads(m.group(1)))
    # Only one screenshot in the chain (no duplicate).
    assert len(payload["args"]["actions"]) == 1


# =====================================================================
# Server-side / RPC-level errors — surfaced as error Stimulus
# =====================================================================


async def test_rpc_envelope_ok_false_returns_error_stimulus_with_tabs():
    """Server reported op-level error (e.g. tab_id not found).
    Tool surfaces it as an error stimulus but STILL includes the
    current tab list so Self can recover."""
    env = FakeEnv(result=(0, _envelope(
        ok=False,
        error="tab_id 'tab_does_not_exist' not found",
        tabs=[
            {"id": "tab_real", "url": "https://x/",
             "title": "Real", "label": ""},
        ],
    ), ""))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x",
        {"env": "local", "action": "close_tab",
         "tab_id": "tab_does_not_exist"},
    )
    assert "browser_exec error:" in s.content
    assert "tab_does_not_exist" in s.content
    # Tab list still present so Self can pick a real id next call.
    assert "tab_real" in s.content


async def test_rpc_envelope_with_log_tail_includes_log_in_body():
    """Server failed to start (e.g. playwright not installed).
    Snippet returns ok=false + log_tail. Tool surfaces the log
    snippet so the operator can fix it."""
    env = FakeEnv(result=(0, _envelope(
        ok=False,
        error="browser server failed to start within 30s",
        tabs=[],
    ).rstrip("}")
       + ', "log_tail": "ModuleNotFoundError: No module named '
       + "'playwright'\\n" + 'cannot import sync_api"}', ""))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x", {"env": "local", "action": "list_tabs"},
    )
    assert "failed to start" in s.content
    assert "ModuleNotFoundError" in s.content


# =====================================================================
# Env-level errors → error Stimulus (snippet never runs)
# =====================================================================


async def test_env_denied_returns_error_stimulus():
    def deny(_name):
        raise EnvironmentDenied("plugin not allow-listed")

    tool = BrowserExecTool(env_resolver=deny)
    s = await tool.execute(
        "x", {"env": "local", "action": "list_tabs"},
    )
    assert s.content.startswith("browser_exec error:")
    assert "denied" in s.content


async def test_env_resolver_unexpected_error_returns_error_stimulus():
    def boom(_name):
        raise RuntimeError("router exploded")

    tool = BrowserExecTool(env_resolver=boom)
    s = await tool.execute(
        "x", {"env": "local", "action": "list_tabs"},
    )
    assert s.content.startswith("browser_exec error:")
    assert "RuntimeError" in s.content


async def test_env_run_timeout_returns_error_stimulus():
    env = FakeEnv(raises=asyncio.TimeoutError())
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x",
        {"env": "local", "action": "list_tabs", "timeout_s": 5},
    )
    assert "dispatch timed out" in s.content


async def test_env_run_unavailable_returns_error_stimulus():
    env = FakeEnv(
        raises=EnvironmentUnavailableError("agent unreachable"),
    )
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x", {"env": "local", "action": "list_tabs"},
    )
    assert "unavailable" in s.content


async def test_env_run_generic_exception_returns_error_stimulus():
    env = FakeEnv(raises=ValueError("something else"))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x", {"env": "local", "action": "list_tabs"},
    )
    assert "ValueError" in s.content


# =====================================================================
# Subprocess result errors
# =====================================================================


async def test_nonzero_rc_returns_error_with_truncated_stderr():
    env = FakeEnv(result=(
        1, "", "Traceback: server crashed",
    ))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x", {"env": "local", "action": "list_tabs"},
    )
    assert "rc=1" in s.content
    assert "server crashed" in s.content


async def test_malformed_stdout_returns_error_stimulus():
    env = FakeEnv(result=(0, "not json at all", ""))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x", {"env": "local", "action": "list_tabs"},
    )
    assert "could not parse dispatch JSON" in s.content


async def test_stdout_missing_keys_returns_error_stimulus():
    env = FakeEnv(result=(
        0, json.dumps({"unexpected": "shape"}), "",
    ))
    tool = BrowserExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x", {"env": "local", "action": "list_tabs"},
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
        "x", {"env": "local", "action": "list_tabs"},
    )
    assert "truncated" in s.content


# =====================================================================
# build_tool factory — config field reads + fallbacks
# =====================================================================
#
# The factory only touches ``ctx.config`` (a dict) and
# ``ctx.environment`` (a callable). A richer ``_CtxWithEnv``
# captures both so we can build a tool, dispatch a real call
# through a FakeEnv, and verify the config values flow through to
# the right places: cmd[0] for python_cmd, the snippet's BROWSER /
# HEADLESS / RPC_TIMEOUT_S literals (since these are Python-source
# constants in the snippet) for the others.


class _CtxWithEnv:
    def __init__(self, config: dict, env: FakeEnv):
        self.config = config
        self._env = env

    def environment(self, _name: str) -> FakeEnv:
        return self._env


def _python_var_value(snippet: str, name: str) -> str:
    """Pull ``<name> = <literal>`` out of the dispatch snippet so
    factory tests can verify config plumbing."""
    import re
    m = re.search(rf"^{name} = (.+)$", snippet, re.MULTILINE)
    assert m, f"could not find {name} = ... in snippet"
    return m.group(1).strip()


# --- python_cmd ------------------------------------------------------


async def test_factory_reads_python_cmd_from_config():
    env = FakeEnv(result=(0, _envelope(tabs=[]), ""))
    tool = build_tool(_CtxWithEnv(  # type: ignore[arg-type]
        {"python_cmd": "/opt/py/bin/python3.11"}, env,
    ))
    await tool.execute(
        "x", {"env": "local", "action": "list_tabs"},
    )
    assert env.calls[0]["cmd"][0] == "/opt/py/bin/python3.11"
    # AND it's threaded into the snippet's PYTHON_CMD constant
    # (used for spawning the server inside the env).
    assert _python_var_value(
        env.calls[0]["cmd"][2], "PYTHON_CMD",
    ) == "'/opt/py/bin/python3.11'"


@pytest.mark.parametrize("config", [
    {},
    {"python_cmd": ""},
    {"python_cmd": "   "},
    {"python_cmd": None},
    {"python_cmd": 42},
    {"python_cmd": ["python3"]},
])
async def test_factory_falls_back_to_default_python_cmd(config):
    env = FakeEnv(result=(0, _envelope(tabs=[]), ""))
    tool = build_tool(_CtxWithEnv(config, env))  # type: ignore[arg-type]
    await tool.execute(
        "x", {"env": "local", "action": "list_tabs"},
    )
    assert env.calls[0]["cmd"][0] == DEFAULT_PYTHON_CMD


# --- headless --------------------------------------------------------


async def test_factory_reads_headless_from_config():
    env = FakeEnv(result=(0, _envelope(tabs=[]), ""))
    tool = build_tool(_CtxWithEnv(  # type: ignore[arg-type]
        {"headless": False}, env,
    ))
    await tool.execute(
        "x", {"env": "local", "action": "list_tabs"},
    )
    assert _python_var_value(
        env.calls[0]["cmd"][2], "HEADLESS",
    ) == "False"


@pytest.mark.parametrize("bad", [None, "true", 1, 0, "yes"])
async def test_factory_falls_back_to_default_headless(bad):
    env = FakeEnv(result=(0, _envelope(tabs=[]), ""))
    tool = build_tool(_CtxWithEnv(  # type: ignore[arg-type]
        {"headless": bad}, env,
    ))
    await tool.execute(
        "x", {"env": "local", "action": "list_tabs"},
    )
    assert _python_var_value(
        env.calls[0]["cmd"][2], "HEADLESS",
    ) == repr(DEFAULT_HEADLESS)


# --- default_browser --------------------------------------------------


@pytest.mark.parametrize("name", ["chromium", "firefox", "webkit"])
async def test_factory_reads_default_browser_from_config(name):
    env = FakeEnv(result=(0, _envelope(tabs=[]), ""))
    tool = build_tool(_CtxWithEnv(  # type: ignore[arg-type]
        {"default_browser": name}, env,
    ))
    await tool.execute(
        "x", {"env": "local", "action": "list_tabs"},
    )
    assert _python_var_value(
        env.calls[0]["cmd"][2], "BROWSER",
    ) == repr(name)


@pytest.mark.parametrize("bad", [
    None, "edge", "", "CHROMIUM", 7, [], {"name": "firefox"},
])
async def test_factory_falls_back_to_default_browser(bad):
    env = FakeEnv(result=(0, _envelope(tabs=[]), ""))
    tool = build_tool(_CtxWithEnv(  # type: ignore[arg-type]
        {"default_browser": bad}, env,
    ))
    await tool.execute(
        "x", {"env": "local", "action": "list_tabs"},
    )
    assert _python_var_value(
        env.calls[0]["cmd"][2], "BROWSER",
    ) == repr(DEFAULT_BROWSER)


# --- per-call override beats config default --------------------------


async def test_per_call_browser_param_beats_config_default():
    env = FakeEnv(result=(0, _envelope(tabs=[]), ""))
    tool = build_tool(_CtxWithEnv(  # type: ignore[arg-type]
        {"default_browser": "chromium"}, env,
    ))
    await tool.execute(
        "x",
        {
            "env": "local", "action": "list_tabs",
            "browser": "webkit",
        },
    )
    assert _python_var_value(
        env.calls[0]["cmd"][2], "BROWSER",
    ) == "'webkit'"


async def test_per_call_headless_beats_config_default():
    env = FakeEnv(result=(0, _envelope(tabs=[]), ""))
    tool = build_tool(_CtxWithEnv(  # type: ignore[arg-type]
        {"headless": True}, env,
    ))
    await tool.execute(
        "x",
        {
            "env": "local", "action": "list_tabs",
            "headless": False,
        },
    )
    assert _python_var_value(
        env.calls[0]["cmd"][2], "HEADLESS",
    ) == "False"
