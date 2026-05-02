"""Unit tests for ``gui_exec`` Tool + ``snippets`` builders.

Run from repo root:

    pytest krakey/plugins/gui_exec

(Same in-plugin convention as ``cli_exec``.)
"""
from __future__ import annotations

import asyncio
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import pytest

from krakey.interfaces.environment import (
    EnvironmentDenied,
    EnvironmentUnavailableError,
)
from krakey.plugins.gui_exec import snippets, tool as tool_mod
from krakey.plugins.gui_exec.tool import (
    DEFAULT_TIMEOUT_S,
    GuiExecTool,
    PYTHON_CMD,
    SCREENSHOT_DIR,
)


# =====================================================================
# snippets.py — pure functions, exact string match
# =====================================================================


def test_click_left_default():
    assert snippets.click(100, 200) == (
        "import pyautogui;"
        "pyautogui.click(100, 200, button='left')"
    )


def test_click_right_explicit():
    assert snippets.click(50, 75, "right") == (
        "import pyautogui;"
        "pyautogui.click(50, 75, button='right')"
    )


def test_click_coerces_floats_to_int():
    # int() truncates floats — documented coercion.
    assert snippets.click(10.9, 20.1) == (
        "import pyautogui;"
        "pyautogui.click(10, 20, button='left')"
    )


def test_double_click_emits_doubleClick():
    assert snippets.double_click(7, 8) == (
        "import pyautogui;pyautogui.doubleClick(7, 8)"
    )


def test_drag_default_duration():
    assert snippets.drag(1, 2, 3, 4) == (
        "import pyautogui;"
        "pyautogui.moveTo(1, 2);"
        "pyautogui.dragTo(3, 4, duration=0.5, button='left')"
    )


def test_drag_custom_duration():
    assert snippets.drag(0, 0, 50, 50, 1.25) == (
        "import pyautogui;"
        "pyautogui.moveTo(0, 0);"
        "pyautogui.dragTo(50, 50, duration=1.25, button='left')"
    )


def test_type_text_uses_repr_for_safety():
    # Embedded quote + backslash must round-trip safely.
    s = snippets.type_text("a'b\\c")
    # repr embeds correctly:
    assert s == (
        "import pyautogui;"
        "pyautogui.typewrite("
        + repr("a'b\\c")
        + ", interval=0.0)"
    )


def test_type_text_with_interval():
    s = snippets.type_text("hi", interval=0.05)
    assert "interval=0.05" in s
    assert "pyautogui.typewrite('hi'" in s


def test_key_single_uses_press():
    assert snippets.key("enter") == (
        "import pyautogui;pyautogui.press('enter')"
    )


def test_key_combo_uses_hotkey():
    assert snippets.key("ctrl+c") == (
        "import pyautogui;pyautogui.hotkey('ctrl', 'c')"
    )


def test_key_three_part_combo():
    assert snippets.key("ctrl+shift+t") == (
        "import pyautogui;"
        "pyautogui.hotkey('ctrl', 'shift', 't')"
    )


def test_key_strips_whitespace_and_drops_empty_parts():
    # " ctrl + + c " → ["ctrl", "c"]
    assert snippets.key(" ctrl + + c ") == (
        "import pyautogui;pyautogui.hotkey('ctrl', 'c')"
    )


def test_screenshot_includes_makedirs():
    s = snippets.screenshot("workspace/data/screenshots/x.png")
    assert "os.makedirs" in s
    assert "exist_ok=True" in s
    assert "pyautogui.screenshot(" in s
    assert "workspace/data/screenshots/x.png" in s


def test_screenshot_accepts_path_object():
    p = PurePosixPath("a/b.png")
    s = snippets.screenshot(p)
    assert "a/b.png" in s


# =====================================================================
# Fakes
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


def _resolver_returning(env: FakeEnv) -> Callable[[str], FakeEnv]:
    return lambda _name: env


# =====================================================================
# Tool — happy paths per action; verify argv shape
# =====================================================================


async def test_click_dispatch_happy():
    env = FakeEnv()
    tool = GuiExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "tap button",
        {"env": "local", "action": "click", "x": 10, "y": 20},
    )
    assert s.type == "tool_feedback"
    assert s.source == "tool:gui_exec"
    assert "rc=0" in s.content
    assert "action=click" in s.content
    assert "(10,20)" in s.content
    assert env.calls[0]["cmd"] == [
        PYTHON_CMD, "-c", snippets.click(10, 20, "left"),
    ]
    assert env.calls[0]["timeout"] == DEFAULT_TIMEOUT_S
    assert env.calls[0]["stdin"] is None


async def test_right_click_uses_right_button_in_snippet():
    env = FakeEnv()
    tool = GuiExecTool(env_resolver=_resolver_returning(env))
    await tool.execute(
        "x",
        {"env": "local", "action": "right_click",
         "x": 5, "y": 6},
    )
    assert env.calls[0]["cmd"][2] == snippets.click(5, 6, "right")


async def test_double_click_dispatch():
    env = FakeEnv()
    tool = GuiExecTool(env_resolver=_resolver_returning(env))
    await tool.execute(
        "x",
        {"env": "local", "action": "double_click",
         "x": 1, "y": 2},
    )
    assert env.calls[0]["cmd"][2] == snippets.double_click(1, 2)


async def test_drag_default_duration():
    env = FakeEnv()
    tool = GuiExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x",
        {"env": "local", "action": "drag",
         "x": 0, "y": 0, "x2": 10, "y2": 20},
    )
    assert env.calls[0]["cmd"][2] == snippets.drag(0, 0, 10, 20, 0.5)
    assert "(0,0)->(10,20)" in s.content


async def test_drag_custom_duration():
    env = FakeEnv()
    tool = GuiExecTool(env_resolver=_resolver_returning(env))
    await tool.execute(
        "x",
        {"env": "local", "action": "drag",
         "x": 0, "y": 0, "x2": 10, "y2": 20, "duration": 2.0},
    )
    assert "duration=2.0" in env.calls[0]["cmd"][2]


async def test_type_dispatch():
    env = FakeEnv()
    tool = GuiExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x",
        {"env": "local", "action": "type", "text": "hello"},
    )
    assert env.calls[0]["cmd"][2] == snippets.type_text("hello")
    assert "len=5" in s.content


async def test_key_single_dispatch():
    env = FakeEnv()
    tool = GuiExecTool(env_resolver=_resolver_returning(env))
    await tool.execute(
        "x",
        {"env": "local", "action": "key", "combo": "enter"},
    )
    assert env.calls[0]["cmd"][2] == snippets.key("enter")


async def test_key_combo_dispatch():
    env = FakeEnv()
    tool = GuiExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x",
        {"env": "local", "action": "key", "combo": "ctrl+c"},
    )
    assert env.calls[0]["cmd"][2] == snippets.key("ctrl+c")
    assert "ctrl+c" in s.content


async def test_screenshot_uses_workspace_data_screenshots_path(
    monkeypatch,
):
    monkeypatch.setattr(tool_mod, "_now_ts", lambda: "FIXED")
    env = FakeEnv()
    tool = GuiExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x",
        {"env": "local", "action": "screenshot"},
    )
    expected_path = SCREENSHOT_DIR / "FIXED.png"
    assert env.calls[0]["cmd"][2] == snippets.screenshot(expected_path)
    assert str(expected_path) in s.content
    # Path uses POSIX separators so the same string lands in the
    # snippet regardless of host OS:
    assert "workspace/data/screenshots/FIXED.png" in s.content


# =====================================================================
# Tool — bad params → error stimulus, env never called
# =====================================================================


@pytest.mark.parametrize(
    "params, expect",
    [
        # env missing/invalid
        ({"action": "click", "x": 1, "y": 2},
         "missing or invalid `env`"),
        ({"env": "", "action": "click", "x": 1, "y": 2},
         "missing or invalid `env`"),
        ({"env": 5, "action": "click", "x": 1, "y": 2},
         "missing or invalid `env`"),
        # action missing/invalid
        ({"env": "local"}, "must be one of"),
        ({"env": "local", "action": "explode"}, "must be one of"),
        ({"env": "local", "action": ""}, "must be one of"),
        # click/right_click/double_click without coords
        ({"env": "local", "action": "click"},
         "requires `x` and `y`"),
        ({"env": "local", "action": "click", "x": 1},
         "requires `x` and `y`"),
        ({"env": "local", "action": "click", "x": "a", "y": 2},
         "requires `x` and `y`"),
        ({"env": "local", "action": "right_click", "y": 2},
         "requires `x` and `y`"),
        ({"env": "local", "action": "double_click"},
         "requires `x` and `y`"),
        # bool rejected as coord (subclass of int but never intended)
        ({"env": "local", "action": "click", "x": True, "y": 2},
         "requires `x` and `y`"),
        # drag missing endpoints
        ({"env": "local", "action": "drag", "x": 0, "y": 0},
         "drag requires `x2` and `y2`"),
        ({"env": "local", "action": "drag",
         "x": 0, "y": 0, "x2": 10},
         "drag requires `x2` and `y2`"),
        # negative duration
        ({"env": "local", "action": "drag",
         "x": 0, "y": 0, "x2": 10, "y2": 10, "duration": -1},
         "`duration` must be"),
        # type without text
        ({"env": "local", "action": "type"},
         "`type` requires `text`"),
        ({"env": "local", "action": "type", "text": 5},
         "`type` requires `text`"),
        # key without combo
        ({"env": "local", "action": "key"},
         "`key` requires non-empty `combo`"),
        ({"env": "local", "action": "key", "combo": "  "},
         "`key` requires non-empty `combo`"),
        ({"env": "local", "action": "key", "combo": 0},
         "`key` requires non-empty `combo`"),
        # key combos that collapse to no keys after splitting on '+':
        # the truthy `combo.strip()` guard above passes them, so the
        # tool must catch the empty-parts case explicitly to avoid
        # emitting `pyautogui.hotkey()` with zero args (silent no-op,
        # returns rc=0 → tool would report SUCCESS while nothing
        # happened).
        ({"env": "local", "action": "key", "combo": "+"},
         "must contain at least one key"),
        ({"env": "local", "action": "key", "combo": "++"},
         "must contain at least one key"),
        ({"env": "local", "action": "key", "combo": " + + "},
         "must contain at least one key"),
    ],
)
async def test_bad_params_return_error_without_calling_env(
    params, expect,
):
    env = FakeEnv()
    tool = GuiExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute("anything", params)
    assert s.content.startswith("gui_exec error:")
    assert expect in s.content
    assert env.calls == []


# =====================================================================
# Tool — denial / timeout / unavailability / non-zero rc / generic
# =====================================================================


async def test_denied_env_returns_error_stimulus():
    def _resolver(_name):
        raise EnvironmentDenied("nope")

    tool = GuiExecTool(env_resolver=_resolver)
    s = await tool.execute(
        "x",
        {"env": "sandbox", "action": "click", "x": 1, "y": 2},
    )
    assert "denied" in s.content
    assert "sandbox" in s.content


async def test_run_timeout_returns_error_stimulus():
    env = FakeEnv(raises=asyncio.TimeoutError())
    tool = GuiExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x",
        {"env": "local", "action": "click", "x": 1, "y": 2},
    )
    assert "timed out" in s.content
    assert str(DEFAULT_TIMEOUT_S) in s.content


async def test_run_unavailable_returns_error_stimulus():
    env = FakeEnv(raises=EnvironmentUnavailableError("guest dead"))
    tool = GuiExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x",
        {"env": "sandbox", "action": "click", "x": 1, "y": 2},
    )
    assert "unavailable" in s.content
    assert "guest dead" in s.content


async def test_run_generic_error_returns_error_stimulus():
    env = FakeEnv(raises=OSError("boom"))
    tool = GuiExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x",
        {"env": "local", "action": "click", "x": 1, "y": 2},
    )
    assert "env.run error" in s.content
    assert "OSError" in s.content


async def test_non_zero_rc_returns_error_with_stderr():
    env = FakeEnv(result=(1, "", "pyautogui boom"))
    tool = GuiExecTool(env_resolver=_resolver_returning(env))
    s = await tool.execute(
        "x",
        {"env": "local", "action": "click", "x": 1, "y": 2},
    )
    assert s.content.startswith("gui_exec error:")
    assert "rc=1" in s.content
    assert "pyautogui boom" in s.content


async def test_resolver_generic_error_returns_error_stimulus():
    def _resolver(_name):
        raise RuntimeError("bus down")

    tool = GuiExecTool(env_resolver=_resolver)
    s = await tool.execute(
        "x",
        {"env": "local", "action": "click", "x": 1, "y": 2},
    )
    assert "env resolver error" in s.content
    assert "RuntimeError" in s.content


# =====================================================================
# Static metadata
# =====================================================================


def test_static_tool_metadata():
    tool = GuiExecTool(
        env_resolver=_resolver_returning(FakeEnv()),
    )
    assert tool.name == "gui_exec"
    schema = tool.parameters_schema
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"env", "action"}
    enum = schema["properties"]["action"]["enum"]
    assert set(enum) == {
        "click", "right_click", "double_click",
        "drag", "type", "key", "screenshot",
    }
    # Description names pyautogui so Self knows what backend ships.
    assert "pyautogui" in tool.description
