"""``gui_exec`` Tool — perform GUI ops in a Self-chosen Environment
via ``pyautogui`` snippets.

Mirrors ``cli_exec``'s envelope (capture ``ctx.environment`` in
factory, validate Self params, soft-fail on every error path) and
swaps the body: each ``action`` lives in
``krakey.plugins.gui_exec.snippets`` and produces a one-liner
Python source string that the tool dispatches as
``[python, "-c", snippet]`` to ``env.run``. The plugin therefore
holds NO platform-specific knowledge — pyautogui handles
Windows / Linux / macOS uniformly inside the env.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Callable

from krakey.interfaces.environment import (
    EnvironmentDenied,
    EnvironmentUnavailableError,
)
from krakey.interfaces.tool import Tool
from krakey.models.stimulus import Stimulus
from krakey.plugins.gui_exec import snippets as snip

if TYPE_CHECKING:
    from krakey.interfaces.environment import Environment
    from krakey.interfaces.plugin_context import PluginContext


DEFAULT_TIMEOUT_S = 15.0
"""Cap on how long any single GUI action may take. Drag is the
longest expected operation (sub-second by default); 15s leaves
slack for slow guests."""

DEFAULT_PYTHON_CMD = "python"
"""Default interpreter name on the env's PATH. Many Linux distros
expose only ``python3`` (no bare ``python``); on those, set
``python_cmd: python3`` (or an absolute path) in
``workspace/plugins/gui_exec/config.yaml`` so the tool dispatches to
an interpreter that exists. The local Windows host has ``python`` by
convention; sandbox guests vary."""

SCREENSHOT_DIR = PurePosixPath("workspace/data/screenshots")
"""Relative path inside the env's filesystem where screenshots
land. ``PurePosixPath`` so the snippet always emits ``/`` joins
regardless of the host OS — the env's Python interprets the path
in its own filesystem semantics."""

ACTIONS = (
    "click", "right_click", "double_click",
    "drag", "type", "key", "screenshot",
)
"""Action enum used in ``parameters_schema`` and dispatch."""


def _now_ts() -> str:
    """Filename-safe timestamp for screenshot output paths.

    Microsecond precision so tight-loop screenshot calls don't
    collide. Module-level (not inlined) so tests can monkeypatch
    it for deterministic path assertions."""
    return datetime.now().strftime("%Y%m%dT%H%M%S_%f")


def build_tool(ctx: "PluginContext") -> "GuiExecTool":
    """Factory for the single ``tool`` component declared in
    ``meta.yaml``. Captures the per-plugin env resolver and reads
    the optional ``python_cmd`` override from the plugin's
    ``config.yaml`` (default ``"python"``). A non-string or empty
    value falls back to the default rather than raising — keeps
    the additive-plugin invariant."""
    raw = ctx.config.get("python_cmd")
    python_cmd = raw if isinstance(raw, str) and raw.strip() else DEFAULT_PYTHON_CMD
    return GuiExecTool(
        env_resolver=ctx.environment, python_cmd=python_cmd,
    )


class GuiExecTool(Tool):
    """Self-facing tool that performs one GUI action per call."""

    def __init__(
        self,
        env_resolver: Callable[[str], "Environment"],
        python_cmd: str = DEFAULT_PYTHON_CMD,
    ):
        self._env_resolver = env_resolver
        self._python_cmd = python_cmd

    @property
    def name(self) -> str:
        return "gui_exec"

    @property
    def description(self) -> str:
        return (
            "Perform a GUI operation in a target Environment via "
            "pyautogui. `env` selects the Environment by name "
            "(e.g. \"local\" or \"sandbox\"); the plugin must be "
            "allow-listed for that env in config. `action` is one "
            "of: click, right_click, double_click, drag, type, key, "
            "screenshot. Coordinates are pixels relative to the "
            "env's primary display top-left. The env's Python "
            "interpreter (default \"python\"; configurable via the "
            "plugin's `python_cmd` config field) must have pyautogui "
            "installed."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["env", "action"],
            "properties": {
                "env": {
                    "type": "string",
                    "description": "Environment name; e.g. \"local\".",
                },
                "action": {
                    "type": "string",
                    "enum": list(ACTIONS),
                    "description": (
                        "Which GUI action to perform; param "
                        "requirements depend on the choice."
                    ),
                },
                "x": {
                    "type": "number",
                    "description": (
                        "X coordinate — required for click, "
                        "right_click, double_click, drag (start)."
                    ),
                },
                "y": {
                    "type": "number",
                    "description": (
                        "Y coordinate — required for click, "
                        "right_click, double_click, drag (start)."
                    ),
                },
                "x2": {
                    "type": "number",
                    "description": "Drag end-point X.",
                },
                "y2": {
                    "type": "number",
                    "description": "Drag end-point Y.",
                },
                "duration": {
                    "type": "number",
                    "description": (
                        "Drag duration in seconds (default 0.5)."
                    ),
                },
                "text": {
                    "type": "string",
                    "description": (
                        "Text to type — required for action=type. "
                        "ASCII-printable + named keys only; use "
                        "action=key for chords."
                    ),
                },
                "combo": {
                    "type": "string",
                    "description": (
                        "Key chord — required for action=key. "
                        "Single key (\"enter\") or "
                        "plus-joined (\"ctrl+shift+t\")."
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

        action = params.get("action")
        if not isinstance(action, str) or action not in ACTIONS:
            return self._err(
                f"`action` must be one of {list(ACTIONS)}",
            )

        try:
            snippet, summary = self._build_snippet(action, params)
        except ValueError as e:
            return self._err(str(e))

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

        cmd = [self._python_cmd, "-c", snippet]
        try:
            rc, out, err = await env.run(
                cmd,
                cwd=Path("."),
                timeout=DEFAULT_TIMEOUT_S,
                stdin=None,
            )
        except asyncio.TimeoutError:
            return self._err(
                f"action {action!r} timed out after "
                f"{DEFAULT_TIMEOUT_S}s",
            )
        except EnvironmentUnavailableError as e:
            return self._err(
                f"environment {env_name!r} unavailable: {e}",
            )
        except Exception as e:  # noqa: BLE001
            return self._err(
                f"env.run error: {type(e).__name__}: {e}",
            )

        if rc != 0:
            return self._err(
                f"action {action!r} returned rc={rc}; stderr: "
                f"{_truncate(err, 600)}",
            )

        body = (
            f"gui_exec env={env_name} action={action} rc=0"
        )
        if summary:
            body = f"{body} {summary}"
        return Stimulus(
            type="tool_feedback",
            source=f"tool:{self.name}",
            content=body,
            timestamp=datetime.now(),
            adrenalin=False,
        )

    def _build_snippet(
        self, action: str, params: dict[str, Any],
    ) -> tuple[str, str]:
        """Produce ``(python_snippet, receipt_summary)``.

        Raises ``ValueError`` on missing/invalid params; ``execute``
        catches and converts to an error Stimulus. Each branch keeps
        param parsing local so adding an action doesn't ripple
        through earlier branches.
        """
        if action == "click":
            x, y = self._xy(params)
            return snip.click(x, y, "left"), f"at ({x},{y})"
        if action == "right_click":
            x, y = self._xy(params)
            return snip.click(x, y, "right"), f"at ({x},{y})"
        if action == "double_click":
            x, y = self._xy(params)
            return snip.double_click(x, y), f"at ({x},{y})"
        if action == "drag":
            x1, y1 = self._xy(params)
            x2 = params.get("x2")
            y2 = params.get("y2")
            if not _is_num(x2) or not _is_num(y2):
                raise ValueError(
                    "drag requires `x2` and `y2` (numbers)",
                )
            duration_raw = params.get("duration", 0.5)
            if (
                not _is_num(duration_raw)
                or duration_raw < 0
            ):
                raise ValueError(
                    "`duration` must be a non-negative number",
                )
            return (
                snip.drag(x1, y1, int(x2), int(y2),
                          float(duration_raw)),
                f"({x1},{y1})->({int(x2)},{int(y2)})",
            )
        if action == "type":
            text = params.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    "`type` requires `text` (string)",
                )
            return snip.type_text(text), f"len={len(text)}"
        if action == "key":
            combo = params.get("combo")
            if not isinstance(combo, str) or not combo.strip():
                raise ValueError(
                    "`key` requires non-empty `combo` (string)",
                )
            # Reject combos that collapse to no keys after splitting
            # on '+' and trimming each part (e.g. "+", "++", " + + ").
            # Without this check ``snippets.key`` would emit
            # ``pyautogui.hotkey()`` with zero args — a silent no-op
            # that returns rc=0 inside the env, and the tool would
            # report SUCCESS even though no chord was pressed.
            if not [p.strip() for p in combo.split("+") if p.strip()]:
                raise ValueError(
                    "`key` `combo` must contain at least one key "
                    "(got only separator characters)",
                )
            return snip.key(combo), f"combo={combo!r}"
        if action == "screenshot":
            out = SCREENSHOT_DIR / f"{_now_ts()}.png"
            return snip.screenshot(out), f"saved to {out}"
        # Unreachable: action validated against ACTIONS above.
        raise ValueError(f"unknown action {action!r}")

    def _xy(self, params: dict[str, Any]) -> tuple[int, int]:
        x = params.get("x")
        y = params.get("y")
        if not _is_num(x) or not _is_num(y):
            raise ValueError(
                "this action requires `x` and `y` (numbers)",
            )
        return int(x), int(y)

    def _err(self, msg: str) -> Stimulus:
        return Stimulus(
            type="tool_feedback",
            source=f"tool:{self.name}",
            content=f"gui_exec error: {msg}",
            timestamp=datetime.now(),
            adrenalin=False,
        )


def _is_num(v: Any) -> bool:
    """Strict numeric check — bool is excluded because ``True``
    is technically an int but never what Self meant for a coord."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"...[+{len(s) - limit}]"
