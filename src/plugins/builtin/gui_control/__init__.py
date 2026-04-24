"""Built-in `gui_control` plugin — PyAutoGUI mouse / keyboard / screenshot.

Krakey's "computer use" hands. Backed by PyAutoGUI in production; a
mockable backend Protocol keeps tests offscreen.

⚠ **Security**: this gives Krakey the same desktop access as the user
running her. Disabled by default in config.yaml. Don't enable on
machines with sensitive material left open.

PyAutoGUI's built-in fail-safe (move mouse to a screen corner) is left
on by default — Krakey can be aborted any time by physically slamming
the cursor into the top-left.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from src.interfaces.tentacle import Tentacle
from src.models.stimulus import Stimulus


MANIFEST = {
    "name": "gui_control",
    "description": "Drive a desktop (click / type / press / screenshot). "
                   "Disabled by default; enable only when you want "
                   "Krakey operating a visible display.",
    "is_internal": True,
    "config_schema": [
        {"field": "sandbox",        "type": "bool", "default": True,
         "help": "Reserved for Phase S2; currently ignored (always runs "
                 "on host PyAutoGUI)."},
        {"field": "screenshot_dir", "type": "text",
         "default": "workspace/screenshots"},
    ],
}


class GuiBackend(Protocol):
    async def click(self, x: int, y: int, *, button: str = "left") -> None: ...
    async def move(self, x: int, y: int) -> None: ...
    async def type_text(self, text: str, *, interval: float = 0.0) -> None: ...
    async def press(self, key: str) -> None: ...
    async def screenshot(self, path: Path) -> None: ...
    async def get_screen_size(self) -> tuple[int, int]: ...


class PyAutoGUIBackend:
    """Real backend wrapping pyautogui (sync) via asyncio.to_thread."""

    async def click(self, x, y, *, button="left"):
        import pyautogui
        await asyncio.to_thread(pyautogui.click, x, y, button=button)

    async def move(self, x, y):
        import pyautogui
        await asyncio.to_thread(pyautogui.moveTo, x, y)

    async def type_text(self, text, *, interval=0.0):
        import pyautogui
        await asyncio.to_thread(pyautogui.write, text, interval)

    async def press(self, key):
        import pyautogui
        await asyncio.to_thread(pyautogui.press, key)

    async def screenshot(self, path: Path):
        import pyautogui
        await asyncio.to_thread(lambda: pyautogui.screenshot(str(path)))

    async def get_screen_size(self) -> tuple[int, int]:
        import pyautogui
        return await asyncio.to_thread(lambda: tuple(pyautogui.size()))


class GuiControlTentacle(Tentacle):
    def __init__(self, backend: GuiBackend, screenshot_dir: str | Path):
        self._backend = backend
        self._screenshot_dir = Path(screenshot_dir)

    @property
    def name(self) -> str:
        return "gui_control"

    @property
    def description(self) -> str:
        return ("Desktop GUI control: click, move, type, press, screenshot, "
                "screen_size. Honest non-sandbox — Krakey can drive the "
                "actual mouse and keyboard. PyAutoGUI fail-safe (corner "
                "abort) stays on.")

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "action": "click|move|type|press|screenshot|screen_size",
            "x": "int (click/move)",
            "y": "int (click/move)",
            "text": "str (type)",
            "key": "str (press, e.g. 'enter')",
            "button": "left|right|middle (click, default left)",
        }

    @property
    def is_internal(self) -> bool:
        return True

    async def execute(self, intent: str,
                        params: dict[str, Any]) -> Stimulus:
        action = (params.get("action") or "").lower()
        try:
            if action == "click":
                return await self._click(params)
            if action == "move":
                return await self._move(params)
            if action == "type":
                return await self._type(params)
            if action == "press":
                return await self._press(params)
            if action == "screenshot":
                return await self._screenshot()
            if action == "screen_size":
                return await self._screen_size()
        except Exception as e:  # noqa: BLE001
            return self._stim(f"GUI error: {e}", adrenalin=True)
        return self._stim(
            f"Unknown action: {action!r}. "
            "Supported: click/move/type/press/screenshot/screen_size."
        )

    async def _click(self, p):
        x, y = p.get("x"), p.get("y")
        if x is None or y is None:
            return self._stim("click: missing required params x, y.")
        button = p.get("button", "left")
        await self._backend.click(int(x), int(y), button=button)
        return self._stim(f"click {button} at ({x}, {y}) ok")

    async def _move(self, p):
        x, y = p.get("x"), p.get("y")
        if x is None or y is None:
            return self._stim("move: missing required params x, y.")
        await self._backend.move(int(x), int(y))
        return self._stim(f"move to ({x}, {y}) ok")

    async def _type(self, p):
        text = p.get("text")
        if text is None:
            return self._stim("type: missing required param text.")
        interval = float(p.get("interval", 0.0))
        await self._backend.type_text(text, interval=interval)
        return self._stim(f"typed {len(text)} chars")

    async def _press(self, p):
        key = p.get("key")
        if not key:
            return self._stim("press: missing required param key.")
        await self._backend.press(key)
        return self._stim(f"pressed {key!r}")

    async def _screenshot(self):
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = self._screenshot_dir / f"shot-{ts}.png"
        await self._backend.screenshot(path)
        return self._stim(f"screenshot saved: {path}")

    async def _screen_size(self):
        w, h = await self._backend.get_screen_size()
        return self._stim(f"screen size: {w}x{h}")

    def _stim(self, content: str, *, adrenalin: bool = False) -> Stimulus:
        return Stimulus(
            type="tentacle_feedback",
            source=f"tentacle:{self.name}",
            content=content,
            timestamp=datetime.now(),
            adrenalin=adrenalin,
        )


def create_tentacle(config: dict, deps: dict) -> Tentacle:
    return GuiControlTentacle(
        backend=PyAutoGUIBackend(),
        screenshot_dir=str(config.get("screenshot_dir",
                                           "workspace/screenshots")),
    )
