"""Phase 3 / E: GUI control tentacle (PyAutoGUI-backed, mocked for tests)."""
from pathlib import Path

import pytest

from src.plugins.builtin.gui_control import GuiControlTentacle


class FakeBackend:
    def __init__(self, screen_size=(1920, 1080), screenshot_bytes=b"fakepng",
                  raises=None):
        self.screen_size = screen_size
        self._screenshot_bytes = screenshot_bytes
        self._raises = raises
        self.actions: list[tuple[str, tuple, dict]] = []

    def _maybe_raise(self):
        if self._raises is not None:
            raise self._raises

    async def click(self, x, y, *, button="left"):
        self.actions.append(("click", (x, y), {"button": button}))
        self._maybe_raise()

    async def move(self, x, y):
        self.actions.append(("move", (x, y), {}))
        self._maybe_raise()

    async def type_text(self, text, *, interval=0.0):
        self.actions.append(("type", (text,), {"interval": interval}))
        self._maybe_raise()

    async def press(self, key):
        self.actions.append(("press", (key,), {}))
        self._maybe_raise()

    async def screenshot(self, path):
        self.actions.append(("screenshot", (str(path),), {}))
        self._maybe_raise()
        Path(path).write_bytes(self._screenshot_bytes)

    async def get_screen_size(self):
        self._maybe_raise()
        return self.screen_size


def test_metadata():
    t = GuiControlTentacle(backend=FakeBackend(), screenshot_dir="x")
    assert t.name == "gui_control"
    assert t.description
    assert t.is_internal is True


async def test_click_invokes_backend(tmp_path):
    backend = FakeBackend()
    t = GuiControlTentacle(backend=backend, screenshot_dir=tmp_path)
    stim = await t.execute("click", {"action": "click", "x": 100, "y": 200})
    assert backend.actions[0][0] == "click"
    assert backend.actions[0][1] == (100, 200)
    assert "click" in stim.content.lower()


async def test_type_text(tmp_path):
    backend = FakeBackend()
    t = GuiControlTentacle(backend=backend, screenshot_dir=tmp_path)
    await t.execute("type", {"action": "type", "text": "hello world"})
    assert backend.actions[0][0] == "type"
    assert backend.actions[0][1] == ("hello world",)


async def test_press_key(tmp_path):
    backend = FakeBackend()
    t = GuiControlTentacle(backend=backend, screenshot_dir=tmp_path)
    await t.execute("press", {"action": "press", "key": "enter"})
    assert backend.actions[0] == ("press", ("enter",), {})


async def test_move(tmp_path):
    backend = FakeBackend()
    t = GuiControlTentacle(backend=backend, screenshot_dir=tmp_path)
    await t.execute("move", {"action": "move", "x": 50, "y": 75})
    assert backend.actions[0] == ("move", (50, 75), {})


async def test_screenshot_saves_to_dir(tmp_path):
    backend = FakeBackend(screenshot_bytes=b"PNGDATA")
    t = GuiControlTentacle(backend=backend, screenshot_dir=tmp_path)
    stim = await t.execute("snap", {"action": "screenshot"})
    files = list(tmp_path.glob("*.png"))
    assert len(files) == 1
    assert files[0].read_bytes() == b"PNGDATA"
    assert str(files[0]) in stim.content or files[0].name in stim.content


async def test_screen_size(tmp_path):
    backend = FakeBackend(screen_size=(2560, 1440))
    t = GuiControlTentacle(backend=backend, screenshot_dir=tmp_path)
    stim = await t.execute("size", {"action": "screen_size"})
    assert "2560" in stim.content
    assert "1440" in stim.content


async def test_unknown_action_returns_error(tmp_path):
    t = GuiControlTentacle(backend=FakeBackend(), screenshot_dir=tmp_path)
    stim = await t.execute("dance", {"action": "dance"})
    assert "unknown" in stim.content.lower() or "unsupported" in stim.content.lower()


async def test_missing_required_param_returns_error(tmp_path):
    t = GuiControlTentacle(backend=FakeBackend(), screenshot_dir=tmp_path)
    stim = await t.execute("click", {"action": "click"})  # no x/y
    assert "missing" in stim.content.lower() or "required" in stim.content.lower()


async def test_backend_failure_returns_adrenalin_stimulus(tmp_path):
    backend = FakeBackend(raises=RuntimeError("display gone"))
    t = GuiControlTentacle(backend=backend, screenshot_dir=tmp_path)
    stim = await t.execute("click", {"action": "click", "x": 1, "y": 1})
    assert stim.adrenalin is True
    assert "display gone" in stim.content
