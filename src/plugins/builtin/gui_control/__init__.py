"""Built-in `gui_control` plugin — PyAutoGUI mouse / keyboard / screenshot."""
from __future__ import annotations

from src.interfaces.tentacle import Tentacle
from src.tentacles.gui_control import GuiControlTentacle, PyAutoGUIBackend


MANIFEST = {
    "name": "gui_control",
    "description": "Drive a desktop (click / type / press / screenshot). "
                   "Disabled by default; enable only when you want "
                   "Krakey operating a visible display.",
    "is_internal": True,
    "config_schema": [
        {"field": "enabled",        "type": "bool", "default": False,
         "help": "Disabled by default — dangerous on the host, "
                 "intended to target the sandbox VM once GUI routing "
                 "lands in sandbox Phase S2."},
        {"field": "sandbox",        "type": "bool", "default": True,
         "help": "Reserved for Phase S2; currently ignored (always runs "
                 "on host PyAutoGUI)."},
        {"field": "screenshot_dir", "type": "text",
         "default": "workspace/screenshots"},
    ],
}


def create_tentacle(config: dict, deps: dict) -> Tentacle:
    return GuiControlTentacle(
        backend=PyAutoGUIBackend(),
        screenshot_dir=str(config.get("screenshot_dir",
                                           "workspace/screenshots")),
    )
