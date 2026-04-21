"""Plugin loader — dynamic tentacle/sensory discovery from workspace.

At boot Runtime scans `workspace/tentacles/` and `workspace/sensories/`,
imports each plugin in isolation (no sys.path pollution), and registers
the result alongside the built-in tentacles/sensories.

Plugin contract lives in PLUGINS.md at repo root.
"""
from src.plugins.loader import (  # noqa: F401
    PluginInfo, discover_sensories, discover_tentacles,
)
