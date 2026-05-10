"""Runtime-owned built-in Tools — registered before plugin loader runs.

These are NOT plugins. The CLAUDE.md "plugins are strictly additive"
invariant says disabling/removing any plugin must not break the
runtime's core loop. The flip-side is that some capabilities are
load-bearing for the runtime itself and shouldn't be optional via
plugin enable/disable. Sleep is one such capability — Self must be
able to choose sleep without depending on any plugin.

Built-in tools are registered directly into ``Runtime.tools`` in the
composition root. They appear in ``[CAPABILITIES]`` like any other
tool so Self learns about them through the same channel.
"""
from krakey.runtime.builtin_tools.sleep_tool import (
    SLEEP_TOOL_NAME,
    SleepTool,
)

__all__ = ["SLEEP_TOOL_NAME", "SleepTool"]
