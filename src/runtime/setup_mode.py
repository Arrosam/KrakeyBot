"""Setup-mode idle loop — runtime banner + sleep when config is incomplete.

When the user hasn't bound the ``self_thinking`` core purpose tag,
Krakey can't actually think. We still keep the process alive so the
dashboard plugin (if enabled) can surface the editor; this module is
the loop that holds the process up in the meantime.

Separated from Runtime because the loop is a self-contained mode —
the runtime composition root doesn't need to know how the banner is
printed or how often we poll the stop flag.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.runtime.runtime import Runtime


_BANNER = (
    "\n=========================================================\n"
    "  Krakey is in SETUP MODE.\n\n"
    "  No `core_purposes.self_thinking` tag binding found in\n"
    "  config.yaml — the heartbeat is paused. Edit config.yaml:\n\n"
    "      1. Make sure 'dashboard' is in your plugins: list\n"
    "         (so you have a Web UI to edit settings in).\n"
    "      2. In the LLM section: add a provider, define a tag,\n"
    "         bind core_purposes.self_thinking + embedding.\n"
    "      3. Save + Restart.\n"
    "=========================================================\n"
)


async def run_setup_mode(runtime: "Runtime") -> None:
    """Print the setup banner and idle until ``runtime._stop`` flips.

    Polls every second so the user's Ctrl-C / kill / dashboard
    /api/restart shows up promptly.
    """
    runtime.log.runtime_error(_BANNER)
    while not runtime._stop:
        await asyncio.sleep(1.0)
