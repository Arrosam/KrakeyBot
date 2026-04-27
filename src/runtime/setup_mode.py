"""Setup-mode idle loop — runtime banner + sleep when config is incomplete.

When the user hasn't bound the ``self_thinking`` core purpose tag,
Krakey can't actually think. We still start the dashboard so the user
can finish configuration via Web UI; this module is the loop that
keeps the process alive in the meantime.

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
    "  config.yaml — the heartbeat is paused. Open the Web UI\n"
    "  to finish setup:\n\n"
    "      http://{host}:{port}\n\n"
    "  Then in the LLM section: add a provider, define a tag,\n"
    "  bind core_purposes.self_thinking + embedding. Save +\n"
    "  Restart. The next boot will run the real heartbeat.\n"
    "=========================================================\n"
)


async def run_setup_mode(runtime: "Runtime") -> None:
    """Print the setup banner and idle until ``runtime._stop`` flips.

    Polls every second so the user's Ctrl-C / kill / dashboard
    /api/restart shows up promptly.
    """
    cfg = runtime.config.dashboard
    runtime.log.runtime_error(_BANNER.format(host=cfg.host, port=cfg.port))
    while not runtime._stop:
        await asyncio.sleep(1.0)
