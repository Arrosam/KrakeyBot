"""Dashboard lifecycle — start + stop + restart wiring, extracted
from Runtime.

Three concerns:

  1. **start_if_enabled()** — build the FastAPI app from the
     ``runtime`` ref + the per-plugin services, bind it to
     host/port, and start serving. Failures (port in use, anything
     else) log + leave the dashboard absent rather than crashing
     the runtime.
  2. **stop()** — best-effort shutdown for ``Runtime.close()``.
  3. **on_restart callback** — re-exec the process so the next
     boot sees edited config.

Holds the ``DashboardServer`` instance and the ``runtime`` ref;
exposes ``server`` for callers that need to read the bound port
(useful in tests).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from src.plugins.dashboard.app_factory import create_app as create_dashboard_app
from src.plugins.dashboard.events import EventBroadcaster
from src.plugins.dashboard.server import DashboardServer

if TYPE_CHECKING:
    from src.runtime.runtime import Runtime


class DashboardLifecycle:
    """Owns the optional dashboard server + the start/stop sequence."""

    def __init__(self, runtime: "Runtime"):
        self._rt = runtime
        self.server: DashboardServer | None = None

    async def start_if_enabled(self) -> None:
        """No-op when ``config.dashboard.enabled`` is False; otherwise
        wire the FastAPI app + start it. Failures log + leave
        ``self.server`` None so the heartbeat keeps running."""
        rt = self._rt
        cfg = getattr(rt.config, "dashboard", None)
        if cfg is None or not cfg.enabled:
            return

        # Route inbound user messages through the web_chat sensory
        # (plugin-registered). If the plugin is disabled, no sensory
        # exists — install a noop callback that logs the drop so the
        # dashboard can still show history in monitor-only mode.
        web_chat_sensory = rt.buffer.get_sensory("web_chat")
        if web_chat_sensory is not None:
            on_user_message = web_chat_sensory.push_user_message
        else:
            async def on_user_message(text: str,
                                          attachments: list[dict] | None = None,
                                          ) -> None:
                rt.log.hb_warn(
                    "web_chat plugin disabled — dropping inbound message "
                    f"({len(text or '')} chars)"
                )

        def on_restart() -> None:
            """Re-exec process so a new instance picks up edited config."""
            rt.log.hb("restart requested via dashboard — re-execing")
            os.execv(sys.executable, [sys.executable, *sys.argv])

        try:
            broadcaster = EventBroadcaster(rt.events)
            self.server = DashboardServer(
                create_dashboard_app(
                    runtime=rt,
                    web_chat_history=rt.web_chat_history,
                    on_user_message=on_user_message,
                    event_broadcaster=broadcaster,
                    config_path=(Path(rt._config_path)
                                  if rt._config_path else None),
                    on_restart=on_restart,
                    plugin_configs_root=rt._plugin_configs_root,
                ),
                host=cfg.host, port=cfg.port,
            )
            await self.server.start()
            rt.log.hb(
                f"dashboard listening on http://{cfg.host}:{self.server.port}"
            )
        except OSError as e:
            rt.log.runtime_error(
                f"dashboard failed to start (port {cfg.port} in use? {e}); "
                "runtime continues without dashboard"
            )
            self.server = None
        except Exception as e:  # noqa: BLE001
            rt.log.runtime_error(f"dashboard startup error: {e}")
            self.server = None

    async def stop(self) -> None:
        """Best-effort shutdown — called from Runtime.close(). Errors
        log but never raise; close() must continue with GM/KB cleanup."""
        if self.server is None:
            return
        try:
            await self.server.stop()
        except Exception as e:  # noqa: BLE001
            self._rt.log.runtime_error(f"dashboard stop error: {e}")
        self.server = None
