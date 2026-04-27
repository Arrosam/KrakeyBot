"""Dashboard plugin sensory — Web UI server lifecycle + inbound chat.

Two classes:

  * ``WebChatSensory`` — minimal Sensory contract. Captures the push
    callback at start(), exposes ``push_user_message(text, attachments)``
    that the chat WS handler calls. No server lifecycle, so unit tests
    can exercise just the inbound-push path without binding a port.

  * ``DashboardSensory`` — the production sensory the dashboard plugin
    factory builds. Inherits WebChatSensory's push behavior AND owns
    the FastAPI/uvicorn lifecycle. ``start()`` spins up the Web UI
    server (editor pages + observation events WS + chat WS); ``stop()``
    tears it down. Replaces the old ``DashboardLifecycle`` that used to
    live in ``src/runtime/dashboard/``.

If port=0, ``DashboardSensory`` skips the server bind — useful for
tests that want the plugin's tentacle/sensory registered without
actually binding a port.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.interfaces.sensory import PushCallback, Sensory
from src.models.stimulus import Stimulus


class WebChatSensory(Sensory):
    """Pure inbound chat: captures push at start(), exposes
    ``push_user_message`` for the chat WS handler."""

    def __init__(self) -> None:
        self._push: PushCallback | None = None

    @property
    def name(self) -> str:
        return "web_chat"

    async def start(self, push: PushCallback) -> None:
        self._push = push

    async def stop(self) -> None:
        self._push = None

    async def push_user_message(
        self, text: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> None:
        """Convert one inbound web-chat message to a Stimulus + push."""
        if self._push is None:
            return  # pre-start or post-stop: silently drop
        content = text
        md: dict[str, Any] = {"channel": "web_chat"}
        if attachments:
            lines = [text] if text else []
            for a in attachments:
                name = a.get("name", "file")
                typ = a.get("type", "")
                size = a.get("size", 0)
                url = a.get("url", "")
                lines.append(f"[附件: {name} ({typ}, {size} bytes) {url}]")
            content = "\n".join(lines)
            md["attachments"] = attachments
        await self._push(Stimulus(
            type="user_message",
            source=f"sensory:{self.name}",
            content=content,
            timestamp=datetime.now(),
            adrenalin=True,
            metadata=md,
        ))


class DashboardSensory(WebChatSensory):
    """Production sensory: WebChatSensory + dashboard server lifecycle."""

    def __init__(
        self, *,
        runtime: Any,
        history: Any,
        host: str,
        port: int,
        plugin_configs_root: Path | str,
    ) -> None:
        super().__init__()
        self._runtime = runtime
        self._history = history
        self._host = host
        self._port = port
        self._plugin_configs_root = plugin_configs_root
        self._server: Any = None  # DashboardServer

    async def start(self, push: PushCallback) -> None:
        await super().start(push)
        await self._start_server()

    async def stop(self) -> None:
        await self._stop_server()
        await super().stop()

    # ---- server lifecycle ---------------------------------------------

    async def _start_server(self) -> None:
        # port=0 → register sensory + tentacle without binding a port
        # (used by tests; the dashboard plugin's tentacle is exercised
        # but no Web UI server runs).
        if int(self._port) == 0:
            return

        from src.plugins.dashboard.app_factory import (
            create_app as create_dashboard_app,
        )
        from src.plugins.dashboard.events import EventBroadcaster
        from src.plugins.dashboard.server import DashboardServer

        rt = self._runtime

        def on_restart() -> None:
            rt.log.hb("restart requested via dashboard — re-execing")
            os.execv(sys.executable, [sys.executable, *sys.argv])

        try:
            broadcaster = EventBroadcaster(rt.events)
            self._server = DashboardServer(
                create_dashboard_app(
                    runtime=rt,
                    web_chat_history=self._history,
                    on_user_message=self.push_user_message,
                    event_broadcaster=broadcaster,
                    config_path=(Path(rt._config_path)
                                  if rt._config_path else None),
                    on_restart=on_restart,
                    plugin_configs_root=self._plugin_configs_root,
                ),
                host=self._host, port=self._port,
            )
            await self._server.start()
            rt.log.hb(
                f"dashboard listening on http://{self._host}:"
                f"{self._server.port}"
            )
        except OSError as e:
            rt.log.runtime_error(
                f"dashboard failed to start (port {self._port} in use? {e}); "
                "runtime continues without dashboard"
            )
            self._server = None
        except Exception as e:  # noqa: BLE001
            rt.log.runtime_error(f"dashboard startup error: {e}")
            self._server = None

    async def _stop_server(self) -> None:
        if self._server is None:
            return
        try:
            await self._server.stop()
        except Exception as e:  # noqa: BLE001
            self._runtime.log.runtime_error(f"dashboard stop error: {e}")
        self._server = None
