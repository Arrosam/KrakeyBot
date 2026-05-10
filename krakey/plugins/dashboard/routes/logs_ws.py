"""/ws/logs WebSocket — live runtime log stream.

On connect: replay the most-recent N lines from the LogCapture's
ring buffer so a freshly-opened tab repaints instantly. After that
push lines as they arrive. Token-gated like every other WS in the
dashboard.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from krakey.plugins.dashboard.auth import ws_check_token
from krakey.plugins.dashboard.log_capture import LogCapture


_log = logging.getLogger(__name__)


def register(
    app: FastAPI,
    *,
    capture: LogCapture,
    auth_token: str | None = None,
) -> None:

    @app.websocket("/ws/logs")
    async def logs_ws(ws: WebSocket):  # noqa: ANN201
        if not await ws_check_token(ws, auth_token):
            return
        await ws.accept()
        loop = asyncio.get_running_loop()

        try:
            await ws.send_json({"kind": "history", "lines": capture.recent()})
        except Exception:  # noqa: BLE001
            return

        async def _send(line: str) -> None:
            try:
                await ws.send_json({"kind": "line", "line": line})
            except Exception:  # noqa: BLE001
                pass

        def _on_line(line: str) -> None:
            # The tee is called on whatever thread does print() — usually
            # the runtime's main thread, occasionally the dashboard's.
            # Hop onto the dashboard loop to do the actual send.
            try:
                asyncio.run_coroutine_threadsafe(_send(line), loop)
            except RuntimeError:
                # Loop closed; the next subscriber-cleanup pass will
                # remove us.
                pass

        capture.subscribe(_on_line)
        try:
            while True:
                # Server-pushed; client doesn't send anything but we
                # keep the recv loop alive so disconnect surfaces
                # promptly.
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception as e:  # noqa: BLE001
            _log.warning("logs ws unexpected error: %r", e)
        finally:
            capture.unsubscribe(_on_line)
