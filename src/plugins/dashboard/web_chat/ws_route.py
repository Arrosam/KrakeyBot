"""/ws/chat WebSocket endpoint.

Lifecycle:
  1. On connect, push the full history snapshot.
  2. Subscribe to broadcasts: every append (user-side or krakey-side)
     fans out to this socket.
  3. Receive loop: persist inbound user messages + hand them to the
     service's dispatcher. Silent on dispatch errors (the message is
     already durable by the time we get there).
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.plugins.dashboard.services.web_chat import WebChatService


_log = logging.getLogger(__name__)


def register(app: FastAPI, *, service: WebChatService) -> None:

    @app.websocket("/ws/chat")
    async def chat_ws(ws: WebSocket):  # noqa: ANN201
        await ws.accept()
        history = service.history
        await ws.send_json(
            {"kind": "history", "messages": history.all_messages()}
        )

        async def _send(msg):
            try:
                await ws.send_json({"kind": "message", "message": msg})
            except Exception:  # noqa: BLE001
                # Client gone — the recv loop will catch disconnect
                # on its next round.
                pass

        history.subscribe(_send)
        try:
            while True:
                data = await ws.receive_json()
                text = (data.get("text") or "").strip()
                attachments = data.get("attachments") or []
                if not text and not attachments:
                    continue
                await service.receive_user_message(text, attachments)
        except WebSocketDisconnect:
            pass
        except Exception as e:  # noqa: BLE001
            # Don't kill the WS process on unexpected errors, but DO
            # leave a breadcrumb — silent swallow makes WS bugs
            # invisible during debugging.
            _log.warning("chat ws unexpected error: %r", e)
        finally:
            history.unsubscribe(_send)
