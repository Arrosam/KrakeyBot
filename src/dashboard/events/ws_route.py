"""/ws/events WebSocket endpoint.

On connect: replay the recent-events ring buffer so the client
repaints timelines instantly. Then subscribe the socket; the
broadcaster pushes every new event until the socket disconnects.
"""
from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.dashboard.services.events import EventBroadcasterService


def register(app: FastAPI, *, broadcaster: EventBroadcasterService) -> None:

    @app.websocket("/ws/events")
    async def events_ws(ws: WebSocket):  # noqa: ANN201
        await ws.accept()
        await ws.send_json({"kind": "history",
                              "events": broadcaster.recent()})

        async def _send(msg):
            await ws.send_json(msg)

        broadcaster.add_socket(_send)
        try:
            while True:
                # Just keep the socket alive; events are server-pushed.
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001
            pass
        finally:
            broadcaster.remove_socket(_send)
