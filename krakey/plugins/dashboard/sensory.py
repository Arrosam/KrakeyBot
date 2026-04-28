"""Web chat sensory — converts inbound chat messages to stimuli.

This sensory is *just* a sensory: ``start(push)`` captures the runtime
push callback, ``push_user_message(text, attachments)`` builds a
``user_message`` Stimulus and ships it. The dashboard server itself is
started by the plugin's factory at registration time (see
``__init__.py``); the sensory only carries a reference to it so its
``stop()`` can shut the server down on the same hook the runtime uses
to stop sensories.

Cross-thread note: the chat WS handler runs in the dashboard server's
own asyncio loop (a daemon thread; see ``threaded_server.py``), but
the runtime queue's push() must run on the runtime's loop. We capture
runtime's loop in ``start()`` and use ``run_coroutine_threadsafe`` to
hop loops if ``push_user_message`` is invoked from a different one.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Protocol

from krakey.interfaces.sensory import PushCallback, Sensory
from krakey.models.stimulus import Stimulus


class _StoppableServer(Protocol):
    """Minimal shape the sensory needs from the dashboard server —
    a synchronous stop() it can run via ``asyncio.to_thread``."""

    def stop(self, timeout: float = ...) -> None: ...


class WebChatSensory(Sensory):
    """Pure inbound chat: capture push at start(), expose
    ``push_user_message`` for the chat WS handler."""

    def __init__(self) -> None:
        self._push: PushCallback | None = None
        self._runtime_loop: asyncio.AbstractEventLoop | None = None
        self._server: _StoppableServer | None = None

    @property
    def name(self) -> str:
        return "web_chat"

    def attach_server(self, server: _StoppableServer) -> None:
        """Attach the dashboard server so ``stop()`` shuts it down too.

        The plugin factory calls this after starting the server in the
        port>0 mode. In port=0 mode (tests) no server runs and this is
        never called; ``stop()`` then degrades to clearing references.
        """
        self._server = server

    async def start(self, push: PushCallback) -> None:
        self._push = push
        # Remember whose loop the push belongs to. push_user_message
        # may be called from the dashboard-server thread (different
        # loop), in which case we'll need to hop.
        self._runtime_loop = asyncio.get_running_loop()

    async def stop(self) -> None:
        self._push = None
        self._runtime_loop = None
        if self._server is not None:
            # server.stop() is synchronous (joins a daemon thread).
            # to_thread offloads it so the runtime loop isn't blocked
            # while uvicorn finishes WS close frames + in-flight HTTP.
            server = self._server
            self._server = None
            await asyncio.to_thread(server.stop)

    async def push_user_message(
        self, text: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> None:
        """Convert one inbound web-chat message to a Stimulus + push.

        Safe to call from any asyncio loop; cross-loop calls are
        scheduled on the runtime's loop via ``run_coroutine_threadsafe``.
        """
        if self._push is None or self._runtime_loop is None:
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
        stim = Stimulus(
            type="user_message",
            source=f"sensory:{self.name}",
            content=content,
            timestamp=datetime.now(),
            adrenalin=True,
            metadata=md,
        )

        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if current is self._runtime_loop:
            await self._push(stim)
        else:
            fut = asyncio.run_coroutine_threadsafe(
                self._push(stim), self._runtime_loop,
            )
            await asyncio.wrap_future(fut)
