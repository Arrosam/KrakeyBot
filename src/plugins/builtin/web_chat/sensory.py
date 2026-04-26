"""Web Chat sensory — dashboard → Krakey inbound.

Passive: the dashboard WebSocket handler hands us user messages via
`push_user_message`; we turn each one into a `user_message` Stimulus
and drop it on the buffer. Mirrors the outbound `WebChatTentacle` so
the whole channel is one plugin project (`web_chat`) that can be
enabled/disabled as a unit.

If the plugin is disabled the sensory never registers; the runtime
installs a noop `on_user_message` on the dashboard so typed messages
are dropped with a log warning instead of crashing.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from src.interfaces.sensory import PushCallback, Sensory
from src.models.stimulus import Stimulus


class WebChatSensory(Sensory):
    def __init__(self) -> None:
        self._push: PushCallback | None = None

    @property
    def name(self) -> str:
        return "web_chat"

    async def start(self, push: PushCallback) -> None:
        # Passive sensory — no background loop to spawn. Just capture
        # the push callback so the dashboard callback can ship inbound
        # messages into the runtime queue.
        self._push = push

    async def stop(self) -> None:
        self._push = None

    async def push_user_message(self, text: str,
                                   attachments: list[dict[str, Any]] | None
                                   = None) -> None:
        """Called by the dashboard WS handler on every inbound message."""
        if self._push is None:
            # Pre-start or post-stop — silently drop; the runtime owns
            # the lifecycle and any misuse is a programmer error, not
            # user data worth surfacing.
            return
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
