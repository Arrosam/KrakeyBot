"""Heartbeat-driven unread-reminder handler for web chat.

``make_heartbeat_reminder_handler(history, push_reminder)`` returns an
async EventBus handler. On each HeartbeatStartEvent it checks for
unread web-chat messages (status == 'delivered') and, if any exist,
pushes one compact non-adrenalin system_event reminder via the
push_reminder callable.
"""
from __future__ import annotations
from datetime import datetime
from krakey.models.stimulus import Stimulus
from krakey.runtime.events.event_types import HeartbeatStartEvent


def make_heartbeat_reminder_handler(history, push_reminder):
    """EventBus handler: on each HeartbeatStartEvent, if there are unread
    web-chat messages (status=='delivered'), push ONE compact non-adrenalin
    system_event reminder via push_reminder (an async (Stimulus)->None)."""
    async def handler(event) -> None:
        if not isinstance(event, HeartbeatStartEvent):
            return
        unread = [m for m in history.all_messages() if m.get("status") == "delivered"]
        if not unread:
            return
        n = len(unread)
        stim = Stimulus(
            type="system_event",
            source="channel:web_chat",
            content=f"You have {n} unread web-chat message(s).",
            timestamp=datetime.now(),
            adrenalin=False,
        )
        await push_reminder(stim)
    return handler
