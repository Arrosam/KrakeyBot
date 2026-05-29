"""Read-receipt handler — flips history records to "read" when the
web_chat_mark_read tool confirms Self has read the messages.

``make_stimulus_read_handler(history)`` returns an async callable
suitable for ``EventBus.subscribe()``. When a ``StimulusReadEvent``
arrives (published by the web_chat_mark_read tool when Self uses it
to read and acknowledge messages) it looks up each message by ID in
the history and updates its persisted status to "read".
"""
from __future__ import annotations

from krakey.runtime.events.event_types import StimulusReadEvent


def make_stimulus_read_handler(history):
    """Return an EventBus-subscribable async handler that persists
    "read" status for every message ID carried by
    ``StimulusReadEvent``. The event is published by the
    web_chat_mark_read tool when Self confirms reading."""

    async def handler(event) -> None:
        if isinstance(event, StimulusReadEvent):
            for mid in event.chat_message_ids:
                await history.update_status(mid, "read")

    return handler
