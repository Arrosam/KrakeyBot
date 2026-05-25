"""Read-receipt handler — flips history records to "read" when the
runtime signals that a stimulus has been drained from the buffer.

``make_stimulus_read_handler(history)`` returns an async callable
suitable for ``EventBus.subscribe()``. When a ``StimulusReadEvent``
arrives it looks up every chat_message_id in the history and updates
its persisted status to "read".
"""
from __future__ import annotations

from krakey.runtime.events.event_types import StimulusReadEvent


def make_stimulus_read_handler(history):
    """Return an EventBus-subscribable async handler that persists
    "read" status for every chat_message_id carried by
    ``StimulusReadEvent``."""

    async def handler(event) -> None:
        if isinstance(event, StimulusReadEvent):
            for mid in event.chat_message_ids:
                await history.update_status(mid, "read")

    return handler
