"""Batch Tracker Sensory (DevSpec §5.5).

Tracks call_ids dispatched during a heartbeat. When every pending id has
been marked completed, pushes an adrenalin `batch_complete` stimulus so
Self wakes up to review the results. Supports extend_batch for mid-flight
dispatches so old stragglers don't false-fire.
"""
from __future__ import annotations

from datetime import datetime

from krakey.interfaces.sensory import PushCallback, Sensory
from krakey.models.stimulus import Stimulus


class BatchTrackerSensory(Sensory):
    def __init__(self):
        self._pending: set[str] = set()
        self._push: PushCallback | None = None

    @property
    def name(self) -> str:
        return "batch_tracker"

    @property
    def default_adrenalin(self) -> bool:
        return True

    async def start(self, push: PushCallback) -> None:
        self._push = push

    async def stop(self) -> None:
        # No background task to cancel.
        pass

    def register_batch(self, call_ids: list[str]) -> None:
        self._pending.update(call_ids)

    def extend_batch(self, new_ids: list[str]) -> None:
        self._pending.update(new_ids)

    async def mark_completed(self, call_id: str) -> None:
        if call_id not in self._pending:
            return
        self._pending.discard(call_id)
        if self._pending:
            return
        if self._push is None:
            return
        await self._push(Stimulus(
            type="batch_complete",
            source=f"sensory:{self.name}",
            content="All dispatched tentacles completed.",
            timestamp=datetime.now(),
            adrenalin=True,
        ))
