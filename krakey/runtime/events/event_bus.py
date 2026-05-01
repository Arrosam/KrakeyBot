"""Runtime event bus for dashboard / external observers.

The Logger stays the runtime's primary console output path. This bus is the
*additional* channel a Dashboard subscribes to. Runtime publishes typed
events (see ``event_types.py``); subscribers do whatever (broadcast WS,
accumulate metrics, ignore).

Subscribers that raise are logged + quarantined; they cannot block runtime
progress. Async subscribers are scheduled via ``asyncio.create_task`` when an
event loop is running, so publish never awaits.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable

from krakey.runtime.events.event_types import _BaseEvent

_log = logging.getLogger(__name__)


Subscriber = Callable[[_BaseEvent], Any]


class EventBus:
    def __init__(self):
        self._subs: list[Subscriber] = []

    def subscribe(self, callback: Subscriber) -> None:
        self._subs.append(callback)

    def unsubscribe(self, callback: Subscriber) -> None:
        try:
            self._subs.remove(callback)
        except ValueError:
            pass

    def publish(self, event: _BaseEvent) -> None:
        for cb in list(self._subs):
            try:
                if inspect.iscoroutinefunction(cb):
                    self._schedule(cb, event)
                else:
                    cb(event)
            except Exception:  # noqa: BLE001
                _log.exception("event subscriber raised; quarantining call")

    @staticmethod
    def _schedule(cb, event):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop — call sync (best-effort)
            try:
                asyncio.run(cb(event))
            except Exception:  # noqa: BLE001
                _log.exception("async subscriber failed (no loop)")
            return
        loop.create_task(cb(event))
