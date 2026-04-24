"""Dataclass event \u2192 JSON-friendly dict, with `kind` discriminator.

Kept as a free function (not a class method) so event types outside
the dashboard package stay free of coupling to it.
"""
from __future__ import annotations

import dataclasses
from typing import Any

from src.runtime.event_bus import _BaseEvent


def serialize_event(event: _BaseEvent) -> dict[str, Any]:
    """Convert a bus event to a flat dict. The `kind` field is always
    first (lets clients dispatch without inspecting the whole payload)."""
    payload: dict[str, Any] = {"kind": event.kind}
    if dataclasses.is_dataclass(event):
        for f in dataclasses.fields(event):
            payload[f.name] = getattr(event, f.name)
    return payload
