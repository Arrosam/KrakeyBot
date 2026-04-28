"""events package — bus \u2192 /ws/events bridge.

Split into:
  - serializer.py : dataclass event \u2192 JSON dict (`kind` discriminator).
  - broadcaster.py: bus subscriber + per-socket fan-out + ring buffer.
  - ws_route.py   : the FastAPI WebSocket endpoint.

Re-exports kept here for the stable public API used by Runtime wiring
and tests.
"""
from krakey.plugins.dashboard.events.broadcaster import EventBroadcaster
from krakey.plugins.dashboard.events.serializer import serialize_event

__all__ = ["EventBroadcaster", "serialize_event"]
