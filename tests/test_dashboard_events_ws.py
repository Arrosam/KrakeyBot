"""Phase 3.F.4: /ws/events WS bridge from RuntimeEventBus to browser."""
import asyncio
import json
from collections import deque

import pytest
from fastapi.testclient import TestClient

from src.dashboard.app_factory import create_app
from src.dashboard.events import EventBroadcaster, serialize_event
from src.runtime.events.event_types import (
    DecisionEvent, GMStatsEvent, ThinkingEvent,
)
from src.runtime.events.event_bus import EventBus


def test_serialize_event_includes_kind_and_fields():
    e = ThinkingEvent(heartbeat_id=3, text="x")
    out = serialize_event(e)
    assert out["kind"] == "thinking"
    assert out["heartbeat_id"] == 3
    assert out["text"] == "x"


def test_serialize_event_handles_nested_payload():
    e = GMStatsEvent(heartbeat_id=1, node_count=10,
                       edge_count=5, fatigue_pct=20)
    out = serialize_event(e)
    assert out["kind"] == "gm_stats"
    assert out["node_count"] == 10


async def test_broadcaster_subscribes_to_bus_and_buffers_recent():
    bus = EventBus()
    bcast = EventBroadcaster(bus, history_size=10)
    bus.publish(ThinkingEvent(1, "a"))
    bus.publish(DecisionEvent(1, "b"))
    # Recent buffer keeps both
    recent = bcast.recent()
    assert len(recent) == 2
    assert recent[0]["kind"] == "thinking"


async def test_broadcaster_history_capped_at_size():
    bus = EventBus()
    bcast = EventBroadcaster(bus, history_size=3)
    for i in range(10):
        bus.publish(ThinkingEvent(i, f"t{i}"))
    recent = bcast.recent()
    assert len(recent) == 3
    assert [e["heartbeat_id"] for e in recent] == [7, 8, 9]


def test_ws_events_sends_recent_then_live():
    bus = EventBus()
    bcast = EventBroadcaster(bus)
    # Pre-load some history
    bus.publish(ThinkingEvent(1, "old1"))
    bus.publish(ThinkingEvent(2, "old2"))

    app = create_app(runtime=None, event_broadcaster=bcast)
    client = TestClient(app)
    with client.websocket_connect("/ws/events") as ws:
        first = ws.receive_json()
        assert first["kind"] == "history"
        assert len(first["events"]) == 2
        assert first["events"][1]["text"] == "old2"

        # New live event arrives after connect
        bus.publish(DecisionEvent(3, "fresh"))
        live = ws.receive_json()
        assert live["kind"] == "decision"
        assert live["text"] == "fresh"


def test_ws_events_multiple_clients_all_receive_live():
    bus = EventBus()
    bcast = EventBroadcaster(bus)
    app = create_app(runtime=None, event_broadcaster=bcast)
    client = TestClient(app)
    with client.websocket_connect("/ws/events") as ws1, \
         client.websocket_connect("/ws/events") as ws2:
        # Drain history frames
        ws1.receive_json()
        ws2.receive_json()
        bus.publish(ThinkingEvent(1, "broadcast"))
        m1 = ws1.receive_json()
        m2 = ws2.receive_json()
        assert m1["text"] == "broadcast"
        assert m2["text"] == "broadcast"
