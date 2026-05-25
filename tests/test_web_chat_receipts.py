"""Web-chat delivery/read receipts — backend contract.

Covers the loop-independent core of the feature:
  - WebChatChannel.push_user_message returns bool + stamps chat_message_id
  - WebChatHistory status persistence: append(status), update_status, delete
  - RuntimeWebChatService.receive_user_message -> {id, status, reason?}
  - the bus subscriber that flips history records to "read" on StimulusReadEvent
  - StimulusReadEvent serializes for the /ws/events stream the frontend reads
  - /ws/chat history snapshot + message echo carry id + status; resend deletes

The live "read" notification to the browser rides the existing /ws/events
broadcaster (cross-loop-correct already); persistence is the bus subscriber
below. The frontend (app.js/CSS/i18n) is verified separately in the browser.
"""
import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from krakey.plugins.dashboard.app_factory import create_app
from krakey.plugins.dashboard.web_chat import WebChatHistory


# ---------------- channel: push returns bool + stamps chat_message_id -------

async def test_push_returns_true_and_stamps_message_id():
    from krakey.plugins.dashboard.channel import WebChatChannel
    pushed = []

    class _Buf:
        async def push(self, s): pushed.append(s)

    ch = WebChatChannel()
    await ch.start(_Buf().push)
    ok = await ch.push_user_message("hi", message_id="m1")
    assert ok is True
    assert pushed[0].chat_message_id == "m1"


async def test_push_offline_returns_false_and_pushes_nothing():
    from krakey.plugins.dashboard.channel import WebChatChannel
    ch = WebChatChannel()  # never started -> offline
    ok = await ch.push_user_message("hi", message_id="m1")
    assert ok is False


# ---------------- history: status persistence -------------------------------

async def test_append_persists_id_and_status(tmp_path):
    h = WebChatHistory(tmp_path / "c.jsonl")
    rec = await h.append("user", "hi", message_id="m1", status="delivered")
    assert rec["id"] == "m1" and rec["status"] == "delivered"
    disk = json.loads(
        (tmp_path / "c.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert disk["id"] == "m1" and disk["status"] == "delivered"


async def test_append_without_status_stays_minimal(tmp_path):
    h = WebChatHistory(tmp_path / "c.jsonl")
    rec = await h.append("krakey", "reply")
    assert "id" not in rec and "status" not in rec


async def test_update_status_mutates_cache_and_disk(tmp_path):
    h = WebChatHistory(tmp_path / "c.jsonl")
    await h.append("user", "hi", message_id="m1", status="delivered")
    out = await h.update_status("m1", "read")
    assert out is not None and out["status"] == "read"
    assert h.all_messages()[0]["status"] == "read"
    disk = json.loads(
        (tmp_path / "c.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert disk["status"] == "read"


async def test_update_status_unknown_id_returns_none(tmp_path):
    h = WebChatHistory(tmp_path / "c.jsonl")
    assert await h.update_status("nope", "read") is None


async def test_delete_removes_record_from_cache_and_disk(tmp_path):
    h = WebChatHistory(tmp_path / "c.jsonl")
    await h.append("user", "a", message_id="m1", status="failed")
    await h.append("user", "b", message_id="m2", status="delivered")
    assert await h.delete("m1") is True
    assert [m.get("id") for m in h.all_messages()] == ["m2"]
    lines = (tmp_path / "c.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["id"] == "m2"


async def test_delete_unknown_id_returns_false(tmp_path):
    h = WebChatHistory(tmp_path / "c.jsonl")
    assert await h.delete("nope") is False


# ---------------- service: receive_user_message -> {id, status} -------------

def _svc(history, on_msg):
    from krakey.plugins.dashboard.web_chat.service import RuntimeWebChatService
    return RuntimeWebChatService(history, on_msg)


async def test_receive_delivered_when_push_succeeds(tmp_path):
    h = WebChatHistory(tmp_path / "c.jsonl")
    seen = []

    async def on_msg(text, attachments, message_id):
        seen.append((text, message_id))
        return True

    res = await _svc(h, on_msg).receive_user_message("hi", [])
    assert res["status"] == "delivered" and res["id"]
    assert seen[0][1] == res["id"]          # same id handed to the callback
    assert h.all_messages()[0]["status"] == "delivered"
    assert h.all_messages()[0]["id"] == res["id"]


async def test_receive_failed_when_no_callback(tmp_path):
    h = WebChatHistory(tmp_path / "c.jsonl")
    res = await _svc(h, None).receive_user_message("hi", [])
    assert res["status"] == "failed"
    assert h.all_messages()[0]["status"] == "failed"


async def test_receive_failed_when_push_returns_false(tmp_path):
    h = WebChatHistory(tmp_path / "c.jsonl")

    async def on_msg(text, attachments, message_id):
        return False

    res = await _svc(h, on_msg).receive_user_message("hi", [])
    assert res["status"] == "failed"
    assert h.all_messages()[0]["status"] == "failed"


async def test_receive_failed_when_push_raises(tmp_path):
    h = WebChatHistory(tmp_path / "c.jsonl")

    async def on_msg(text, attachments, message_id):
        raise RuntimeError("boom")

    res = await _svc(h, on_msg).receive_user_message("hi", [])
    assert res["status"] == "failed"
    assert h.all_messages()[0]["status"] == "failed"


# ---------------- bus subscriber -> mark history "read" ---------------------

async def test_stimulus_read_event_flips_history_to_read(tmp_path):
    from krakey.plugins.dashboard.web_chat.read_receipts import (
        make_stimulus_read_handler,
    )
    from krakey.runtime.events.event_bus import EventBus
    from krakey.runtime.events.event_types import (
        StimulusReadEvent, HeartbeatStartEvent,
    )

    h = WebChatHistory(tmp_path / "c.jsonl")
    await h.append("user", "a", message_id="m1", status="delivered")
    await h.append("user", "b", message_id="m2", status="delivered")

    bus = EventBus()
    bus.subscribe(make_stimulus_read_handler(h))
    # An unrelated event must NOT touch chat status.
    bus.publish(HeartbeatStartEvent(heartbeat_id=1, stimulus_count=0))
    bus.publish(StimulusReadEvent(chat_message_ids=["m1"]))
    await asyncio.sleep(0.02)  # let the async handler task run

    assert h.all_messages()[0]["status"] == "read"
    assert h.all_messages()[1]["status"] == "delivered"  # m2 untouched


# ---------------- read payload serializes for /ws/events --------------------

def test_stimulus_read_event_serializes_for_events_stream():
    from krakey.plugins.dashboard.events.serializer import serialize_event
    from krakey.runtime.events.event_types import StimulusReadEvent
    out = serialize_event(StimulusReadEvent(chat_message_ids=["m1", "m2"]))
    assert out == {"kind": "stimulus_read", "chat_message_ids": ["m1", "m2"]}


# ---------------- WS: snapshot + echo carry id+status; resend deletes -------

def test_ws_history_snapshot_carries_id_and_status(tmp_path):
    h = WebChatHistory(tmp_path / "c.jsonl")
    asyncio.run(h.append("user", "hi", message_id="m1", status="delivered"))
    app = create_app(runtime=None, web_chat_history=h)
    with TestClient(app).websocket_connect("/ws/chat") as ws:
        first = ws.receive_json()
        assert first["kind"] == "history"
        m = first["messages"][0]
        assert m["id"] == "m1" and m["status"] == "delivered"


def test_ws_user_message_echo_carries_status_and_id(tmp_path):
    h = WebChatHistory(tmp_path / "c.jsonl")

    async def on_user_message(text, attachments=None, message_id=None):
        return True

    app = create_app(runtime=None, web_chat_history=h,
                     on_user_message=on_user_message)
    with TestClient(app).websocket_connect("/ws/chat") as ws:
        ws.receive_json()                       # discard history
        ws.send_json({"text": "hello"})
        msg = ws.receive_json()
        assert msg["kind"] == "message"
        rec = msg["message"]
        assert rec["sender"] == "user"
        assert rec["status"] == "delivered"
        assert rec["id"]


def test_ws_resend_deletes_failed_entry(tmp_path):
    h = WebChatHistory(tmp_path / "c.jsonl")
    asyncio.run(h.append("user", "oops", message_id="m1", status="failed"))

    async def on_user_message(text, attachments=None, message_id=None):
        return True

    app = create_app(runtime=None, web_chat_history=h,
                     on_user_message=on_user_message)
    with TestClient(app).websocket_connect("/ws/chat") as ws:
        ws.receive_json()                       # discard history
        ws.send_json({"text": "oops", "resend_of": "m1"})
        ws.receive_json()                       # new message echo
        ids = [m.get("id") for m in h.all_messages()]
        assert "m1" not in ids                  # failed entry removed
        assert any(m["status"] == "delivered" for m in h.all_messages())
