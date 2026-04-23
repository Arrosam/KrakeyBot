"""Phase 3.F.3: web chat history + tentacle + WS endpoint."""
import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.dashboard.server import create_app
from src.dashboard.web_chat import WebChatHistory
from src.plugins.builtin.web_chat.tentacle import WebChatTentacle


# ---------------- WebChatHistory ----------------

async def test_history_append_persists_to_disk(tmp_path):
    h = WebChatHistory(tmp_path / "chat.jsonl")
    await h.append("user", "hello")
    await h.append("krakey", "hi")

    lines = (tmp_path / "chat.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["content"] == "hello"
    assert json.loads(lines[1])["sender"] == "krakey"


async def test_history_loads_existing_file_on_init(tmp_path):
    p = tmp_path / "chat.jsonl"
    p.write_text(
        json.dumps({"sender": "user", "content": "old", "ts": "2026"}) + "\n"
        + json.dumps({"sender": "krakey", "content": "older", "ts": "2026"}) + "\n",
        encoding="utf-8",
    )
    h = WebChatHistory(p)
    msgs = h.all_messages()
    assert [m["content"] for m in msgs] == ["old", "older"]


async def test_history_subscribers_get_appended_messages(tmp_path):
    h = WebChatHistory(tmp_path / "chat.jsonl")
    received = []
    h.subscribe(received.append)
    await h.append("user", "hi")
    assert len(received) == 1
    assert received[0]["content"] == "hi"


async def test_history_async_subscriber_supported(tmp_path):
    h = WebChatHistory(tmp_path / "chat.jsonl")
    received = []

    async def cb(msg):
        received.append(msg)

    h.subscribe(cb)
    await h.append("user", "async hi")
    assert received == [{"sender": "user", "content": "async hi",
                         "ts": received[0]["ts"]}]


async def test_history_subscriber_exception_does_not_break_append(tmp_path):
    h = WebChatHistory(tmp_path / "chat.jsonl")
    survivor = []

    def boom(_): raise RuntimeError("ws gone")
    def good(m): survivor.append(m)

    h.subscribe(boom)
    h.subscribe(good)
    await h.append("user", "hi")
    assert len(survivor) == 1
    # Persistence still happened
    assert (tmp_path / "chat.jsonl").exists()


async def test_unsubscribe_stops_delivery(tmp_path):
    h = WebChatHistory(tmp_path / "chat.jsonl")
    seen = []
    cb = seen.append
    h.subscribe(cb)
    await h.append("user", "1")
    h.unsubscribe(cb)
    await h.append("user", "2")
    assert len(seen) == 1


# ---------------- WebChatSensory ----------------


async def test_sensory_push_creates_user_message_stimulus(tmp_path):
    from src.plugins.builtin.web_chat.sensory import WebChatSensory

    pushed = []

    class _Buf:
        async def push(self, s): pushed.append(s)

    sens = WebChatSensory()
    await sens.start(_Buf())
    await sens.push_user_message("hello krakey")
    assert len(pushed) == 1
    s = pushed[0]
    assert s.type == "user_message"
    assert s.source == "sensory:web_chat"
    assert s.content == "hello krakey"
    assert s.adrenalin is True
    assert s.metadata["channel"] == "web_chat"


async def test_sensory_push_appends_attachment_notices(tmp_path):
    from src.plugins.builtin.web_chat.sensory import WebChatSensory

    pushed = []

    class _Buf:
        async def push(self, s): pushed.append(s)

    sens = WebChatSensory()
    await sens.start(_Buf())
    await sens.push_user_message(
        "see file",
        attachments=[{"name": "a.png", "type": "image/png",
                         "size": 123, "url": "/u/a.png"}],
    )
    s = pushed[0]
    assert "see file" in s.content
    assert "[附件: a.png" in s.content
    assert s.metadata["attachments"][0]["name"] == "a.png"


async def test_sensory_push_before_start_silently_drops():
    from src.plugins.builtin.web_chat.sensory import WebChatSensory

    sens = WebChatSensory()
    # No start() — buffer is None. Must not raise.
    await sens.push_user_message("dropped")


async def test_sensory_push_after_stop_silently_drops():
    from src.plugins.builtin.web_chat.sensory import WebChatSensory

    pushed = []

    class _Buf:
        async def push(self, s): pushed.append(s)

    sens = WebChatSensory()
    await sens.start(_Buf())
    await sens.stop()
    await sens.push_user_message("dropped")
    assert pushed == []


# ---------------- WebChatTentacle ----------------

def test_tentacle_metadata():
    t = WebChatTentacle(history=None)  # noqa
    assert t.name == "web_chat_reply"
    assert t.is_internal is False  # outbound chat to a human


async def test_tentacle_send_appends_to_history(tmp_path):
    h = WebChatHistory(tmp_path / "chat.jsonl")
    t = WebChatTentacle(history=h)
    stim = await t.execute("hello world", {"text": "hello world"})
    msgs = h.all_messages()
    assert msgs[0]["sender"] == "krakey"
    assert msgs[0]["content"] == "hello world"
    assert "sent" in stim.content.lower() or "已发送" in stim.content


async def test_tentacle_intent_used_when_no_text_param(tmp_path):
    h = WebChatHistory(tmp_path / "chat.jsonl")
    t = WebChatTentacle(history=h)
    await t.execute("free-form intent", {})
    assert h.all_messages()[0]["content"] == "free-form intent"


async def test_tentacle_empty_text_returns_clear_msg(tmp_path):
    h = WebChatHistory(tmp_path / "chat.jsonl")
    t = WebChatTentacle(history=h)
    stim = await t.execute("", {"text": "   "})
    assert h.all_messages() == []
    assert "empty" in stim.content.lower() or "空" in stim.content


async def test_tentacle_history_failure_returns_adrenalin_error(tmp_path):
    class BrokenHistory:
        async def append(self, sender, content):
            raise RuntimeError("disk full")

    t = WebChatTentacle(history=BrokenHistory())
    stim = await t.execute("hi", {"text": "hi"})
    assert stim.adrenalin is True
    assert "disk full" in stim.content


# ---------------- WS endpoint ----------------

def test_ws_chat_sends_full_history_on_connect(tmp_path):
    h = WebChatHistory(tmp_path / "chat.jsonl")
    asyncio.run(h.append("user", "old1"))
    asyncio.run(h.append("krakey", "old2"))

    app = create_app(runtime=None, web_chat_history=h)
    client = TestClient(app)
    with client.websocket_connect("/ws/chat") as ws:
        first = ws.receive_json()
        assert first["kind"] == "history"
        contents = [m["content"] for m in first["messages"]]
        assert contents == ["old1", "old2"]


def test_ws_chat_user_message_pushes_to_runtime_callback(tmp_path):
    h = WebChatHistory(tmp_path / "chat.jsonl")
    received_user_msgs = []

    async def on_user_message(text, attachments=None):
        received_user_msgs.append(text)

    app = create_app(runtime=None, web_chat_history=h,
                       on_user_message=on_user_message)
    client = TestClient(app)
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # discard initial history
        ws.send_json({"text": "hello krakey"})
        # Allow callback dispatch
        msg = ws.receive_json()
        assert msg["kind"] == "message"
        assert msg["message"]["content"] == "hello krakey"
        assert msg["message"]["sender"] == "user"
    assert received_user_msgs == ["hello krakey"]


def test_ws_chat_krakey_messages_broadcast_to_clients(tmp_path):
    h = WebChatHistory(tmp_path / "chat.jsonl")
    app = create_app(runtime=None, web_chat_history=h)
    client = TestClient(app)
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()  # initial history (empty)
        # Tentacle writes to history → all subscribed sockets receive
        asyncio.run(h.append("krakey", "from server"))
        msg = ws.receive_json()
        assert msg["kind"] == "message"
        assert msg["message"]["content"] == "from server"
