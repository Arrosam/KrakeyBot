"""Phase 3 / D: Telegram inbound (Sensory) + outbound (Reply Tentacle)."""
import asyncio
from datetime import datetime

import pytest

from src.runtime.stimuli.stimulus_buffer import StimulusBuffer
from src.plugins.telegram.sensory import TelegramSensory
from src.plugins.telegram.tentacle import TelegramReplyTentacle


class FakeClient:
    def __init__(self, updates_batches=None, send_raises=None):
        self._batches = list(updates_batches or [])
        self._send_raises = send_raises
        self.sent: list[tuple[int, str]] = []
        self.update_calls: list[int] = []

    async def get_updates(self, offset, timeout=10):
        self.update_calls.append(offset)
        if not self._batches:
            await asyncio.sleep(0.05)
            return []
        batch = self._batches.pop(0)
        if isinstance(batch, Exception):
            raise batch
        return batch

    async def send_message(self, chat_id, text):
        if self._send_raises is not None:
            raise self._send_raises
        self.sent.append((chat_id, text))


def _msg(update_id, chat_id, text):
    return {
        "update_id": update_id,
        "message": {"chat": {"id": chat_id}, "text": text},
    }


# ---------------- TelegramSensory ----------------

async def test_sensory_pushes_incoming_message_as_user_stimulus():
    client = FakeClient(updates_batches=[
        [_msg(10, 123, "hello bot")],
    ])
    buf = StimulusBuffer()
    sensory = TelegramSensory(client=client)
    await sensory.start(buf.push)
    await asyncio.sleep(0.1)
    await sensory.stop()

    drained = buf.drain()
    assert any(s.content == "hello bot" for s in drained)
    assert any(s.type == "user_message" for s in drained)
    assert any(s.adrenalin for s in drained)
    assert any(s.metadata.get("chat_id") == 123 for s in drained)


async def test_sensory_advances_offset_so_old_msgs_not_replayed():
    client = FakeClient(updates_batches=[
        [_msg(5, 1, "first")],
        [_msg(6, 1, "second")],
    ])
    buf = StimulusBuffer()
    sensory = TelegramSensory(client=client)
    await sensory.start(buf.push)
    await asyncio.sleep(0.15)
    await sensory.stop()

    # Offset 0 → 6 → 7 (first call uses 0, second uses 6, third uses 7)
    assert client.update_calls[0] == 0
    assert 6 in client.update_calls or 7 in client.update_calls


async def test_sensory_allowed_chat_filter():
    client = FakeClient(updates_batches=[
        [_msg(1, 999, "from stranger"),
         _msg(2, 42, "from friend")],
    ])
    buf = StimulusBuffer()
    sensory = TelegramSensory(client=client, allowed_chat_ids={42})
    await sensory.start(buf.push)
    await asyncio.sleep(0.1)
    await sensory.stop()

    drained = buf.drain()
    contents = [s.content for s in drained]
    assert "from friend" in contents
    assert "from stranger" not in contents


async def test_sensory_handles_get_updates_exception_and_continues():
    client = FakeClient(updates_batches=[
        RuntimeError("net flap"),
        [_msg(1, 1, "after recovery")],
    ])
    buf = StimulusBuffer()
    sensory = TelegramSensory(client=client, error_backoff=0.01)
    await sensory.start(buf.push)
    await asyncio.sleep(0.15)
    await sensory.stop()
    drained = buf.drain()
    assert any(s.content == "after recovery" for s in drained)


async def test_sensory_skips_messages_without_text():
    client = FakeClient(updates_batches=[
        [{"update_id": 1, "message": {"chat": {"id": 1}}},  # no text
         _msg(2, 1, "real")],
    ])
    buf = StimulusBuffer()
    sensory = TelegramSensory(client=client)
    await sensory.start(buf.push)
    await asyncio.sleep(0.1)
    await sensory.stop()
    drained = buf.drain()
    assert [s.content for s in drained] == ["real"]


def test_sensory_metadata():
    sensory = TelegramSensory(client=FakeClient())
    assert sensory.name == "telegram"
    assert sensory.default_adrenalin is True


# ---------------- TelegramReplyTentacle ----------------

def test_tentacle_metadata():
    t = TelegramReplyTentacle(client=FakeClient())
    assert t.name == "telegram_reply"


async def test_tentacle_sends_via_client_with_explicit_chat_id():
    client = FakeClient()
    t = TelegramReplyTentacle(client=client)
    stim = await t.execute("hi friend",
                              {"chat_id": 42, "text": "hi friend"})
    assert client.sent == [(42, "hi friend")]
    assert "sent" in stim.content.lower() or "已发送" in stim.content


async def test_tentacle_uses_default_chat_id_when_param_missing():
    client = FakeClient()
    t = TelegramReplyTentacle(client=client, default_chat_id=99)
    await t.execute("hi", {})
    assert client.sent[0][0] == 99


async def test_tentacle_intent_used_as_text_when_no_text_param():
    client = FakeClient()
    t = TelegramReplyTentacle(client=client, default_chat_id=1)
    await t.execute("free-form intent text", {})
    assert client.sent[0][1] == "free-form intent text"


async def test_tentacle_no_chat_id_returns_error_stimulus():
    client = FakeClient()
    t = TelegramReplyTentacle(client=client)  # no default chat
    stim = await t.execute("hi", {})
    assert client.sent == []
    assert "no chat" in stim.content.lower() or "缺少" in stim.content


async def test_tentacle_send_failure_returns_adrenalin_error():
    client = FakeClient(send_raises=RuntimeError("network down"))
    t = TelegramReplyTentacle(client=client, default_chat_id=1)
    stim = await t.execute("hi", {})
    assert stim.adrenalin is True
    assert "network down" in stim.content
