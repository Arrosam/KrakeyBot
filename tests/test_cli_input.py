import asyncio

import pytest

from src.runtime.stimulus_buffer import StimulusBuffer
from src.sensories.cli_input import CliInputSensory


async def test_pushes_user_message_with_adrenalin():
    lines = asyncio.Queue()
    await lines.put("hello\n")
    await lines.put("world\n")
    await lines.put(None)  # EOF

    async def reader():
        return await lines.get()

    buf = StimulusBuffer()
    sensory = CliInputSensory(default_adrenalin=True, reader=reader)
    await sensory.start(buf)

    # Let the reader loop drain both lines
    await asyncio.sleep(0.05)
    await sensory.stop()

    drained = buf.drain()
    assert [s.content for s in drained] == ["hello", "world"]
    assert all(s.type == "user_message" for s in drained)
    assert all(s.source == "sensory:cli_input" for s in drained)
    assert all(s.adrenalin for s in drained)


async def test_non_adrenalin_config():
    lines = asyncio.Queue()
    await lines.put("hey\n")
    await lines.put(None)

    async def reader():
        return await lines.get()

    buf = StimulusBuffer()
    sensory = CliInputSensory(default_adrenalin=False, reader=reader)
    await sensory.start(buf)
    await asyncio.sleep(0.05)
    await sensory.stop()

    drained = buf.drain()
    assert drained[0].adrenalin is False


async def test_name_and_default_adrenalin_property():
    sensory = CliInputSensory(default_adrenalin=True)
    assert sensory.name == "cli_input"
    assert sensory.default_adrenalin is True
