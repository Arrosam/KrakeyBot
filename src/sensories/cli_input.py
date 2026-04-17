"""CLI Sensory (DevSpec §5.4).

Reads lines from stdin (or injected reader), pushes each as a
Stimulus(type="user_message") to the buffer. Default adrenalin
is config-driven.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from typing import Awaitable, Callable

from src.interfaces.sensory import Sensory
from src.models.stimulus import Stimulus
from src.runtime.stimulus_buffer import StimulusBuffer


LineReader = Callable[[], Awaitable[str | None]]


async def _default_stdin_reader() -> str | None:
    loop = asyncio.get_running_loop()
    line = await loop.run_in_executor(None, sys.stdin.readline)
    if line == "":
        return None
    return line


class CliInputSensory(Sensory):
    def __init__(self, default_adrenalin: bool = True,
                 reader: LineReader | None = None):
        self._default_adrenalin = default_adrenalin
        self._reader = reader or _default_stdin_reader
        self._task: asyncio.Task | None = None
        self._stopped = False

    @property
    def name(self) -> str:
        return "cli_input"

    @property
    def default_adrenalin(self) -> bool:
        return self._default_adrenalin

    async def start(self, buffer: StimulusBuffer) -> None:
        if self._task and not self._task.done():
            return
        self._stopped = False
        self._task = asyncio.create_task(self._read_loop(buffer))

    async def stop(self) -> None:
        self._stopped = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _read_loop(self, buffer: StimulusBuffer) -> None:
        while not self._stopped:
            line = await self._reader()
            if line is None:
                break
            if self._stopped:
                break
            text = line.rstrip("\r\n")
            if text:
                await buffer.push(Stimulus(
                    type="user_message",
                    source=f"sensory:{self.name}",
                    content=text,
                    timestamp=datetime.now(),
                    adrenalin=self._default_adrenalin,
                ))
            await asyncio.sleep(0)  # yield so cancellation can take effect
