"""CLI override commands (DevSpec §19).

The CLI sensory pushes everything as a `user_message` stimulus. The runtime's
override phase scans drained stimuli for a leading `/<cmd>`; when matched the
stimulus is consumed (Self never sees it) and the action runs out-of-band.

Supported commands:
  /status        — print runtime + GM stats
  /memory_stats  — print full GM dump
  /sleep         — trigger 7-phase Sleep immediately
  /wake          — no-op for now (Sleep is synchronous)
  /kill          — graceful shutdown after current heartbeat

This module owns parsing + dispatch only. The actual rendering for
``/status`` and ``/memory_stats`` lives in
``src.runtime.overrides.state_formatters``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from src.runtime.overrides.state_formatters import (
    format_memory_stats, format_status,
)

if TYPE_CHECKING:
    from src.main import Runtime


KNOWN_COMMANDS: set[str] = {"status", "memory_stats", "sleep", "wake", "kill"}


class OverrideAction(Enum):
    NONE = "none"      # informational only — caller continues normal flow
    SLEEP = "sleep"    # caller should run _perform_sleep
    KILL = "kill"      # caller should set _stop = True


@dataclass
class OverrideResult:
    action: OverrideAction
    output: str       # human-readable summary printed via logger


def parse_override(content: str | None) -> str | None:
    """Return the command name (without slash) iff the content is a known
    override; otherwise None."""
    if not content:
        return None
    text = content.strip()
    if not text.startswith("/"):
        return None
    cmd = text[1:].split(maxsplit=1)[0].lower()
    return cmd if cmd in KNOWN_COMMANDS else None


async def handle_override(cmd: str, runtime: "Runtime") -> OverrideResult:
    if cmd == "status":
        return OverrideResult(OverrideAction.NONE, await format_status(runtime))
    if cmd == "memory_stats":
        return OverrideResult(OverrideAction.NONE,
                              await format_memory_stats(runtime))
    if cmd == "sleep":
        return OverrideResult(OverrideAction.SLEEP,
                              "manual sleep request received")
    if cmd == "wake":
        return OverrideResult(
            OverrideAction.NONE,
            "wake noop — current Sleep impl is synchronous, runtime is "
            "already awake when this fires.",
        )
    if cmd == "kill":
        return OverrideResult(OverrideAction.KILL,
                              "manual shutdown requested")
    return OverrideResult(OverrideAction.NONE,
                          f"unknown command: /{cmd}")
