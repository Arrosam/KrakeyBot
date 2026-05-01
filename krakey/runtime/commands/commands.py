"""CLI / chat slash-commands (DevSpec §19).

Any channel that surfaces user text (CLI, web chat, telegram, ...)
pushes it as a ``user_message`` stimulus. The runtime's command phase
scans drained stimuli for a leading ``/<cmd>``; when matched the
stimulus is consumed (Self never sees it) and the action runs
out-of-band — the command bypasses Self's decision loop entirely.

Supported commands:
  /status        — print runtime + GM stats
  /memory_stats  — print full GM dump
  /sleep         — trigger 7-phase Sleep immediately
  /kill          — graceful shutdown after current heartbeat
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from krakey.main import Runtime


KNOWN_COMMANDS: set[str] = {"status", "memory_stats", "sleep", "kill"}


class CommandAction(Enum):
    NONE = "none"      # informational only — caller continues normal flow
    SLEEP = "sleep"    # caller should run _perform_sleep
    KILL = "kill"      # caller should set _stop = True


@dataclass
class CommandResult:
    action: CommandAction
    output: str       # human-readable summary printed via logger


def parse_command(content: str | None) -> str | None:
    """Return the command name (without slash) iff the content is a
    recognised slash-command; otherwise None."""
    if not content:
        return None
    text = content.strip()
    if not text.startswith("/"):
        return None
    cmd = text[1:].split(maxsplit=1)[0].lower()
    return cmd if cmd in KNOWN_COMMANDS else None


async def handle_command(cmd: str, runtime: "Runtime") -> CommandResult:
    """Interpret ``cmd`` (already validated by ``parse_command`` to be
    in ``KNOWN_COMMANDS``) into a ``CommandResult``. Caller acts on
    ``result.action``; SLEEP/KILL trigger their side-effects in the
    heartbeat orchestrator's command phase, NONE just logs the output."""
    if cmd == "status":
        return CommandResult(CommandAction.NONE, await _format_status(runtime))
    if cmd == "memory_stats":
        return CommandResult(CommandAction.NONE,
                                await _format_memory_stats(runtime))
    if cmd == "sleep":
        return CommandResult(CommandAction.SLEEP,
                                "manual sleep request received")
    # cmd == "kill" — only remaining KNOWN_COMMANDS entry.
    return CommandResult(CommandAction.KILL,
                            "manual shutdown requested")


# ---------------- formatters ----------------


async def _format_status(runtime: "Runtime") -> str:
    nodes = await runtime.gm.count_nodes()
    edges = await runtime.gm.count_edges()
    pct = int(nodes / runtime.config.fatigue.gm_node_soft_limit * 100) \
        if runtime.config.fatigue.gm_node_soft_limit else 0
    name = runtime.self_model.get("identity", {}).get("name", "(unnamed)")
    return (
        f"name={name} "
        f"heartbeats={runtime.heartbeat_count} "
        f"gm_nodes={nodes} gm_edges={edges} fatigue={pct}% "
        f"sleep_cycles={runtime._sleep_cycles} "
        f"bootstrap_complete={not runtime.is_bootstrap}"
    )


async def _format_memory_stats(runtime: "Runtime") -> str:
    nodes = await runtime.gm.count_nodes()
    edges = await runtime.gm.count_edges()
    cat_counts = await runtime.gm.counts_by_category()
    src_counts = await runtime.gm.counts_by_source()
    kbs = await runtime.kb_registry.list_kbs()

    by_cat = ", ".join(f"{k}={v}" for k, v in cat_counts.items()) or "(none)"
    by_src = ", ".join(f"{k}={v}" for k, v in src_counts.items()) or "(none)"
    kb_line = f"{len(kbs)} KB(s)" + (
        ": " + ", ".join(f"{k['kb_id']}({k['entry_count']})" for k in kbs)
        if kbs else ""
    )
    return (f"gm: {nodes} nodes, {edges} edges  |  by_cat: {by_cat}  |  "
            f"by_src: {by_src}  |  {kb_line}")
