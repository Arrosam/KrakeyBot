"""Action executor — parses Self's ``[ACTION]...[/ACTION]`` block into
``TentacleCall`` objects.

This is the default tentacle-dispatch path for strong models that can
emit structured calls directly. Skips the Hypothalamus translation
LLM entirely: one Self LLM call produces a decision *and* the calls,
the executor parses, runtime dispatches. Cheaper + faster than the
Hypothalamus path.

Format (locked 2026-04-25, see docs/design/reflects-and-self-model.md):
OpenAI tool_calls flavored JSONL, one call per line, wrapped in
``[ACTION]...[/ACTION]`` sentinels::

    [ACTION]
    {"name": "web_chat_reply", "arguments": {"text": "Hi"}}
    {"name": "search", "arguments": {"query": "..."}, "adrenalin": true}
    [/ACTION]

Fields per call:
    name: str (required)            — tentacle name
    arguments: dict (optional)      — params for the tentacle; default {}
    adrenalin: bool (optional)      — urgency flag; default False

Failure modes are isolated per-line: a single malformed JSON line is
skipped (logged), the rest of the block still dispatches. We don't
collapse the whole block on one bad line because that would make the
parser brittle in a way Hypothalamus's whole-blob JSON parsing
already showed is risky.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.interfaces.reflect import TentacleCall

_log = logging.getLogger(__name__)

_ACTION_BLOCK = re.compile(
    r"\[ACTION\](.*?)\[/ACTION\]",
    re.DOTALL | re.IGNORECASE,
)


def parse_action_block(self_text: str) -> list[TentacleCall]:
    """Extract all ``[ACTION]...[/ACTION]`` JSONL blocks from
    ``self_text`` and return the parsed ``TentacleCall`` list.

    Multiple blocks are concatenated (rare but legal — Self might emit
    actions split across [DECISION] sections). Empty / missing blocks
    return ``[]``. Malformed lines are skipped with a warning.
    """
    if not self_text:
        return []
    calls: list[TentacleCall] = []
    for block_match in _ACTION_BLOCK.finditer(self_text):
        body = block_match.group(1)
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            call = _parse_one_call(line)
            if call is not None:
                calls.append(call)
    return calls


def _parse_one_call(line: str) -> TentacleCall | None:
    """Parse a single line. Returns None on any failure mode.

    The line must JSON-decode to an object with at least a ``name``
    string. Anything else is skipped with a single-line warning so a
    bad call doesn't poison adjacent good ones.
    """
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        _log.warning("action executor: skipping unparseable line %r (%s)",
                     line, e)
        return None
    if not isinstance(obj, dict):
        _log.warning("action executor: line is JSON but not an object: %r",
                     line)
        return None
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        _log.warning("action executor: line missing/empty `name`: %r", line)
        return None
    arguments = obj.get("arguments") or {}
    if not isinstance(arguments, dict):
        _log.warning(
            "action executor: arguments is not an object on call %r; "
            "treating as empty", name,
        )
        arguments = {}
    adrenalin = bool(obj.get("adrenalin", False))
    # Intent string is a human-readable label. The structured path
    # doesn't carry it natively; we synthesize from name + a short
    # arg preview so the dashboard's dispatch line is informative.
    intent = _synth_intent(name, arguments)
    return TentacleCall(
        tentacle=name, intent=intent, params=arguments,
        adrenalin=adrenalin,
    )


def _synth_intent(name: str, arguments: dict[str, Any]) -> str:
    """Compact one-line label for /dispatch event display.

    Avoids dumping the entire arguments dict — large prompts /
    file-write tentacles would render unreadably long.
    """
    if not arguments:
        return name
    keys = ", ".join(arguments.keys())
    return f"{name}({keys})"
