"""Tool-call parser — extracts ``<tool_call>...</tool_call>`` blocks
out of Self's raw response into ``TentacleCall`` objects.

This is the default tentacle-dispatch path when no decision-translator
Reflect (e.g. the hypothalamus plugin) is registered. Format chosen
for breadth of training coverage in modern open-source models —
Hermes / Qwen 2.5+ emit this format natively (their tokenizers
reserve ``<tool_call>`` / ``</tool_call>`` as special tokens), and
Llama / Mistral / DeepSeek families emit it readily with one or two
in-prompt examples because the inner ``name``+``arguments`` JSON
shape matches what they were already trained on.

Format:

    <tool_call>
    {"name": "<tentacle_name>", "arguments": {...}}
    </tool_call>

Parallel calls = repeat the tag. Each tag wraps exactly one JSON
object. Fields per call:

    name:       str (required)         — tentacle name
    arguments:  dict (optional)        — params for the tentacle; default {}
    adrenalin:  bool (optional)        — urgency flag; default False

Failure modes are isolated per-block: a single malformed payload is
skipped (logged), the rest of Self's response still dispatches.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from krakey.interfaces.reflect import TentacleCall

_log = logging.getLogger(__name__)

# Match <tool_call>...</tool_call> non-greedily; tolerant of leading/
# trailing whitespace and newlines inside the block. Case-insensitive
# in case a model emits TOOL_CALL or similar.
_TOOL_CALL_BLOCK = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)


def parse_tool_calls(self_text: str) -> list[TentacleCall]:
    """Extract every ``<tool_call>...</tool_call>`` block from
    ``self_text`` and return the parsed ``TentacleCall`` list.

    Empty / missing blocks return ``[]``. Malformed payloads are
    skipped with a warning so one bad tag doesn't poison the others.
    """
    if not self_text:
        return []
    calls: list[TentacleCall] = []
    for block_match in _TOOL_CALL_BLOCK.finditer(self_text):
        payload = block_match.group(1).strip()
        if not payload:
            continue
        call = _parse_one_call(payload)
        if call is not None:
            calls.append(call)
    return calls


# Back-compat alias — old name still imported in a couple of places.
parse_action_block = parse_tool_calls


def _parse_one_call(payload: str) -> TentacleCall | None:
    """Parse the JSON payload of one ``<tool_call>`` block.

    The payload must JSON-decode to an object with a non-empty
    ``name`` string. Anything else is skipped with a warning.
    """
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError as e:
        _log.warning(
            "tool_call: skipping unparseable payload %r (%s)", payload, e,
        )
        return None
    if not isinstance(obj, dict):
        _log.warning(
            "tool_call: payload is JSON but not an object: %r", payload,
        )
        return None
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        _log.warning("tool_call: payload missing/empty `name`: %r", payload)
        return None
    arguments = obj.get("arguments") or {}
    if not isinstance(arguments, dict):
        _log.warning(
            "tool_call: arguments is not an object on call %r; "
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
    """Compact one-line label for the dispatch event display.

    Avoids dumping the entire arguments dict — large prompts /
    file-write tentacles would render unreadably long.
    """
    if not arguments:
        return name
    keys = ", ".join(arguments.keys())
    return f"{name}({keys})"
