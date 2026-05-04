"""Tool-call parser — extracts ``<tool_call>...</tool_call>`` blocks
out of Self's raw response into ``ToolCall`` objects.

This is the default tool-dispatch path when no decision-translator
Modifier (e.g. the hypothalamus plugin) is registered. Format chosen
for breadth of training coverage in modern open-source models —
Hermes / Qwen 2.5+ emit this format natively (their tokenizers
reserve ``<tool_call>`` / ``</tool_call>`` as special tokens), and
Llama / Mistral / DeepSeek families emit it readily with one or two
in-prompt examples because the inner ``name``+``arguments`` JSON
shape matches what they were already trained on.

Format:

    <tool_call>
    {"name": "<tool_name>", "arguments": {...}}
    </tool_call>

Parallel calls = repeat the tag. Each tag wraps exactly one JSON
object. Fields per call:

    name:       str (required)         — tool name
    arguments:  dict (optional)        — params for the tool; default {}
    adrenalin:  bool (optional)        — urgency flag; default False

Failure modes are isolated per-block: a single malformed payload is
skipped, the rest of Self's response still dispatches. The richer
entry point ``parse_tool_calls_with_failures`` ALSO returns a list
of ``ParseFailure`` describing each skipped block — the orchestrator
uses this to push a corrective ``system_event`` stimulus back to
Self so format drift is visible in-context on the next beat,
rather than silently absorbed (the trade-off being a wasted beat
when Self misformats vs. Self never knowing it misformatted).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from krakey.interfaces.modifier import ToolCall

_log = logging.getLogger(__name__)

# Match <tool_call>...</tool_call> non-greedily; tolerant of leading/
# trailing whitespace and newlines inside the block. Case-insensitive
# in case a model emits TOOL_CALL or similar.
_TOOL_CALL_BLOCK = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class ParseFailure:
    """One ``<tool_call>`` block that couldn't be turned into a
    ``ToolCall``. Carried back to the orchestrator so it can push
    a corrective stimulus to Self with the offending payload + the
    parser's diagnosis."""
    payload: str       # raw text inside <tool_call>...</tool_call>
    error: str         # human-readable reason (JSONDecodeError msg, shape error)
    block_index: int   # 0-based position among all <tool_call> blocks


def parse_tool_calls_with_failures(
    self_text: str,
) -> tuple[list[ToolCall], list[ParseFailure]]:
    """Like ``parse_tool_calls`` but ALSO returns the list of
    ``ParseFailure`` for blocks that couldn't be parsed. The
    success list is independent of failures — partial parse must
    not block the calls that did parse cleanly.
    """
    if not self_text:
        return [], []
    calls: list[ToolCall] = []
    failures: list[ParseFailure] = []
    for idx, block_match in enumerate(_TOOL_CALL_BLOCK.finditer(self_text)):
        payload = block_match.group(1).strip()
        if not payload:
            # Empty <tool_call></tool_call> isn't a parse failure —
            # just a noise tag with nothing to dispatch. Don't push
            # a corrective stimulus for it.
            continue
        call, failure = _parse_one_call(payload, block_index=idx)
        if call is not None:
            calls.append(call)
        elif failure is not None:
            failures.append(failure)
    return calls, failures


def parse_tool_calls(self_text: str) -> list[ToolCall]:
    """Extract every ``<tool_call>...</tool_call>`` block from
    ``self_text`` and return the parsed ``ToolCall`` list.

    Back-compat shim over ``parse_tool_calls_with_failures`` that
    drops the failure list. Existing callers (tests, the legacy
    ``parse_action_block`` alias) that only want the call list keep
    working unchanged.
    """
    calls, _ = parse_tool_calls_with_failures(self_text)
    return calls


# Back-compat alias — old name still imported in a couple of places.
parse_action_block = parse_tool_calls


def _parse_one_call(
    payload: str, *, block_index: int = 0,
) -> tuple[ToolCall | None, ParseFailure | None]:
    """Parse the JSON payload of one ``<tool_call>`` block.

    Returns ``(ToolCall, None)`` on success; ``(None, ParseFailure)``
    on any failure mode (JSON decode, wrong shape, missing name,
    bad argument type). Logs at warning level so terminal output
    still has a breadcrumb when the orchestrator's stimulus path
    isn't wired in (e.g. unit tests calling the parser directly).
    """
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError as e:
        msg = f"JSON decode error: {e}"
        _log.warning(
            "tool_call: skipping unparseable payload %r (%s)", payload, e,
        )
        return None, ParseFailure(
            payload=payload, error=msg, block_index=block_index,
        )
    if not isinstance(obj, dict):
        msg = (
            f"payload is valid JSON but not an object "
            f"(got {type(obj).__name__})"
        )
        _log.warning(
            "tool_call: payload is JSON but not an object: %r", payload,
        )
        return None, ParseFailure(
            payload=payload, error=msg, block_index=block_index,
        )
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        msg = "missing or empty `name` field"
        _log.warning("tool_call: payload missing/empty `name`: %r", payload)
        return None, ParseFailure(
            payload=payload, error=msg, block_index=block_index,
        )
    arguments = obj.get("arguments") or {}
    if not isinstance(arguments, dict):
        # Coerce-to-empty path: don't fail the whole call, just log.
        # Self gets the dispatch but with no args; a separate
        # parse_failure stimulus would imply the call was lost.
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
    return ToolCall(
        tool=name, intent=intent, params=arguments,
        adrenalin=adrenalin,
    ), None


def _synth_intent(name: str, arguments: dict[str, Any]) -> str:
    """Compact one-line label for the dispatch event display.

    Avoids dumping the entire arguments dict — large prompts /
    file-write tools would render unreadably long.
    """
    if not arguments:
        return name
    keys = ", ".join(arguments.keys())
    return f"{name}({keys})"
