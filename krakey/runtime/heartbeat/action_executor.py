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
    """One ``<tool_call>`` block whose parsing surfaced a problem.

    Two flavors:
      * ``salvaged=False`` (default) — call was lost; orchestrator
        reports it as a failed dispatch.
      * ``salvaged=True``  — JSON had trailing junk but the parser
        recovered the object; the call DID dispatch and a
        ``ToolCall`` was emitted alongside this ``ParseFailure``.
        Surfaced so Self gets format-correction feedback even on
        the salvage path.
    """
    payload: str       # raw text inside <tool_call>...</tool_call>
    error: str         # human-readable reason (JSONDecodeError msg, shape error)
    block_index: int   # 0-based position among all <tool_call> blocks
    salvaged: bool = False


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
        # Salvage path: the parser can return BOTH a successful
        # ToolCall AND a ParseFailure (the call was recovered from
        # trailing junk; the failure carries the diagnostic so
        # Self still gets corrective feedback). So append both,
        # not one-or-the-other.
        if call is not None:
            calls.append(call)
        if failure is not None:
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

    Salvage behavior on JSON "Extra data": some open-source models
    append trailing junk after the JSON object (e.g. stray
    ``</arg_value>`` from XML-style training, or invisible
    characters like zero-width space / BOM). The decoder's
    ``e.pos`` points at the first byte past the JSON, so we
    truncate to ``payload[:e.pos]`` and retry — but we ALSO emit
    a ``ParseFailure`` so Self still gets the corrective stimulus
    (with the trailing-junk visible via ``repr()``) on the next
    beat. The call dispatches; the format drift gets surfaced;
    Self sees both action effect AND the diagnostic.
    """
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError as e:
        # Salvage path for "Extra data" only — every other decode
        # failure (Expecting value, Unterminated string, etc.) has
        # no safe truncation point.
        salvaged: Any | None = None
        if "Extra data" in str(e) and e.pos > 0:
            try:
                salvaged = json.loads(payload[:e.pos])
            except json.JSONDecodeError:
                salvaged = None

        if salvaged is None:
            msg = f"JSON decode error: {e}"
            _log.warning(
                "tool_call: skipping unparseable payload %r (%s)",
                payload, e,
            )
            return None, ParseFailure(
                payload=payload, error=msg, block_index=block_index,
            )

        # Salvaged: continue with the truncated obj, but also build
        # a ParseFailure describing the trailing junk so Self gets
        # corrective feedback on the next beat. The payload field
        # uses repr() so invisible characters (zero-width space,
        # BOM, control chars) appear as escape sequences instead
        # of rendering as nothing.
        trailing = payload[e.pos:]
        salvage_failure = ParseFailure(
            payload=repr(payload),
            error=(
                f"trailing data after JSON object — {e}. "
                f"Trailing bytes: {trailing!r}. "
                "The call was salvaged this time, but emit ONLY "
                "the JSON object inside <tool_call>...</tool_call> "
                "with no characters after the closing }."
            ),
            block_index=block_index,
            salvaged=True,
        )
        obj = salvaged
        _log.warning(
            "tool_call: salvaged payload by truncating trailing "
            "junk %r (%s)", trailing, e,
        )
        # Fall through into shape validation below; emit the
        # salvage failure alongside the successful ToolCall via the
        # explicit return at the bottom of the success path.
    else:
        salvage_failure = None  # type: ignore[assignment]
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
    # ``salvage_failure`` is non-None only when the JSON parser
    # truncated trailing junk to recover. Return it alongside the
    # successful ToolCall so the orchestrator pushes a corrective
    # stimulus AND the dispatch goes through. This is the key
    # difference from "failed parse" — Self gets BOTH effects so
    # it can iterate on its format without losing forward progress.
    return ToolCall(
        tool=name, intent=intent, params=arguments,
        adrenalin=adrenalin,
    ), salvage_failure


_INTENT_VALUE_PREVIEW_CHARS = 40
"""Per-arg value preview cap inside ``_synth_intent`` output."""

_INTENT_TOTAL_PREVIEW_CHARS = 120
"""Total cap on the synthesized intent string."""


def _synth_intent(name: str, arguments: dict[str, Any]) -> str:
    """Compact one-line label for the dispatch event display.

    Includes short value previews for primitive / list args so
    the dashboard's tool-usage line says ``cli_exec(env='local',
    cmd=['python', '--version'])`` instead of the previously
    keys-only ``cli_exec(env, cmd, cwd, timeout_s, stdin)`` —
    the keys-only form was useless for telling apart back-to-
    back invocations of the same tool.

    Large / non-primitive values render as ``key=...`` to keep
    the line compact (full args ride along separately on
    ``DispatchEvent.params`` for callers that need the full
    structure).
    """
    if not arguments:
        return name
    parts: list[str] = []
    for k, v in arguments.items():
        parts.append(f"{k}={_format_arg_value(v)}")
    inner = ", ".join(parts)
    if len(inner) > _INTENT_TOTAL_PREVIEW_CHARS:
        inner = inner[:_INTENT_TOTAL_PREVIEW_CHARS - 3] + "..."
    return f"{name}({inner})"


def _format_arg_value(v: Any) -> str:
    """Render a single arg value compactly. Strings + numbers +
    bools print verbatim (truncated). Lists of primitives print
    as ``[a, b, c]`` (truncated). Everything else collapses to
    ``...`` so a giant blob doesn't blow up the line."""
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return _truncate_repr(v)
    if isinstance(v, list) and all(
        isinstance(x, (str, int, float, bool)) or x is None
        for x in v
    ):
        rendered = "[" + ", ".join(_format_arg_value(x) for x in v) + "]"
        if len(rendered) > _INTENT_VALUE_PREVIEW_CHARS:
            rendered = rendered[:_INTENT_VALUE_PREVIEW_CHARS - 3] + "...]"
        return rendered
    return "..."


def _truncate_repr(s: str) -> str:
    r = repr(s)
    if len(r) > _INTENT_VALUE_PREVIEW_CHARS:
        r = r[:_INTENT_VALUE_PREVIEW_CHARS - 4] + "...'"
    return r
