"""Self output parser (DevSpec §3.3).

Regex extracts [THINKING] / [DECISION] / [NOTE] / [IDLE].
If no markers present, the whole text fills both thinking and decision
so Hypothalamus still receives something to translate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParsedSelfOutput:
    thinking: str = ""
    decision: str = ""
    note: str = ""
    idle_seconds: int | None = None
    # Full unparsed response. Kept so the tool-call parser
    # (default-off path) can locate ``<tool_call>...</tool_call>``
    # blocks wherever Self placed them in the response, not just
    # inside one of the four known tag sections.
    raw: str = ""


_TAGS = ("THINKING", "DECISION", "NOTE", "IDLE")
_TAG_PATTERN = re.compile(r"\[(" + "|".join(_TAGS) + r")\]", re.IGNORECASE)
_INT_PATTERN = re.compile(r"-?\d+")


def parse_self_output(raw: str) -> ParsedSelfOutput:
    sections = _split_sections(raw)

    if not sections:
        # Fallback: no markers → treat whole body as thinking+decision
        body = raw.strip()
        return ParsedSelfOutput(thinking=body, decision=body, raw=raw)

    return ParsedSelfOutput(
        thinking=sections.get("THINKING", "").strip(),
        decision=sections.get("DECISION", "").strip(),
        note=sections.get("NOTE", "").strip(),
        idle_seconds=_parse_int(sections.get("IDLE", "")),
        raw=raw,
    )


def _split_sections(raw: str) -> dict[str, str]:
    matches = list(_TAG_PATTERN.finditer(raw))
    if not matches:
        return {}
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        tag = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        sections[tag] = raw[start:end]
    return sections


def _parse_int(text: str) -> int | None:
    m = _INT_PATTERN.search(text)
    return int(m.group(0)) if m else None
