"""Self output parser (DevSpec §3.3).

Regex extracts [THINKING] / [DECISION] / [NOTE] / [HIBERNATE].
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
    hibernate_seconds: int | None = None


_TAGS = ("THINKING", "DECISION", "NOTE", "HIBERNATE")
_TAG_PATTERN = re.compile(r"\[(" + "|".join(_TAGS) + r")\]", re.IGNORECASE)
_INT_PATTERN = re.compile(r"-?\d+")


def parse_self_output(raw: str) -> ParsedSelfOutput:
    sections = _split_sections(raw)

    if not sections:
        # Fallback: no markers → treat whole body as thinking+decision
        body = raw.strip()
        return ParsedSelfOutput(thinking=body, decision=body)

    return ParsedSelfOutput(
        thinking=sections.get("THINKING", "").strip(),
        decision=sections.get("DECISION", "").strip(),
        note=sections.get("NOTE", "").strip(),
        hibernate_seconds=_parse_int(sections.get("HIBERNATE", "")),
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
