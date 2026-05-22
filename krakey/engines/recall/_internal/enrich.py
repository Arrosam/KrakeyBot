"""Semantic-association enricher — recall engine private.

Sends a stimulus text to a bound ``ChatLike`` client and asks it to
derive multiple self-contained search phrases (key persons, dated
events with absolute dates, places, relations). Each phrase can then
become its own recall query through the existing vec_search → rerank
→ weighted-merge loop, widening coverage beyond the raw stimulus text.

Entirely optional and default-off:
- If ``SemanticAssociationEnricher`` is not constructed (``enricher=None``
  on ``IncrementalRecall``), the path is completely inert — not even
  instantiated, let alone called.
- If the LLM call raises or returns unusable output, ``enrich`` returns
  ``None`` and the caller falls back to the original stimulus text.
- ``NEVER`` calls ``datetime.now()`` internally; ``now`` is always
  supplied by the caller so test doubles remain deterministic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from krakey.interfaces.duck import ChatLike


_SYSTEM_PROMPT = """\
You extract searchable key information from a message for memory retrieval.

Output ONE self-contained search phrase per line. Surface:
- Key persons mentioned
- Key events (resolve EVERY relative date — "tomorrow", "next week", weekday names — to an absolute ISO YYYY-MM-DD computed against the provided CURRENT DATETIME)
- Key places
- Key relations between entities
- Situational/role context (when the message implies Self is being asked to act, respond, or do something, emit a phrase describing the request pattern, e.g. "Self asked to perform a task by Sam")
- Procedural/task context (when the message contains an actionable task, emit a phrase naming the task type and domain, e.g. "flight booking procedure", "travel planning experience")

Rules:
- No numbering, no bullet points, no markdown, no JSON
- No commentary, no preamble, no labels — phrases only, one per line
- If nothing meaningful is extractable, output the original message verbatim as the single line

Example:
CURRENT DATETIME: 2026-05-17T00:00:00
MESSAGE: Sam: "Can you book me a flight to Auckland next Friday for a client meeting?"
Output:
Sam (key person)
Flight to Auckland on 2026-05-22 (key event)
Auckland (key place)
Sam travelling to Auckland for a client meeting
Self asked to perform a task by Sam
Flight booking procedure\
"""


class SemanticAssociationEnricher:
    """Derives multiple recall-query phrases from a stimulus text via LLM.

    Constructed with a single ``ChatLike`` client (already resolved by
    the engine from the LLM factory). Stateless beyond that reference —
    safe to share across beats if needed, but the engine currently
    creates one per session for a fresh client reference each beat.
    """

    def __init__(self, client: "ChatLike") -> None:
        self._client = client

    async def enrich(self, text: str | None, *,
                     now: "datetime") -> list[str] | None:
        """Return a list of search phrases derived from *text*, or ``None``.

        ``None`` signals "no usable enrichment" — the caller should fall
        back to the raw stimulus text. Returning ``None`` is NOT an error.

        Args:
            text: Raw stimulus content. ``None`` or blank → returns ``None``
                  immediately without calling the LLM.
            now:  Current datetime supplied by the caller. Used to resolve
                  relative dates in the LLM prompt. Never derived
                  internally (keeps callers testable).
        """
        if text is None or not text.strip():
            return None

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"CURRENT DATETIME: {now.isoformat()}\n\n"
                    f"MESSAGE:\n{text}"
                ),
            },
        ]
        try:
            out = await self._client.chat(messages)
        except Exception:  # noqa: BLE001
            return None

        if not isinstance(out, str) or not out.strip():
            return None

        kept: list[str] = []
        for line in out.splitlines():
            s = line.strip()
            if s == "":
                continue
            if s.startswith("```"):
                continue
            if s in {"{", "}", "[", "]"}:
                continue
            kept.append(s)

        return kept if kept else None
