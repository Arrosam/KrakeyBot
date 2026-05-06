"""Built-in ``duckduckgo_search`` plugin — web search via the ddgs library.

Krakey's "look out the window" — fetches results and returns them as a
tool_feedback stimulus. Self decides whether/how to relay them
to the user.

Tool name is the abstract verb ``search`` so Self can address it
without caring about the backend; the plugin folder name pins the
implementation (DuckDuckGo via ``ddgs``).
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Protocol

from krakey.interfaces.tool import Tool
from krakey.models.stimulus import Stimulus


_BODY_CHARS = 240
# Hard cap on results-per-call so a stray ``max_results: 9999`` from
# Self can't blow up the next prompt. ~25 results × 240 char bodies
# ≈ 6 KB content, comfortable for the heartbeat. Self can issue
# multiple searches if she actually needs more breadth.
_MAX_RESULTS_CAP = 25


class SearchBackend(Protocol):
    async def search(self, query: str,
                       max_results: int) -> list[dict[str, str]]: ...


class DDGSBackend:
    """Default backend wrapping the synchronous `ddgs` library in to_thread."""

    async def search(self, query: str,
                       max_results: int) -> list[dict[str, str]]:
        from ddgs import DDGS

        def _run() -> list[dict[str, str]]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        return await asyncio.to_thread(_run)


class SearchTool(Tool):
    def __init__(self, backend: SearchBackend, *, max_results: int = 5):
        self._backend = backend
        # Default-only floor — used when Self doesn't pass
        # ``max_results``. She can override per-call up to
        # ``_MAX_RESULTS_CAP``.
        self._max_results = max_results

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return (
            "Web search via DuckDuckGo. Use when you want fresh "
            "external info you don't already have in GM/KB — "
            "current events, docs lookup, fact-checking a claim, "
            "finding a URL to hand to ``browser_exec``. "
            "Returns ranked results (title + url + snippet) as a "
            "tool_feedback stimulus; pick which (if any) to act on. "
            "You control breadth via ``max_results``: small (3–5) "
            "for a quick check, larger (10–25) when you need to "
            "compare sources or the first hit might be wrong. "
            "If you don't pass it, the per-plugin config default "
            "is used. Cap is 25 per call — issue multiple queries "
            "with refined keywords if you need more."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search keywords. Defaults to the "
                        "natural-language intent if omitted."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": _MAX_RESULTS_CAP,
                    "description": (
                        "How many results to return. Scale up for "
                        "thoroughness, down for a quick sanity "
                        "check. Default comes from the plugin's "
                        "config (typically 5). Hard cap "
                        f"{_MAX_RESULTS_CAP}."
                    ),
                },
            },
            "additionalProperties": False,
        }


    async def execute(self, intent: str,
                        params: dict[str, Any]) -> Stimulus:
        query = (params.get("query") or intent or "").strip()

        # max_results validation: Self should be able to set this
        # per-call, but a bad type / out-of-range value shouldn't
        # crash the dispatcher — return a clear error stim instead.
        raw = params.get("max_results")
        if raw is None:
            max_results = self._max_results
        else:
            try:
                max_results = int(raw)
            except (TypeError, ValueError):
                return self._stim(
                    f"Search failed: max_results must be an integer, "
                    f"got {raw!r}.",
                )
            if max_results < 1:
                return self._stim(
                    "Search failed: max_results must be >= 1.",
                )
            if max_results > _MAX_RESULTS_CAP:
                # Soft-cap: clamp + tell Self what happened so she
                # learns the ceiling without us aborting her call.
                max_results = _MAX_RESULTS_CAP

        try:
            results = await self._backend.search(query, max_results)
        except Exception as e:  # noqa: BLE001
            return self._stim(f"Search failed: {e}")

        if not results:
            return self._stim(
                f"No results for query: {query!r}.",
            )

        return self._stim(_format(results, query))

    def _stim(self, content: str) -> Stimulus:
        return Stimulus(
            type="tool_feedback",
            source=f"tool:{self.name}",
            content=content,
            timestamp=datetime.now(),
            adrenalin=False,
        )


def _format(results: list[dict[str, str]], query: str) -> str:
    lines = [f"Search results for {query!r} — {len(results)}:"]
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "(untitled)").strip()
        href = (r.get("href") or "").strip()
        body = (r.get("body") or "").strip().replace("\n", " ")
        if len(body) > _BODY_CHARS:
            body = body[: _BODY_CHARS - 1] + "…"
        lines.append(f"{i}. [{title}] {href}")
        if body:
            lines.append(f"    {body}")
    return "\n".join(lines)


def build_tool(ctx) -> Tool:
    """Unified-format factory (Phase 2). Pulls config from
    ctx.config; no shared services needed."""
    return SearchTool(
        backend=DDGSBackend(),
        max_results=int(ctx.config.get("max_results", 5)),
    )
