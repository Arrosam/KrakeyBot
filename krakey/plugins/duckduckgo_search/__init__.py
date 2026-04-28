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
        self._max_results = max_results

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return ("Web search via DuckDuckGo. Use when you want fresh "
                "external info you don't already have in GM/KB. "
                "Returns top N results (title + url + snippet).")

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "query": "search keywords (defaults to the natural-language intent)",
            "max_results": "max number of results (default 5)",
        }


    async def execute(self, intent: str,
                        params: dict[str, Any]) -> Stimulus:
        query = (params.get("query") or intent or "").strip()
        max_results = int(params.get("max_results") or self._max_results)

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
