"""HTTP client for a SearXNG ``/search?format=json`` endpoint.

Async wrapper over ``aiohttp`` matching the ``SearxngBackend``
Protocol declared in ``tool.py``. Production factory wires this;
tests substitute a dict-returning fake so the test path never imports
``aiohttp``.

Pagination
----------
SearXNG returns ~10 results per page by default. ``search()`` walks
``pageno=1, 2, ...`` over a single ``aiohttp.ClientSession``
(connection reuse) until any of:

  * the requested ``max_results`` is reached,
  * SearXNG returns an empty page (out of results), or
  * a page yields no results we haven't already seen by URL
    (engines occasionally repeat top hits across pages — without
    URL-based dedup the loop would burn page budget on duplicates),
  * ``MAX_PAGES`` is hit (safety cap; prevents a runaway loop if
    SearXNG ever returns infinite stable pages).

Why POST: SearXNG accepts both GET and POST on ``/search``. POST
keeps long queries off the URL line (some upstream proxies log full
GET URLs and we'd rather not leak Self's queries that way). It also
sidesteps hardened SearXNG deployments that disable GET-format=json
queries to defeat scraping bots — ``limiter.toml`` typically
allows POST through.

Required SearXNG settings.yml: ``search.formats`` must include
``json`` for this endpoint to return parseable output. The official
Docker image ships JSON disabled by default; if the plugin gets HTTP
403 the operator should mount a custom settings file with JSON
enabled.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

import aiohttp


# Internal pagination cap — at default ~10 results / page this lets
# Self reach the tool-layer cap (50) twice over before bailing. Set
# higher than strictly needed because SearXNG sometimes returns < 10
# per page when an engine times out.
MAX_PAGES = 10

# SearXNG filter keys we forward verbatim to the HTTP body. Empty /
# None values are stripped so SearXNG falls back to its own defaults
# instead of being told "filter to nothing".
_FILTER_KEYS: tuple[str, ...] = (
    "categories", "language", "time_range", "safesearch", "engines",
)


class HttpSearxngClient:
    """Async ``SearxngBackend`` impl targeting a SearXNG instance."""

    def __init__(
        self, instance_url: str, *, timeout_s: float = 15.0,
    ):
        self._url = instance_url.rstrip("/") + "/search"
        # Total wall-clock per request; SearXNG fans queries out to
        # many engines in parallel internally, so a single timeout
        # is enough.
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)

    async def search(
        self, query: str, *, max_results: int, **filters: Any,
    ) -> dict[str, Any]:
        body_base = build_body(query, filters)
        async with aiohttp.ClientSession(timeout=self._timeout) as s:
            async def _fetch(body: dict[str, Any]) -> list[dict[str, Any]]:
                async with s.post(self._url, data=body) as r:
                    r.raise_for_status()
                    payload = await r.json()
                return list(payload.get("results") or [])

            results = await paginate(
                body_base, max_results=max_results, fetch_page=_fetch,
            )
        return {"query": query, "results": results}


def build_body(
    query: str, filters: dict[str, Any],
) -> dict[str, Any]:
    """Pure helper — base body for every ``/search`` POST.

    ``pageno`` is added per-request by the pagination loop, NOT
    here, so the caller can mutate the page number without rebuilding
    the rest. Empty / None filter values are skipped; sending
    ``language=""`` would override the SearXNG instance default to
    "no language", which isn't what Self asked for.
    """
    body: dict[str, Any] = {"q": query, "format": "json"}
    for key in _FILTER_KEYS:
        v = filters.get(key)
        if v == "" or v is None:
            continue
        body[key] = v
    return body


async def paginate(
    body_base: dict[str, Any],
    *,
    max_results: int,
    fetch_page: Callable[[dict[str, Any]], Awaitable[list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """Walk SearXNG pages until ``max_results`` reached or exhausted.

    Pure-logic — the HTTP client supplies ``fetch_page``; tests
    supply an in-memory fake. Dedup keyed on result URL because
    SearXNG occasionally returns the same top hit on consecutive
    pages and Self should only see each result once.

    Stops on first of:
      * ``len(out) >= max_results``
      * empty page (SearXNG returns ``results: []``)
      * page contributed zero NEW URLs (all dupes — engines
        ran out of fresh hits even if SearXNG keeps paginating)
      * ``MAX_PAGES`` reached
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pageno in range(1, MAX_PAGES + 1):
        if len(out) >= max_results:
            break
        page = await fetch_page({**body_base, "pageno": pageno})
        if not page:
            break
        added = 0
        for r in page:
            url = r.get("url")
            if url and url in seen:
                continue
            if url:
                seen.add(url)
            out.append(r)
            added += 1
            if len(out) >= max_results:
                break
        if added == 0:
            break
    return out
