"""Unit tests for the pure pieces of ``HttpSearxngClient``.

Run from repo root:

    pytest krakey/plugins/searxng_search

The two pieces under test (``build_body`` + ``paginate``) carry
all the non-trivial logic; the HTTP plumbing in ``search()`` is a
two-line ``async with`` over ``aiohttp`` that we leave to integration
testing. The split lets us unit-test pagination + dedup behavior
without an HTTP mock.
"""
from __future__ import annotations

from typing import Any

from krakey.plugins.searxng_search.client import (
    MAX_PAGES,
    build_body,
    paginate,
)


# --------------------------------------------------------------------
# build_body — pure helper
# --------------------------------------------------------------------


def test_build_body_minimal():
    body = build_body("hello", {})
    assert body == {"q": "hello", "format": "json"}


def test_build_body_skips_empty_string_filters():
    body = build_body("hello", {
        "categories": "",
        "language": "",
        "time_range": "",
        "safesearch": 0,
        "engines": "",
    })
    # Only ``safesearch`` stays — empty strings are dropped, but a
    # numeric 0 is intentional ("safesearch off") and forwarded.
    assert body == {"q": "hello", "format": "json", "safesearch": 0}


def test_build_body_skips_none_filters():
    body = build_body("hello", {"categories": None, "language": None})
    assert body == {"q": "hello", "format": "json"}


def test_build_body_includes_provided_filters():
    body = build_body("hello", {
        "categories": "general,news",
        "language": "en",
        "time_range": "week",
        "safesearch": 1,
        "engines": "google,bing",
    })
    assert body == {
        "q": "hello",
        "format": "json",
        "categories": "general,news",
        "language": "en",
        "time_range": "week",
        "safesearch": 1,
        "engines": "google,bing",
    }


# --------------------------------------------------------------------
# paginate — pure-logic loop
# --------------------------------------------------------------------


def _result(url: str, title: str = "") -> dict[str, Any]:
    return {"url": url, "title": title or url, "content": ""}


class _ScriptedFetcher:
    """Stub for ``fetch_page`` — returns one scripted page per call.

    Records every body it was handed so tests can assert ``pageno``
    progression."""

    def __init__(self, pages: list[list[dict[str, Any]]]):
        self._pages = pages
        self.bodies: list[dict[str, Any]] = []

    async def __call__(
        self, body: dict[str, Any],
    ) -> list[dict[str, Any]]:
        self.bodies.append(body)
        idx = len(self.bodies) - 1
        if idx >= len(self._pages):
            return []
        return self._pages[idx]


async def test_paginate_stops_at_max_results():
    fetcher = _ScriptedFetcher([
        [_result(f"u{i}") for i in range(10)],
        [_result(f"u{i}") for i in range(10, 20)],
        [_result(f"u{i}") for i in range(20, 30)],
    ])

    out = await paginate(
        {"q": "x"}, max_results=15, fetch_page=fetcher,
    )

    assert len(out) == 15
    # Page 1 (10) + part of page 2 (5 more) → exactly two fetches.
    assert len(fetcher.bodies) == 2
    assert fetcher.bodies[0]["pageno"] == 1
    assert fetcher.bodies[1]["pageno"] == 2


async def test_paginate_stops_on_empty_page():
    fetcher = _ScriptedFetcher([
        [_result(f"u{i}") for i in range(5)],
        [],  # SearXNG out of results
    ])

    out = await paginate(
        {"q": "x"}, max_results=50, fetch_page=fetcher,
    )

    assert len(out) == 5
    assert len(fetcher.bodies) == 2


async def test_paginate_stops_when_all_dupes():
    """Page contributes zero new URLs → stop. Otherwise we'd burn the
    page budget on an engine that keeps re-emitting the same hits."""
    fetcher = _ScriptedFetcher([
        [_result("u0"), _result("u1")],
        [_result("u0"), _result("u1")],  # identical → all dupes
    ])

    out = await paginate(
        {"q": "x"}, max_results=50, fetch_page=fetcher,
    )

    assert [r["url"] for r in out] == ["u0", "u1"]
    assert len(fetcher.bodies) == 2  # second page fetched, then quit


async def test_paginate_dedups_by_url_across_pages():
    fetcher = _ScriptedFetcher([
        [_result("u0"), _result("u1"), _result("u2")],
        [_result("u2"), _result("u3")],  # u2 repeats; u3 fresh
    ])

    out = await paginate(
        {"q": "x"}, max_results=50, fetch_page=fetcher,
    )

    assert [r["url"] for r in out] == ["u0", "u1", "u2", "u3"]


async def test_paginate_caps_at_max_pages():
    # Every page returns one fresh result — would loop forever if
    # not for MAX_PAGES.
    fetcher = _ScriptedFetcher([
        [_result(f"u{i}")] for i in range(MAX_PAGES + 5)
    ])

    out = await paginate(
        {"q": "x"}, max_results=999, fetch_page=fetcher,
    )

    assert len(out) == MAX_PAGES
    assert len(fetcher.bodies) == MAX_PAGES


async def test_paginate_pageno_increments_from_one():
    fetcher = _ScriptedFetcher([
        [_result("u0")], [_result("u1")], [_result("u2")],
    ])

    await paginate(
        {"q": "x"}, max_results=3, fetch_page=fetcher,
    )

    assert [b["pageno"] for b in fetcher.bodies] == [1, 2, 3]


async def test_paginate_preserves_body_base_keys():
    """Filters (categories / language / ...) must propagate to every
    paginated request, NOT just the first."""
    fetcher = _ScriptedFetcher([
        [_result("u0")], [_result("u1")],
    ])

    await paginate(
        {"q": "x", "categories": "news", "language": "en"},
        max_results=2, fetch_page=fetcher,
    )

    for body in fetcher.bodies:
        assert body["categories"] == "news"
        assert body["language"] == "en"


async def test_paginate_zero_max_results_no_fetch():
    fetcher = _ScriptedFetcher([[_result("u0")]])

    out = await paginate(
        {"q": "x"}, max_results=0, fetch_page=fetcher,
    )

    assert out == []
    assert fetcher.bodies == []
