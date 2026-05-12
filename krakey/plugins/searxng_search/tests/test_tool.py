"""Unit tests for ``searxng_search`` Tool.

Run from repo root:

    pytest krakey/plugins/searxng_search

(``pytest.ini`` pins ``testpaths = tests`` + ``asyncio_mode = auto``,
matching the ``cli_exec`` per-plugin contract.)

Tests use a ``FakeBackend`` that records call params and returns a
scripted SearXNG-shaped payload, so no real HTTP / Docker is touched.
"""
from __future__ import annotations

from typing import Any

import pytest

from krakey.plugins.searxng_search.tool import (
    DEFAULT_MAX_RESULTS,
    MAX_RESULTS_CAP,
    SearxngSearchTool,
)


class FakeBackend:
    """Minimal SearxngBackend stub. Records every ``search()`` call's
    args (including the keyword-only ``max_results``) + returns a
    scripted payload (or raises a scripted error)."""

    def __init__(
        self,
        results: list[dict[str, Any]] | None = None,
        raises: BaseException | None = None,
    ):
        self._results = results if results is not None else []
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    async def search(
        self, query: str, *, max_results: int, **filters: Any,
    ) -> dict[str, Any]:
        self.calls.append(
            {"query": query, "max_results": max_results, **filters},
        )
        if self._raises is not None:
            raise self._raises
        return {
            "query": query,
            "number_of_results": len(self._results),
            "results": self._results,
        }


# --------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------


async def test_happy_path_returns_tool_feedback_stimulus():
    backend = FakeBackend(results=[
        {
            "title": "Title-X",
            "url": "https://x.example",
            "content": "snippet x",
            "engine": "google",
        },
    ])
    tool = SearxngSearchTool(backend=backend, default_max_results=10)

    s = await tool.execute("anything", {"query": "foo"})

    assert s.type == "tool_feedback"
    assert s.source == "tool:searxng_search"
    assert "foo" in s.content
    assert "Title-X" in s.content
    assert "https://x.example" in s.content
    assert s.adrenalin is False


async def test_query_falls_back_to_intent_when_omitted():
    backend = FakeBackend(results=[])
    tool = SearxngSearchTool(backend=backend, default_max_results=5)

    await tool.execute("python asyncio", {})

    assert backend.calls[0]["query"] == "python asyncio"


async def test_minimal_call_only_query_uses_all_defaults():
    """Smoke test for "only `query` is required". Self can fire the
    tool with nothing else and the call still goes through with
    sane defaults."""
    backend = FakeBackend(results=[])
    tool = SearxngSearchTool(
        backend=backend, default_max_results=10,
        default_categories="", default_language="",
        default_safesearch=0,
    )

    await tool.execute("ignored intent", {"query": "foo"})

    c = backend.calls[0]
    assert c["query"] == "foo"
    assert c["max_results"] == 10
    assert c["categories"] == ""
    assert c["language"] == ""
    assert c["time_range"] == ""
    assert c["safesearch"] == 0
    assert c["engines"] == ""


async def test_max_results_threaded_to_backend():
    backend = FakeBackend(results=[])
    tool = SearxngSearchTool(backend=backend, default_max_results=7)

    await tool.execute("x", {"query": "y"})
    await tool.execute("x", {"query": "y", "max_results": 25})

    assert backend.calls[0]["max_results"] == 7
    assert backend.calls[1]["max_results"] == 25


async def test_max_results_clamped_value_threaded_to_backend():
    """Above-cap values are clamped before reaching the backend so a
    runaway max_results from Self can't burn the backend's page
    budget pointlessly."""
    backend = FakeBackend(results=[])
    tool = SearxngSearchTool(backend=backend, default_max_results=5)

    await tool.execute(
        "x", {"query": "y", "max_results": MAX_RESULTS_CAP + 100},
    )

    assert backend.calls[0]["max_results"] == MAX_RESULTS_CAP


async def test_categories_threaded_to_backend():
    backend = FakeBackend(results=[])
    tool = SearxngSearchTool(backend=backend, default_max_results=5)

    await tool.execute(
        "x", {"query": "y", "categories": "general,news"},
    )

    assert backend.calls[0]["categories"] == "general,news"


async def test_engines_array_threaded_to_backend_as_csv():
    backend = FakeBackend(results=[])
    tool = SearxngSearchTool(backend=backend, default_max_results=5)

    await tool.execute(
        "x", {"query": "y", "engines": ["google", "bing"]},
    )

    # Backend receives engines as csv even though Self passes an array;
    # SearXNG's HTTP API takes csv.
    assert backend.calls[0]["engines"] == "google,bing"


async def test_language_safesearch_time_range_threaded():
    backend = FakeBackend(results=[])
    tool = SearxngSearchTool(backend=backend, default_max_results=5)

    await tool.execute(
        "x",
        {
            "query": "y",
            "language": "en",
            "safesearch": 1,
            "time_range": "week",
        },
    )

    c = backend.calls[0]
    assert c["language"] == "en"
    assert c["safesearch"] == 1
    assert c["time_range"] == "week"


async def test_results_post_trimmed_to_max_results():
    backend = FakeBackend(results=[
        {
            "title": f"T{i}",
            "url": f"https://x.example/{i}",
            "content": "",
            "engine": "g",
        }
        for i in range(20)
    ])
    tool = SearxngSearchTool(backend=backend, default_max_results=10)

    s = await tool.execute("x", {"query": "y", "max_results": 5})

    listed = [
        line for line in s.content.splitlines()
        if line and line[0].isdigit() and ". [" in line[:6]
    ]
    assert len(listed) == 5


async def test_max_results_above_cap_silently_clamped():
    backend = FakeBackend(results=[])
    tool = SearxngSearchTool(backend=backend, default_max_results=5)

    s = await tool.execute(
        "x", {"query": "y", "max_results": MAX_RESULTS_CAP + 100},
    )

    # Soft-cap; no error returned.
    assert "error:" not in s.content


async def test_per_plugin_default_max_results_used_when_omitted():
    backend = FakeBackend(results=[
        {"title": f"T{i}", "url": f"https://x/{i}",
         "content": "", "engine": "g"}
        for i in range(20)
    ])
    tool = SearxngSearchTool(backend=backend, default_max_results=3)

    s = await tool.execute("x", {"query": "y"})

    listed = [
        line for line in s.content.splitlines()
        if line and line[0].isdigit() and ". [" in line[:6]
    ]
    assert len(listed) == 3


# --------------------------------------------------------------------
# Bad params → error stimulus, fake backend never called
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "params, expect",
    [
        ({"max_results": 0}, "max_results"),
        ({"max_results": -3}, "max_results"),
        ({"max_results": "abc"}, "max_results"),
        ({"max_results": True}, "max_results"),
        ({"safesearch": 9}, "safesearch"),
        ({"safesearch": "x"}, "safesearch"),
        ({"safesearch": True}, "safesearch"),
        ({"time_range": "decade"}, "time_range"),
        ({"engines": "google"}, "engines"),  # must be an array
        ({"engines": [1, 2]}, "engines"),
        ({"categories": 9}, "categories"),
        ({"language": []}, "language"),
    ],
)
async def test_bad_params_return_error_without_calling_backend(
    params, expect,
):
    backend = FakeBackend()
    tool = SearxngSearchTool(backend=backend, default_max_results=5)

    s = await tool.execute("x", {"query": "y", **params})

    assert s.content.startswith("searxng_search error:")
    assert expect in s.content
    assert backend.calls == []


async def test_empty_query_and_intent_returns_error():
    backend = FakeBackend()
    tool = SearxngSearchTool(backend=backend, default_max_results=5)

    s = await tool.execute("", {"query": ""})

    assert "error:" in s.content
    assert backend.calls == []


# --------------------------------------------------------------------
# Backend failures → error stimulus
# --------------------------------------------------------------------


async def test_backend_exception_returns_error_stim():
    backend = FakeBackend(raises=RuntimeError("instance unreachable"))
    tool = SearxngSearchTool(backend=backend, default_max_results=5)

    s = await tool.execute("x", {"query": "y"})

    assert s.content.startswith("searxng_search error:")
    assert "instance unreachable" in s.content


async def test_no_results_returns_explanatory_stim():
    backend = FakeBackend(results=[])
    tool = SearxngSearchTool(backend=backend, default_max_results=5)

    s = await tool.execute("x", {"query": "obscure"})

    assert "No results" in s.content
    assert "obscure" in s.content


# --------------------------------------------------------------------
# Tool ABC surface — describe()-equivalent introspection
# --------------------------------------------------------------------


def test_static_tool_metadata():
    tool = SearxngSearchTool(
        backend=FakeBackend(), default_max_results=DEFAULT_MAX_RESULTS,
    )
    assert tool.name == "searxng_search"
    schema = tool.parameters_schema
    assert schema["type"] == "object"
    assert "query" in schema["required"]
    props = schema["properties"]
    for k in (
        "query", "max_results", "categories", "language",
        "time_range", "safesearch", "engines",
    ):
        assert k in props
    # Description names the cap so Self knows the per-call ceiling.
    assert str(MAX_RESULTS_CAP) in tool.description
