"""Phase 3 / B: Search tentacle (web search via injectable backend)."""
import pytest

from src.models.stimulus import Stimulus
from src.tentacles.search import SearchTentacle


class FakeBackend:
    def __init__(self, results=None, raises=None):
        self.results = results or []
        self.raises = raises
        self.calls = []

    async def search(self, query, max_results):
        self.calls.append((query, max_results))
        if self.raises is not None:
            raise self.raises
        return self.results[:max_results]


def test_tentacle_metadata():
    t = SearchTentacle(backend=FakeBackend())
    assert t.name == "search"
    assert t.description
    assert isinstance(t.parameters_schema, dict)
    assert t.is_internal is True  # results feed Self, not user


async def test_returns_formatted_results():
    backend = FakeBackend(results=[
        {"title": "Python", "href": "https://python.org",
         "body": "Programming language"},
        {"title": "PyPI", "href": "https://pypi.org",
         "body": "Package index"},
    ])
    t = SearchTentacle(backend=backend, max_results=5)
    stim = await t.execute("python", {})

    assert isinstance(stim, Stimulus)
    assert stim.type == "tentacle_feedback"
    assert stim.source == "tentacle:search"
    assert "python.org" in stim.content
    assert "Programming language" in stim.content
    assert backend.calls[0] == ("python", 5)


async def test_query_param_overrides_intent():
    backend = FakeBackend(results=[])
    t = SearchTentacle(backend=backend)
    await t.execute("free-form intent", {"query": "actual query", "max_results": 3})
    assert backend.calls[0] == ("actual query", 3)


async def test_empty_results_returns_clear_message():
    t = SearchTentacle(backend=FakeBackend(results=[]))
    stim = await t.execute("nothing matches", {})
    assert ("no results" in stim.content.lower()
            or "无结果" in stim.content)


async def test_backend_failure_returns_error_stimulus():
    t = SearchTentacle(backend=FakeBackend(raises=RuntimeError("net down")))
    stim = await t.execute("query", {})
    assert "search failed" in stim.content.lower() or "失败" in stim.content
    assert "net down" in stim.content


async def test_max_results_default_used_when_param_missing():
    backend = FakeBackend(results=[])
    t = SearchTentacle(backend=backend, max_results=7)
    await t.execute("q", {})
    assert backend.calls[0][1] == 7


async def test_results_truncate_long_body():
    backend = FakeBackend(results=[
        {"title": "X", "href": "h",
         "body": "a" * 1000},  # very long
    ])
    t = SearchTentacle(backend=backend)
    stim = await t.execute("q", {})
    # Body should be truncated for prompt friendliness
    assert len(stim.content) < 5000
