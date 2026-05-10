"""Phase 3 / B: Search tool (web search via injectable backend)."""
import pytest

from krakey.models.stimulus import Stimulus
from krakey.plugins.duckduckgo_search import SearchTool


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


def test_tool_metadata():
    t = SearchTool(backend=FakeBackend())
    assert t.name == "search"
    assert t.description
    assert isinstance(t.parameters_schema, dict)


async def test_returns_formatted_results():
    backend = FakeBackend(results=[
        {"title": "Python", "href": "https://python.org",
         "body": "Programming language"},
        {"title": "PyPI", "href": "https://pypi.org",
         "body": "Package index"},
    ])
    t = SearchTool(backend=backend, max_results=5)
    stim = await t.execute("python", {})

    assert isinstance(stim, Stimulus)
    assert stim.type == "tool_feedback"
    assert stim.source == "tool:search"
    assert "python.org" in stim.content
    assert "Programming language" in stim.content
    assert backend.calls[0] == ("python", 5)


async def test_query_param_overrides_intent():
    backend = FakeBackend(results=[])
    t = SearchTool(backend=backend)
    await t.execute("free-form intent", {"query": "actual query", "max_results": 3})
    assert backend.calls[0] == ("actual query", 3)


async def test_empty_results_returns_clear_message():
    t = SearchTool(backend=FakeBackend(results=[]))
    stim = await t.execute("nothing matches", {})
    assert "no results" in stim.content.lower()


async def test_backend_failure_returns_error_stimulus():
    t = SearchTool(backend=FakeBackend(raises=RuntimeError("net down")))
    stim = await t.execute("query", {})
    assert "search failed" in stim.content.lower()
    assert "net down" in stim.content


async def test_max_results_default_used_when_param_missing():
    backend = FakeBackend(results=[])
    t = SearchTool(backend=backend, max_results=7)
    await t.execute("q", {})
    assert backend.calls[0][1] == 7


async def test_results_truncate_long_body():
    backend = FakeBackend(results=[
        {"title": "X", "href": "h",
         "body": "a" * 1000},  # very long
    ])
    t = SearchTool(backend=backend)
    stim = await t.execute("q", {})
    # Body should be truncated for prompt friendliness
    assert len(stim.content) < 5000


# ---------------------------------------------------------------------
# max_results: Self autonomously controls breadth per call
# ---------------------------------------------------------------------


def test_parameters_schema_advertises_max_results_range():
    """Schema must surface the int type + cap so Self knows the
    contract from [CAPABILITIES] alone."""
    schema = SearchTool(backend=FakeBackend()).parameters_schema
    assert schema["type"] == "object"
    assert schema["properties"]["max_results"]["type"] == "integer"
    assert schema["properties"]["max_results"]["minimum"] == 1
    assert schema["properties"]["max_results"]["maximum"] == 25
    assert schema["properties"]["query"]["type"] == "string"


def test_description_mentions_self_control_of_count():
    desc = SearchTool(backend=FakeBackend()).description
    assert "max_results" in desc
    # Should hint at the scaling decision Self makes.
    assert "25" in desc


async def test_self_can_request_more_than_default():
    """Per-call override beats the constructor default."""
    backend = FakeBackend(results=[])
    t = SearchTool(backend=backend, max_results=5)
    await t.execute("q", {"max_results": 15})
    assert backend.calls[0][1] == 15


async def test_max_results_clamped_to_cap():
    """A wild value gets soft-clamped, not rejected — Self still
    gets results, just capped at the ceiling."""
    backend = FakeBackend(results=[])
    t = SearchTool(backend=backend, max_results=5)
    await t.execute("q", {"max_results": 9999})
    assert backend.calls[0][1] == 25


async def test_max_results_rejects_non_integer():
    backend = FakeBackend(results=[])
    t = SearchTool(backend=backend)
    stim = await t.execute("q", {"max_results": "abc"})
    assert "must be an integer" in stim.content
    assert backend.calls == []  # backend never called


async def test_max_results_rejects_zero_or_negative():
    backend = FakeBackend(results=[])
    t = SearchTool(backend=backend)
    stim = await t.execute("q", {"max_results": 0})
    assert "must be >= 1" in stim.content
    assert backend.calls == []
