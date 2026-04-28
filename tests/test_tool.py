from datetime import datetime

import pytest

from krakey.interfaces.tool import Tool, ToolRegistry
from krakey.models.stimulus import Stimulus


class MockTool(Tool):
    @property
    def name(self) -> str:
        return "mock"

    @property
    def description(self) -> str:
        return "Mock tool for testing."

    @property
    def parameters_schema(self) -> dict:
        return {"query": "string"}

    async def execute(self, intent, params) -> Stimulus:
        return Stimulus(
            type="tool_feedback",
            source=f"tool:{self.name}",
            content=f"got:{intent}",
            timestamp=datetime.now(),
        )


async def test_register_and_get_by_name():
    reg = ToolRegistry()
    t = MockTool()
    reg.register(t)
    assert reg.get("mock") is t


async def test_get_unknown_raises_keyerror():
    reg = ToolRegistry()
    with pytest.raises(KeyError):
        reg.get("missing")


async def test_execute_returns_stimulus():
    reg = ToolRegistry()
    reg.register(MockTool())
    out = await reg.get("mock").execute("hello", {})
    assert isinstance(out, Stimulus)
    assert out.content == "got:hello"
    assert out.source == "tool:mock"


def test_list_descriptions_contains_registered():
    reg = ToolRegistry()
    reg.register(MockTool())
    descs = reg.list_descriptions()
    assert any(d["name"] == "mock" for d in descs)
    assert any("testing" in d["description"] for d in descs)


def test_register_duplicate_raises():
    reg = ToolRegistry()
    reg.register(MockTool())
    with pytest.raises(ValueError):
        reg.register(MockTool())
