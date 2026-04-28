from datetime import datetime

import pytest

from krakey.interfaces.tentacle import Tentacle, TentacleRegistry
from krakey.models.stimulus import Stimulus


class MockTentacle(Tentacle):
    @property
    def name(self) -> str:
        return "mock"

    @property
    def description(self) -> str:
        return "Mock tentacle for testing."

    @property
    def parameters_schema(self) -> dict:
        return {"query": "string"}

    async def execute(self, intent, params) -> Stimulus:
        return Stimulus(
            type="tentacle_feedback",
            source=f"tentacle:{self.name}",
            content=f"got:{intent}",
            timestamp=datetime.now(),
        )


async def test_register_and_get_by_name():
    reg = TentacleRegistry()
    t = MockTentacle()
    reg.register(t)
    assert reg.get("mock") is t


async def test_get_unknown_raises_keyerror():
    reg = TentacleRegistry()
    with pytest.raises(KeyError):
        reg.get("missing")


async def test_execute_returns_stimulus():
    reg = TentacleRegistry()
    reg.register(MockTentacle())
    out = await reg.get("mock").execute("hello", {})
    assert isinstance(out, Stimulus)
    assert out.content == "got:hello"
    assert out.source == "tentacle:mock"


def test_list_descriptions_contains_registered():
    reg = TentacleRegistry()
    reg.register(MockTentacle())
    descs = reg.list_descriptions()
    assert any(d["name"] == "mock" for d in descs)
    assert any("testing" in d["description"] for d in descs)


def test_register_duplicate_raises():
    reg = TentacleRegistry()
    reg.register(MockTentacle())
    with pytest.raises(ValueError):
        reg.register(MockTentacle())
