"""Tentacle ABC + Registry (DevSpec §5.1)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.models.stimulus import Stimulus


class Tentacle(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]: ...

    @property
    def sandboxed(self) -> bool:
        return True

    @property
    def is_internal(self) -> bool:
        """When True, the tentacle's output is for Self's inner consumption
        only (e.g. memory_recall) and should NOT be rendered to the human
        as Krakey's outward chat. Defaults to False — most tentacles
        (action, …) speak on Krakey's behalf."""
        return False

    @abstractmethod
    async def execute(self, intent: str, params: dict[str, Any]) -> Stimulus: ...


class TentacleRegistry:
    def __init__(self):
        self._tentacles: dict[str, Tentacle] = {}

    def register(self, tentacle: Tentacle) -> None:
        if tentacle.name in self._tentacles:
            raise ValueError(f"tentacle '{tentacle.name}' already registered")
        self._tentacles[tentacle.name] = tentacle

    def get(self, name: str) -> Tentacle:
        if name not in self._tentacles:
            raise KeyError(f"no tentacle named '{name}'")
        return self._tentacles[name]

    def list_descriptions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters_schema": t.parameters_schema,
            }
            for t in self._tentacles.values()
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._tentacles
