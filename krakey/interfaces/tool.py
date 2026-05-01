"""Tool ABC + Registry (DevSpec §5.1)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from krakey.models.stimulus import Stimulus


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]: ...

    @abstractmethod
    async def execute(self, intent: str, params: dict[str, Any]) -> Stimulus: ...


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"no tool named '{name}'")
        return self._tools[name]

    def list_descriptions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters_schema": t.parameters_schema,
            }
            for t in self._tools.values()
        ]

    def names(self) -> list[str]:
        """Sorted snapshot of all registered tool names."""
        return sorted(self._tools.keys())

    def all(self) -> list[Tool]:
        """Snapshot of every registered tool (insertion order)."""
        return list(self._tools.values())

    def __contains__(self, name: str) -> bool:
        return name in self._tools
