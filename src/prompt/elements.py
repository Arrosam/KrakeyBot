"""PromptElements — ordered named layers of the Self prompt.

The runtime builds a ``PromptElements`` instance each heartbeat with
the canonical default layers (DNA, self-model, capabilities,
action_format, stimulus, recall, in_mind_round, history, status,
heartbeat_question). Plugins can read / overwrite / delete any
element via a per-plugin binding (``elements.for_plugin(name)``);
modifications are tracked. If a second plugin modifies an element
that an earlier plugin already touched in the same heartbeat,
``PromptElements`` logs a warning so the conflict is visible.

Render order = insertion order. Plugins that need a value injected at
a specific position should write into a key the runtime already
defined as empty (e.g. ``in_mind_round`` is pre-inserted between
``recall`` and ``history``); plugins that introduce truly new keys
get them appended at the end.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Iterable

_log = logging.getLogger(__name__)


class PromptElements:
    """Ordered named string layers + per-plugin modification tracking."""

    def __init__(
        self, initial: Iterable[tuple[str, str]] | None = None,
    ):
        self._elements: OrderedDict[str, str] = OrderedDict(initial or [])
        # key → list of plugin names that have written to it (in order)
        self._modified_by: dict[str, list[str]] = {}

    # ---- read --------------------------------------------------------

    def __contains__(self, key: str) -> bool:
        return key in self._elements

    def __getitem__(self, key: str) -> str:
        return self._elements[key]

    def get(self, key: str, default: str = "") -> str:
        return self._elements.get(key, default)

    def keys(self) -> list[str]:
        return list(self._elements.keys())

    # ---- bind to a plugin -------------------------------------------

    def for_plugin(self, plugin_name: str) -> "BoundPromptElements":
        """Return a thin wrapper that records ``plugin_name`` against
        every modification routed through it."""
        return BoundPromptElements(self, plugin_name)

    # ---- internal mutation (called by BoundPromptElements) ----------

    def _set(self, key: str, value: str, plugin: str) -> None:
        self._note_modification(key, plugin)
        self._elements[key] = value

    def _delete(self, key: str, plugin: str) -> None:
        self._note_modification(key, plugin)
        self._elements.pop(key, None)

    def _note_modification(self, key: str, plugin: str) -> None:
        existing = self._modified_by.get(key)
        if existing:
            _log.warning(
                "PromptElements: plugin %r modifying element %r already "
                "modified this heartbeat by %s",
                plugin, key, existing,
            )
        self._modified_by.setdefault(key, []).append(plugin)

    # ---- render ------------------------------------------------------

    def render(self, separator: str = "\n\n") -> str:
        """Concatenate values in insertion order; skip empty/None."""
        return separator.join(
            v for v in self._elements.values() if v
        )

    # ---- introspection (for tests / debugging) ----------------------

    def modified_by(self, key: str) -> list[str]:
        """Plugin names that wrote to ``key`` this heartbeat (in order)."""
        return list(self._modified_by.get(key, []))


class BoundPromptElements:
    """Per-plugin wrapper. Reads pass through; writes/deletes record
    the plugin name on the underlying ``PromptElements``."""

    def __init__(self, elements: PromptElements, plugin_name: str):
        self._elements = elements
        self._plugin = plugin_name

    def __contains__(self, key: str) -> bool:
        return key in self._elements

    def __getitem__(self, key: str) -> str:
        return self._elements[key]

    def get(self, key: str, default: str = "") -> str:
        return self._elements.get(key, default)

    def keys(self) -> list[str]:
        return self._elements.keys()

    def __setitem__(self, key: str, value: str) -> None:
        self._elements._set(key, value, self._plugin)

    def __delitem__(self, key: str) -> None:
        self._elements._delete(key, self._plugin)
