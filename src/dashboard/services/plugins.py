"""PluginsService — Protocol for the /api/plugins route."""
from __future__ import annotations

from typing import Any, Protocol


class PluginsService(Protocol):
    """Unified tentacle + sensory + plugin report.

    The shape mirrors what Runtime.plugin_report() returns, but routes
    depend on this Protocol so a fake (or a non-Runtime backing store)
    can substitute in tests.
    """

    def report(self) -> dict[str, Any]: ...
