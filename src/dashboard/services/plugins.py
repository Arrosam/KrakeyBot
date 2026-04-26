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

    def update_config(
        self, project: str, body: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist a dashboard edit to a plugin's
        ``workspace/plugins/<project>/config.yaml`` file.

        Body shape: ``{"values": {...}}``. ``enabled`` is no longer
        per-plugin — enable/disable is driven by the central
        ``config.yaml``'s ``plugins:`` list — and is silently dropped
        if present in the body. Returns a summary dict (project name,
        file path, resulting config).
        """
