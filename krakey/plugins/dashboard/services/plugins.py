"""PluginsService — Protocol for the /api/plugins route."""
from __future__ import annotations

from typing import Any, Protocol


class PluginsService(Protocol):
    """Unified tool + channel + plugin report.

    Routes depend on this Protocol so a fake (or a non-Runtime backing
    store) can substitute in tests. The default adapter combines two
    things:
      * runtime observation (which components are currently loaded —
        from ``runtime.loaded_plugin_report()``);
      * direct disk reads/writes against ``workspace/plugins/<name>/
        config.yaml`` via its own ``FilePluginConfigStore``.

    The runtime is NEVER in the write path — the dashboard owns
    plugin-config edits.
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

    def deps_status(self) -> dict[str, Any]:
        """Snapshot of plugin-dependency install state.

        Returns::

            {
              "pending":  bool,            # any plugin needs install?
              "plugins": {
                "<name>": {
                  "dependencies": [...pip-spec strings],
                  "post_install": [{args, description, optional}],
                  "installed":    bool,    # in install_state.json's
                                            # installed list
                  "satisfied":    bool,    # currently importable +
                                            # post_install marker present
                },
                ...
              },
              "state": {
                  "installed_at": "iso-ts" | null,
                  "deps_hash":    "..."   | null,
              },
            }

        Used by the dashboard's plugin list to show ⚠ next to
        plugins whose deps haven't been installed in this venv.
        """

    async def hot_reload(self) -> dict[str, Any]:
        """Re-read ``config.yaml``'s ``plugins:`` list and hot-add
        any plugins enabled there but not currently loaded. Returns
        the runtime's ``hot_reload_plugins`` report (added /
        skipped / errors / still_pending_remove). Plugins flagged
        as ``still_pending_remove`` need a full restart — the
        dashboard surfaces this as a "Restart Krakey" hint."""

    def install(self, body: dict[str, Any]) -> dict[str, Any]:
        """Run ``krakey install`` programmatically. Body fields::

            {
              "upgrade":  bool,    # default false
            }

        Returns ``{rc: int, stdout: str, stderr: str}``. Output
        truncated to keep the JSON response from blowing up on a
        chatty pip session — full log goes to the runtime log if
        the dashboard is attached to one. State file IS written
        if rc==0; subsequent /api/plugins/deps_status reflects
        the new state."""
