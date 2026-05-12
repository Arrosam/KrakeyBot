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

    def find_stale_configs(self) -> list[dict[str, Any]]:
        """List per-plugin config dirs that no longer back any
        installed plugin.

        A directory under the plugin-configs root is "stale" if it
        is NOT itself a workspace plugin (no ``meta.yaml``) AND its
        name is absent from the live catalogue. These are leftovers
        from removed / renamed plugins; the dashboard's "delete
        stale" button removes them.

        Returns a list of ``{name, path, has_config}`` dicts. The
        ``has_config`` flag tells the UI whether the folder
        contains a saved ``config.yaml`` (vs. an empty leftover
        folder) so it can label the row accordingly.
        """

    def delete_stale_config(self, name: str) -> dict[str, Any]:
        """Delete a single stale plugin-config folder.

        Safety contract:
          * ``name`` must be a single segment of safe characters
            (no path separators / dots / leading slash) — raises
            ``ValueError`` otherwise so a malicious or malformed
            name can never escape the plugin-configs root.
          * The named plugin must NOT be in the live catalogue and
            its folder must NOT contain a ``meta.yaml`` (which
            would mean it IS a workspace plugin, just one whose
            meta failed to parse). Both rejected with
            ``ValueError`` so a typo can't wipe live plugin data.
          * Missing folder → ``LookupError`` (404 from the route).

        Returns ``{name, path, deleted: True}`` on success.
        """
