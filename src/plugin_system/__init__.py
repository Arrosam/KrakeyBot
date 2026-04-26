"""Plugin system infrastructure (discovery + per-plugin config persistence).

Sits between Runtime / Dashboard and the actual plugin folders. The
plugin folders themselves (``src/plugins/<name>/`` for built-in,
``workspace/plugins/<name>/`` for user-installed) only contain plugin
CODE + ``meta.yaml`` + per-plugin ``config.yaml`` (user settings) —
they don't import anything from this package.

Two responsibilities, two modules:

  * ``discovery``  — pure-text scan of plugin folders' ``meta.yaml``
                     files, lazy ``importlib`` loading on enable. Used
                     by the dashboard (to show what's available) and
                     by Runtime (to load enabled plugins by name).
  * ``config``     — ``FilePluginConfigStore`` for the dashboard's
                     plugin-config form (enabled flag + values). Will
                     be merged with the in-folder ``config.yaml`` in a
                     follow-up; currently they coexist.
"""
from src.plugin_system.config import FilePluginConfigStore  # noqa: F401
from src.plugin_system.discovery import (  # noqa: F401
    ComponentMetadata,
    PluginMetadata,
    discover_plugins,
    load_component,
    load_plugin_meta,
)
