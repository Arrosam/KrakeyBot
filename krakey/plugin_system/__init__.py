"""Plugin system infrastructure (load + per-plugin config persistence).

Two responsibilities, two modules:

  * ``loader``  — runtime side: ``load_plugin_meta(name)`` reads one
                  meta.yaml by name, ``load_component(c, ctx)`` lazy-
                  imports + invokes the factory. ``parse_meta(path)``
                  is the shared parser also used by the dashboard's
                  catalogue scanner.
  * ``config``  — ``FilePluginConfigStore`` for the dashboard's
                  in-folder config.yaml read/write.

Catalogue scanning ("show me all installed plugins") is intentionally
NOT here — that's a Web UI concern and lives in
``krakey/plugins/dashboard/services/plugin_catalogue.py``. Runtime only loads
plugins by name (the names listed in ``config.yaml``'s ``plugins:``).
"""
from krakey.plugin_system.config import FilePluginConfigStore  # noqa: F401
from krakey.plugin_system.loader import (  # noqa: F401
    BUILTIN_ROOT,
    WORKSPACE_ROOT,
    ComponentMetadata,
    PluginMetadata,
    load_component,
    load_plugin_meta,
    parse_meta,
)
