"""Plugin discovery + per-plugin config layer.

Each plugin lives at ``src/plugins/builtin/<name>/`` (ships with
Krakey) or ``workspace/plugins/<name>/`` (user-dropped) and is
described by a ``meta.yaml`` manifest. The actual loader is
``src.plugins.unified_discovery``; per-plugin YAML config lives
under ``workspace/plugin-configs/`` via
``src.plugins.plugin_config.FilePluginConfigStore``.

The dashboard's plugin-row dataclass lives in
``src.runtime.plugin_registrar.PluginInfo`` (one module up because
the registrar is the only producer).
"""
