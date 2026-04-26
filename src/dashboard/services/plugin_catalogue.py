"""Plugin catalogue scanner — Web UI side.

Walks the same plugin folder roots as ``src/plugin_system/loader.py``
and returns one ``PluginMetadata`` per ``meta.yaml`` found. Pure-text
(no plugin module is imported), so the Web UI can show a complete
"available plugins" list without spinning up plugin code.

Lives under ``dashboard/services/`` because the **only** consumer is
the dashboard. Runtime never enumerates installed plugins — it loads
by name from ``config.yaml``'s ``plugins:`` list via
``plugin_system.load_plugin_meta(name)``.

Workspace overrides built-in (later iteration wins) so users can
shadow a shipped plugin's meta with a customised version.
"""
from __future__ import annotations

import sys

from src.plugin_system.loader import (
    BUILTIN_ROOT,
    WORKSPACE_ROOT,
    PluginMetadata,
    parse_meta,
)


def list_available_plugins() -> dict[str, PluginMetadata]:
    """Scan both plugin roots and return ``name → PluginMetadata``.

    Malformed manifests log a warning and are skipped — best-effort
    scan. The dashboard prefers to render a partial catalogue over
    crashing when a single plugin folder has a bad meta.yaml.
    """
    out: dict[str, PluginMetadata] = {}
    for root in (BUILTIN_ROOT, WORKSPACE_ROOT):
        if not root.exists():
            continue
        for meta_path in sorted(root.glob("*/meta.yaml")):
            try:
                meta = parse_meta(meta_path)
            except Exception as e:  # noqa: BLE001
                print(
                    f"warning: failed to parse {meta_path}: {e}; skipping",
                    file=sys.stderr,
                )
                continue
            out[meta.name] = meta
    return out
