"""Plugin catalogue scanner — pure-text enumeration of installable plugins.

Walks the same plugin folder roots as ``krakey/plugin_system/loader.py``
and returns one ``PluginMetadata`` per ``meta.yaml`` found. **No
plugin module is imported** — only the YAML manifest is read — so
callers can show a complete "available plugins" list without
spinning up plugin code or pulling in plugin-only dependencies.

Lives under ``plugin_system/`` because both the onboarding wizard
(non-plugin) and the dashboard plugin need it. Previously it was
buried inside ``dashboard/services/`` and the wizard's import of
it created a core-→-plugin coupling that broke the additive-plugin
invariant (see CLAUDE.md). Runtime itself never enumerates
installed plugins — it loads by name from ``config.yaml``'s
``plugins:`` list via ``plugin_system.load_plugin_meta(name)``.

Workspace overrides built-in (later iteration wins) so users can
shadow a shipped plugin's meta with a customised version.
"""
from __future__ import annotations

import sys

from krakey.plugin_system.loader import (
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
