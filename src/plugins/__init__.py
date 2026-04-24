"""Plugin loader — dynamic tentacle/sensory discovery from workspace.

At boot Runtime scans `src/plugins/builtin/` (ships with Krakey) and
`workspace/plugins/` (user-dropped), imports each plugin project in
isolation (no sys.path pollution), and registers the produced
tentacles + sensories alongside the core ones.

Plugin contract lives in PLUGINS.md at repo root.
"""
from src.plugins.loader import PluginInfo, discover_plugins  # noqa: F401
