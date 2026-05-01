"""Per-plugin user config persistence.

Each plugin owns one YAML file at ``<root>/<name>/config.yaml`` —
co-located with its meta.yaml + component code. The dashboard
writes here on save; the runtime + plugin code read from here.

The legacy ``workspace/plugin-configs/<name>.yaml`` location was
retired alongside the FilePluginConfigStore's old enabled-flag
handling; enable/disable is now driven exclusively by the central
``config.yaml``'s ``plugins:`` list. Per-plugin config files contain
ONLY the plugin's own settings (sandbox flag, history_path, llm
purpose bindings, etc) — no enabled flag.

Built-in plugin code lives at ``krakey/plugins/<name>/``. Their user
config still lives in workspace (``workspace/plugins/<name>/config.yaml``)
to keep the repo unmodified by user edits — same path users get for
third-party plugins they install at ``workspace/plugins/<name>/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class FilePluginConfigStore:
    """Read/write the per-plugin ``<root>/<name>/config.yaml`` file.

    Thin wrapper around YAML I/O. ``read`` returns ``{}`` when the
    file is absent (no auto-init — plugins should fall back to their
    own defaults; the file is only written on dashboard save or by
    the user).
    """

    def __init__(self, root: Path | str):
        self._root = Path(root)

    def path_for(self, name: str) -> Path:
        return self._root / name / "config.yaml"

    def read(self, name: str) -> dict[str, Any]:
        path = self.path_for(name)
        if not path.exists():
            return {}
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}

    def write(self, name: str, config: dict[str, Any]) -> Path:
        path = self.path_for(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return path
