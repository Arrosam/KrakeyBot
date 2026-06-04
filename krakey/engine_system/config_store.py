"""Per-engine settings file store.

Each engine impl may declare a ``config_path`` in its ``meta.yaml``
``builtin_engines`` entry — a workspace-relative path pointing at that
engine's own settings YAML file (e.g. ``data/memory/settings.yaml``).
This module provides ``FileEngineConfigStore``, which reads and writes
those files.

Absent file / empty path → ``{}`` — the engine then uses its own
built-in defaults. No auto-initialisation on read.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class FileEngineConfigStore:
    """Read/write a single engine impl's own settings YAML file.

    Each engine impl declares a workspace-relative ``config_path`` in its
    meta.yaml. This store reads/writes ``<workspace_root>/<config_path>``.
    ``read`` returns {} when the path is empty or the file is absent (the
    engine then uses its own defaults) — no auto-init. ``write`` is only
    called on dashboard save or by the engine itself.
    """

    def __init__(self, workspace_root: Path | str) -> None:
        self._root = Path(workspace_root)

    def read(self, config_path: str) -> dict[str, Any]:
        """Return the stored config dict, or ``{}`` on any non-fatal miss.

        Non-fatal misses:
          - ``config_path`` is empty / falsy → ``{}``
          - file does not exist → ``{}``
          - YAML parses to a non-dict (list, scalar) → ``{}``

        ``yaml.YAMLError`` (malformed YAML) is re-raised so a corrupt
        file surfaces loudly rather than silently returning ``{}``.
        """
        if not config_path:
            return {}
        full_path = self._root / config_path
        if not full_path.exists():
            return {}
        raw = yaml.safe_load(full_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        return raw

    def write(self, config_path: str, config: dict[str, Any]) -> Path:
        """Persist ``config`` to ``<workspace_root>/<config_path>``.

        Creates parent directories as needed. Raises ``ValueError`` when
        ``config_path`` is empty. Returns the full path written.
        """
        if not config_path:
            raise ValueError(
                "FileEngineConfigStore.write: config_path must be a "
                "non-empty workspace-relative path"
            )
        full_path = self._root / config_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return full_path
