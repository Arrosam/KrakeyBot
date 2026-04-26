"""Per-plugin config files.

One YAML file per plugin project at ``<root>/<project>.yaml``.
Eliminates the central `plugins:` pile inside ``config.yaml`` — each
plugin owns its own settings, dashboard reads/writes per file, and
plugins without a ``config_schema`` produce no file at all.

File lifecycle:

* **First discovery.** The loader calls ``load_or_init(project, schema)``
  with the manifest's ``config_schema``. If the file does not exist,
  one is written using schema defaults (plus ``enabled: false``).
  Legacy values under the ``plugins.<project>`` section of the old
  central ``config.yaml`` are carried across on first run as a
  one-time migration.
* **Subsequent runs.** The file is the sole source of truth.
* **Dashboard edits.** Route handlers call ``write(project, cfg)``
  to persist form submissions.

Plugins with an empty ``config_schema`` are respected: no file is
written, and the returned config is just ``{"enabled": <legacy?>}``.

This file is the only place that knows the on-disk layout; everything
else talks to the ``PluginConfigStore`` protocol.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import yaml


class PluginConfigStore(Protocol):
    """Shape the loader (and dashboard routes) depend on."""

    def load_or_init(
        self, project: str, schema: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return the final config dict for a project.

        The returned dict always carries an ``enabled`` key (loader
        contract). The implementation may persist a file so the next
        run is stable.
        """

    def peek_config(self, project: str) -> dict[str, Any]:
        """Best-effort config lookup without knowing the schema.

        Used by code paths that run *before* the loader imports
        modules (e.g. Runtime picking ``web_chat.history_path`` before
        plugin discovery). Returns ``{}`` if nothing is known.
        """

    def write(self, project: str, config: dict[str, Any]) -> Path:
        """Persist an updated config and return the file path."""

    def path_for(self, project: str) -> Path:
        """Filesystem path for a project's config file."""


# ---------------- file-backed implementation ----------------


class FilePluginConfigStore:
    """The real store: one YAML file per plugin, in ``root``.

    ``legacy_plugins`` is the deprecated central ``config.yaml``
    ``plugins:`` dict; values there seed newly-created files exactly
    once, then become irrelevant. Pass ``None`` or ``{}`` to skip
    migration.
    """

    def __init__(
        self,
        root: Path | str,
        legacy_plugins: dict[str, dict[str, Any]] | None = None,
    ):
        self._root = Path(root)
        self._legacy = dict(legacy_plugins or {})

    # ---- public API ---------------------------------------------------

    def path_for(self, project: str) -> Path:
        return self._root / f"{project}.yaml"

    def peek_config(self, project: str) -> dict[str, Any]:
        path = self.path_for(project)
        if path.exists():
            return _read_yaml(path)
        return dict(self._legacy.get(project) or {})

    def load_or_init(
        self, project: str, schema: list[dict[str, Any]],
    ) -> dict[str, Any]:
        path = self.path_for(project)
        if path.exists():
            return _read_yaml(path)

        legacy = dict(self._legacy.get(project) or {})

        # User rule: "如果没有声明可配置参数，那就不理" — a plugin
        # that declares no schema never gets a file. Only `enabled`
        # survives, sourced from legacy if present.
        if not schema:
            return {"enabled": bool(legacy.get("enabled", False)), **legacy}

        base: dict[str, Any] = {
            "enabled": bool(legacy.get("enabled", False)),
        }
        for field in schema:
            default = field.get("default")
            if default is not None:
                base[field["field"]] = default
        # Overlay any legacy keys — user's previous values trump defaults.
        for k, v in legacy.items():
            base[k] = v

        self._write(path, base)
        return base

    def write(self, project: str, config: dict[str, Any]) -> Path:
        path = self.path_for(project)
        self._write(path, config)
        return path

    # ---- internals ----------------------------------------------------

    def _write(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )


# ---------------- in-memory shim for tests ----------------


class DictPluginConfigStore:
    """Legacy in-memory store — wraps the old ``configs=dict`` path so
    tests that pre-date file-backed storage keep working. Never touches
    the filesystem."""

    def __init__(self, configs: dict[str, dict[str, Any]] | None = None):
        self._configs = dict(configs or {})

    def path_for(self, project: str) -> Path:
        return Path("<memory>") / f"{project}.yaml"

    def peek_config(self, project: str) -> dict[str, Any]:
        return dict(self._configs.get(project) or {})

    def load_or_init(
        self, project: str, schema: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # Match the historical behavior: hand back exactly what the
        # caller passed in. Schema defaults are applied by the loader
        # via _apply_defaults, not here.
        return dict(self._configs.get(project) or {})

    def write(self, project: str, config: dict[str, Any]) -> Path:
        self._configs[project] = dict(config)
        return self.path_for(project)


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}
