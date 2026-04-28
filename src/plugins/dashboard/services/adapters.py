"""Concrete Service adapters \u2014 Runtime \u2192 Protocol bridges.

Each adapter is a thin wrapper that reshapes Runtime's existing
methods into the service Protocol's narrower surface. Routes depend
on the Protocol; ``app_factory.create_app`` instantiates the adapter
inline when no test override was passed for that service slot. Tests
substitute hand-built fakes via the ``*_service`` kwargs.

Every adapter raises its own failure mode (usually RuntimeError or
ValueError); the routes translate those into HTTPException.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

import yaml as _yaml

from src.models.config_backup import backup_config


# ---------------- memory ----------------


class _MemoryRuntime(Protocol):
    """Narrow shape this module needs from Runtime \u2014 declared here so
    the adapter doesn't depend on the full Runtime class. Lets tests
    substitute a stand-in with just ``gm`` and ``kb_registry``."""
    gm: Any
    kb_registry: Any


class RuntimeMemoryService:
    """Adapts Runtime.gm + Runtime.kb_registry to MemoryService."""

    def __init__(self, runtime: _MemoryRuntime | None):
        self._rt = runtime

    def _require(self) -> _MemoryRuntime:
        if self._rt is None or not hasattr(self._rt, "gm"):
            raise RuntimeError("runtime not available")
        return self._rt

    async def list_gm_nodes(
        self, *, category: str | None, limit: int,
    ) -> list[dict[str, Any]]:
        rt = self._require()
        nodes = await rt.gm.list_nodes(category=category, limit=limit)
        return [_serialize_node(n) for n in nodes]

    async def list_gm_edges(
        self, *, limit: int,
    ) -> list[dict[str, Any]]:
        rt = self._require()
        return await rt.gm.list_edges_named(limit=limit)

    async def gm_stats(self) -> dict[str, Any]:
        rt = self._require()
        total_nodes = await rt.gm.count_nodes()
        total_edges = await rt.gm.count_edges()
        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "by_category": await rt.gm.counts_by_category(),
            "by_source": await rt.gm.counts_by_source(),
        }

    async def list_kbs(self) -> list[dict[str, Any]]:
        rt = self._require()
        return await rt.kb_registry.list_kbs()

    async def kb_entries(
        self, *, kb_id: str, limit: int,
    ) -> list[dict[str, Any]]:
        rt = self._require()
        try:
            kb = await rt.kb_registry.open_kb(kb_id)
        except KeyError:
            raise LookupError(f"KB {kb_id!r} not found")
        return await kb.list_active_entries(limit=limit)


def _serialize_node(n: dict[str, Any]) -> dict[str, Any]:
    """Trim raw embedding from API response; keep its presence as a flag."""
    out = {k: v for k, v in n.items() if k != "embedding"}
    out["has_embedding"] = n.get("embedding") is not None
    return out


# ---------------- prompts ----------------


class RuntimePromptsService:
    def __init__(self, runtime: Any):
        self._rt = runtime

    def recent(self, *, limit: int) -> list[dict[str, Any]]:
        if self._rt is None:
            raise RuntimeError("runtime not available")
        if not hasattr(self._rt, "recent_prompts"):
            return []
        return self._rt.recent_prompts(limit=limit)


# ---------------- plugins ----------------


class RuntimePluginsService:
    """Combines runtime observation (which plugins are loaded) with
    direct file I/O on per-plugin config files.

    Plugin config WRITES never go through Runtime — the dashboard owns
    a ``FilePluginConfigStore`` and edits files directly. Runtime is
    the source of truth ONLY for "is this component currently
    registered?" — the rest (descriptions, schemas, current values)
    comes from the on-disk plugin folders.
    """

    def __init__(
        self,
        runtime: Any,
        plugin_configs_root: Path | str = "workspace/plugins",
    ):
        from src.plugin_system.config import FilePluginConfigStore
        self._rt = runtime
        self._store = FilePluginConfigStore(root=plugin_configs_root)

    def report(self) -> dict[str, Any]:
        if self._rt is None:
            raise RuntimeError("runtime not available")
        if not hasattr(self._rt, "loaded_plugin_report"):
            return {"tentacles": [], "sensories": []}
        report = self._rt.loaded_plugin_report()
        for kind in ("tentacles", "sensories"):
            for entry in report.get(kind, []):
                project = entry.get("project") or ""
                entry["values"] = (
                    self._store.read(project) if project else {}
                )
                entry.setdefault("description", "")
                entry.setdefault("config_schema", [])
                entry.setdefault("path", "")
                entry.setdefault("enabled", True)
        return report

    def update_config(
        self, project: str, body: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist a dashboard edit by writing the per-plugin
        ``<root>/<project>/config.yaml`` directly. Runtime is not
        involved — the dashboard owns this file."""
        if not project or not isinstance(project, str):
            raise ValueError("project name required")
        values = dict(body.get("values") or {})
        values.pop("enabled", None)  # central config.yaml owns enable/disable
        path = self._store.write(project, values)
        return {"project": project, "path": str(path), "config": values}


# ---------------- config ----------------


class FileConfigService:
    """Reads/writes a single config.yaml file. Backup policy lives here."""

    def __init__(
        self,
        config_path: Path | None,
        on_restart: Callable[[], None] | None = None,
    ):
        self._path = Path(config_path) if config_path is not None else None
        self._on_restart = on_restart

    @property
    def path(self) -> Path | None:
        return self._path

    def read(self) -> tuple[str, Any]:
        if self._path is None:
            raise RuntimeError("config_path not provided")
        if not self._path.exists():
            raise FileNotFoundError(f"config not found: {self._path}")
        raw = self._path.read_text(encoding="utf-8")
        try:
            parsed = _yaml.safe_load(raw)
        except _yaml.YAMLError:
            parsed = None
        return raw, parsed

    def write(
        self, *, raw: str | None, parsed: Any, backup_dir: str,
    ) -> Path | None:
        if self._path is None:
            raise RuntimeError("config_path not provided")
        if parsed is not None:
            try:
                new_raw = _yaml.safe_dump(parsed, allow_unicode=True,
                                           sort_keys=False)
            except _yaml.YAMLError as e:
                raise ValueError(f"cannot serialize: {e}")
        else:
            if raw is None or not isinstance(raw, str):
                raise ValueError("missing 'parsed' or 'raw' field")
            try:
                _yaml.safe_load(raw)
            except _yaml.YAMLError as e:
                raise ValueError(f"invalid YAML: {e}")
            new_raw = raw
        backup_path = backup_config(self._path, backup_dir)
        self._path.write_text(new_raw, encoding="utf-8")
        return backup_path

    def restart(self) -> None:
        if self._on_restart is None:
            raise RuntimeError("restart not wired")
        self._on_restart()
