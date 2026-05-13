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

import re
import shutil
from pathlib import Path
from typing import Any, Callable, Protocol

import yaml as _yaml

from krakey.models.config_backup import backup_config
from krakey.plugin_system.catalogue import list_available_plugins


# ---------------- memory ----------------


class _MemoryRuntime(Protocol):
    """Narrow shape this module needs from Runtime \u2014 declared here so
    the adapter doesn't depend on the full Runtime class. Lets tests
    substitute a stand-in with just ``memory``."""
    memory: Any


class RuntimeMemoryService:
    """Adapts Runtime.memory to MemoryService.

    After the Engine refactor (2026-05) the runtime exposes a single
    MemoryEngine that fulfills both the GM + KB-registry surfaces, so
    this adapter routes every call through ``rt.memory``.
    """

    def __init__(self, runtime: _MemoryRuntime | None):
        self._rt = runtime

    def _require(self) -> _MemoryRuntime:
        if self._rt is None or not hasattr(self._rt, "memory"):
            raise RuntimeError("runtime not available")
        return self._rt

    async def list_gm_nodes(
        self, *, category: str | None, limit: int,
    ) -> list[dict[str, Any]]:
        rt = self._require()
        nodes = await rt.memory.list_nodes(category=category, limit=limit)
        return [_serialize_node(n) for n in nodes]

    async def list_gm_edges(
        self, *, limit: int,
    ) -> list[dict[str, Any]]:
        rt = self._require()
        return await rt.memory.list_edges_named(limit=limit)

    async def gm_stats(self) -> dict[str, Any]:
        rt = self._require()
        total_nodes = await rt.memory.count_nodes()
        total_edges = await rt.memory.count_edges()
        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "by_category": await rt.memory.counts_by_category(),
            "by_source": await rt.memory.counts_by_source(),
        }

    async def list_kbs(self) -> list[dict[str, Any]]:
        rt = self._require()
        return await rt.memory.list_kbs()

    async def kb_entries(
        self, *, kb_id: str, limit: int,
    ) -> list[dict[str, Any]]:
        rt = self._require()
        try:
            kb = await rt.memory.open_kb(kb_id)
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
        from krakey.plugin_system.config import FilePluginConfigStore
        self._rt = runtime
        self._store = FilePluginConfigStore(root=plugin_configs_root)

    def report(self) -> dict[str, Any]:
        if self._rt is None:
            raise RuntimeError("runtime not available")
        if not hasattr(self._rt, "loaded_plugin_report"):
            return {"tools": [], "channels": [], "modifiers": []}
        report = self._rt.loaded_plugin_report()
        # Decorate every kind the runtime emits — tools, channels,
        # modifiers — with the per-plugin config + UI placeholders so
        # the JS catalog renderer has a uniform shape. Modifiers were
        # historically excluded from this enrichment, which made
        # modifier-only plugin rows in the dashboard panel show as
        # "not loaded" even when the modifier was registered.
        for kind in ("tools", "channels", "modifiers"):
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

    # ---- deps install status / install dispatch ----
    # Both delegate to the DefaultInstallService utility class
    # imported from ``krakey.install``. Install is a pre-startup
    # concern — the runtime has no reference to it.

    def _install_service(self):
        from krakey.install import DefaultInstallService
        return DefaultInstallService()

    def deps_status(self) -> dict[str, Any]:
        return self._install_service().deps_status()

    async def hot_reload(self) -> dict[str, Any]:
        """Re-parse the central config.yaml and ask the runtime to
        hot-add any newly-enabled plugins. Returns the runtime's
        report verbatim."""
        if self._rt is None:
            raise RuntimeError("runtime not available")
        # Re-read config.yaml from disk so the latest edit is what
        # we diff against (the runtime's in-memory ``config.plugins``
        # was set at startup and may be stale). The plugin enable/
        # disable surface is the dashboard's settings page; users
        # save then click "Apply changes".
        if hasattr(self._rt, "config"):
            target_names = list(self._rt.config.plugins or [])
        else:
            target_names = []
        if not hasattr(self._rt, "hot_reload_plugins"):
            raise RuntimeError(
                "runtime does not support hot reload "
                "(older Runtime; restart required)",
            )
        return await self._rt.hot_reload_plugins(target_names)

    def install(self, body: dict[str, Any]) -> dict[str, Any]:
        """Delegate to the runtime's InstallService. Truncates
        output so a chatty pip session doesn't blow up the JSON
        response."""
        upgrade = bool(body.get("upgrade", False))
        result = self._install_service().install(
            upgrade=upgrade, dry_run=False,
        )
        return {
            "rc":     int(result.rc),
            "stdout": _truncate(result.stdout, 8000),
            "stderr": _truncate(result.stderr, 8000),
        }

    # ---- stale-config detection + deletion ----
    # The plugin-configs root is the same one ``update_config``
    # writes to, so we walk it directly here. Catalogue check uses
    # the same ``list_available_plugins`` source the dashboard
    # already trusts elsewhere (settings route's
    # ``/api/modifiers/available`` endpoint) — keeping ONE source
    # of truth for "what plugins exist" so a folder can't appear
    # stale here but live in the install banner.

    def find_stale_configs(self) -> list[dict[str, Any]]:
        root = self._store._root  # noqa: SLF001 — same package
        if not root.exists():
            return []
        catalogue = list_available_plugins()
        out: list[dict[str, Any]] = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            name = child.name
            if name in catalogue:
                continue
            # A folder with its own meta.yaml is potentially a
            # workspace plugin whose manifest failed to parse;
            # the catalogue scanner skips bad meta but the folder
            # is still a plugin. Never auto-delete.
            if (child / "meta.yaml").exists():
                continue
            out.append({
                "name": name,
                "path": str(child),
                "has_config": (child / "config.yaml").exists(),
            })
        return out

    def delete_stale_config(self, name: str) -> dict[str, Any]:
        if not _is_safe_plugin_name(name):
            raise ValueError(
                f"refusing to delete plugin-config folder with "
                f"unsafe name {name!r}: must match "
                f"[A-Za-z0-9_-]{{1,64}}"
            )
        root = self._store._root  # noqa: SLF001
        target = root / name
        if not target.exists() or not target.is_dir():
            raise LookupError(
                f"plugin-config folder {name!r} not found under {root}",
            )
        if (target / "meta.yaml").exists():
            raise ValueError(
                f"refusing to delete {name!r}: folder has a meta.yaml "
                f"(it's a workspace plugin, not a stale config)",
            )
        if name in list_available_plugins():
            raise ValueError(
                f"refusing to delete {name!r}: plugin is still in "
                f"the live catalogue",
            )
        shutil.rmtree(target)
        return {
            "name": name,
            "path": str(target),
            "deleted": True,
        }


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n...[truncated, total {len(s)} chars]"


# Plugin name policy — matches the on-disk folder convention used
# everywhere else (lower_snake_case + occasional dashes). The strict
# regex is the load-bearing safety boundary for ``delete_stale_config``;
# no character that could escape the plugin-configs root (``.``, ``/``,
# ``\``, whitespace) is permitted.
_SAFE_PLUGIN_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _is_safe_plugin_name(name: Any) -> bool:
    return isinstance(name, str) and bool(_SAFE_PLUGIN_NAME.match(name))


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
