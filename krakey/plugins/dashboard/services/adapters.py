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

from krakey.models.config_backup import backup_config


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
        from krakey.plugin_system.config import FilePluginConfigStore
        self._rt = runtime
        self._store = FilePluginConfigStore(root=plugin_configs_root)

    def report(self) -> dict[str, Any]:
        if self._rt is None:
            raise RuntimeError("runtime not available")
        if not hasattr(self._rt, "loaded_plugin_report"):
            return {"tools": [], "channels": []}
        report = self._rt.loaded_plugin_report()
        for kind in ("tools", "channels"):
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

    def deps_status(self) -> dict[str, Any]:
        from krakey.cli import install as install_mod

        plugin_deps = install_mod.collect_plugin_dependencies()
        plugin_post = install_mod.collect_plugin_post_install()
        state = install_mod.read_install_state() or {}
        installed_set = set(state.get("installed") or [])
        live_hash = install_mod.deps_hash(plugin_deps, plugin_post)
        recorded_hash = state.get("deps_hash")
        # A plugin counts as "satisfied" when:
        #   - it appears in the recorded installed list, AND
        #   - the recorded global hash matches the live one (so a
        #     plugin whose deps were installed but THEN edited
        #     correctly shows up as un-satisfied again).
        # Plugins with no third-party deps are trivially satisfied.
        plugins_out: dict[str, dict[str, Any]] = {}
        all_names = sorted(set(plugin_deps) | set(plugin_post))
        any_pending = False
        for name in all_names:
            deps = plugin_deps.get(name) or []
            post = plugin_post.get(name) or []
            if not deps and not post:
                satisfied = True
            else:
                satisfied = (
                    name in installed_set
                    and recorded_hash == live_hash
                )
            if not satisfied:
                any_pending = True
            plugins_out[name] = {
                "dependencies": list(deps),
                "post_install": list(post),
                "installed":    name in installed_set,
                "satisfied":    satisfied,
            }
        return {
            "pending":  any_pending or recorded_hash != live_hash,
            "plugins":  plugins_out,
            "state": {
                "installed_at": state.get("installed_at"),
                "deps_hash":    recorded_hash,
                "live_hash":    live_hash,
            },
        }

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
        """Run ``krakey install`` programmatically with
        ``upgrade`` flag from body. Captures stdout / stderr and
        returns them along with rc. Truncates output so a chatty
        pip session doesn't blow up the JSON response."""
        import argparse
        import contextlib
        import io

        from krakey.cli import install as install_mod

        upgrade = bool(body.get("upgrade", False))
        out_buf = io.StringIO()
        err_buf = io.StringIO()

        try:
            with (
                contextlib.redirect_stdout(out_buf),
                contextlib.redirect_stderr(err_buf),
            ):
                rc = install_mod.install(
                    argparse.Namespace(
                        dry_run=False, upgrade=upgrade,
                    ),
                )
        except Exception as e:  # noqa: BLE001
            return {
                "rc": -1,
                "stdout": _truncate(out_buf.getvalue(), 8000),
                "stderr": _truncate(err_buf.getvalue(), 8000)
                          + f"\n[crash] {type(e).__name__}: {e}",
            }

        return {
            "rc": int(rc),
            "stdout": _truncate(out_buf.getvalue(), 8000),
            "stderr": _truncate(err_buf.getvalue(), 8000),
        }


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n...[truncated, total {len(s)} chars]"


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
