"""Concrete Service adapters \u2014 Runtime \u2192 Protocol bridges.

Each adapter is a thin wrapper that reshapes Runtime's existing
methods into the service Protocol's narrower surface. Routes depend
on the Protocol; `app_factory.build_services(runtime)` picks the
adapter here. Tests can substitute hand-built fakes.

Every adapter raises its own failure mode (usually RuntimeError or
ValueError); the routes translate those into HTTPException.
"""
from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml as _yaml

from src.models.config_backup import backup_config


# ---------------- memory ----------------


class RuntimeMemoryService:
    """Adapts Runtime.gm + Runtime.kb_registry to MemoryService."""

    def __init__(self, runtime: Any):
        self._rt = runtime

    def _require(self) -> Any:
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
        db = rt.gm._require()  # noqa: SLF001
        async with db.execute(
            "SELECT na.name AS source, e.predicate AS predicate, "
            "nb.name AS target FROM gm_edges e "
            "JOIN gm_nodes na ON na.id=e.node_a "
            "JOIN gm_nodes nb ON nb.id=e.node_b "
            "ORDER BY e.id ASC LIMIT ?", (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [{"source": r["source"], "target": r["target"],
                  "predicate": r["predicate"]} for r in rows]

    async def gm_stats(self) -> dict[str, Any]:
        rt = self._require()
        total_nodes = await rt.gm.count_nodes()
        total_edges = await rt.gm.count_edges()
        db = rt.gm._require()  # noqa: SLF001
        async with db.execute(
            "SELECT category, COUNT(*) FROM gm_nodes GROUP BY category"
        ) as cur:
            cat_rows = await cur.fetchall()
        async with db.execute(
            "SELECT source_type, COUNT(*) FROM gm_nodes GROUP BY source_type"
        ) as cur:
            src_rows = await cur.fetchall()
        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "by_category": {r[0]: r[1] for r in cat_rows},
            "by_source": {r[0]: r[1] for r in src_rows},
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
        db = kb._require()  # noqa: SLF001
        async with db.execute(
            "SELECT id, content, source, tags, importance, created_at "
            "FROM kb_entries WHERE is_active = 1 ORDER BY id DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        entries = []
        for r in rows:
            tags = _json.loads(r["tags"]) if r["tags"] else []
            entries.append({"id": r["id"], "content": r["content"],
                              "source": r["source"], "tags": tags,
                              "importance": r["importance"],
                              "created_at": r["created_at"]})
        return entries


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
    def __init__(self, runtime: Any):
        self._rt = runtime

    def report(self) -> dict[str, Any]:
        if self._rt is None:
            raise RuntimeError("runtime not available")
        if not hasattr(self._rt, "plugin_report"):
            return {"tentacles": [], "sensories": []}
        return self._rt.plugin_report()

    def update_config(
        self, project: str, body: dict[str, Any],
    ) -> dict[str, Any]:
        if self._rt is None:
            raise RuntimeError("runtime not available")
        if not hasattr(self._rt, "update_plugin_config"):
            raise RuntimeError("runtime does not support plugin config updates")
        return self._rt.update_plugin_config(project, body)


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
