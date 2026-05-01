"""Phase 3 (config overhaul): /api/plugins/... round-trips.

Covers:
  - GET /api/plugins returns values + enabled per component
  - POST /api/plugins/<project>/config persists to per-plugin file
  - Shapes errors (400 / 503) correctly
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import yaml

from krakey.plugins.dashboard.app_factory import create_app
from krakey.plugins.dashboard.services.plugins import PluginsService


class _FakePluginsService:
    """Stand-in PluginsService backed by an in-memory report + file
    store. Mirrors RuntimePluginsService but without needing a Runtime."""

    def __init__(self, report: dict, cfg_root: Path):
        self._report = report
        self._cfg_root = cfg_root

    def report(self):
        return self._report

    def update_config(self, project, body):
        if not project:
            raise ValueError("project required")
        enabled = bool(body.get("enabled", False))
        values = dict(body.get("values") or {})
        values.pop("enabled", None)
        final = {"enabled": enabled, **values}
        self._cfg_root.mkdir(parents=True, exist_ok=True)
        path = self._cfg_root / f"{project}.yaml"
        path.write_text(yaml.safe_dump(final, sort_keys=False),
                          encoding="utf-8")
        return {"project": project, "path": str(path), "config": final}


def _client(plugins_service):
    app = create_app(runtime=None, plugins_service=plugins_service)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ---------------- GET /api/plugins ----------------


async def test_report_returns_schema_and_values(tmp_path):
    report = {
        "tools": [
            {
                "name": "search", "kind": "tool", "source": "builtin",
                "path": "", "project": "search",
                "description": "web search",
                "config_schema": [
                    {"field": "max_results", "type": "number",
                     "default": 5, "help": ""},
                ],
                "enabled": True,
                "values": {"max_results": 8},
                "loaded": True, "error": None,
            },
        ],
        "channels": [],
    }
    svc = _FakePluginsService(report, tmp_path / "cfgs")
    async with _client(svc) as c:
        r = await c.get("/api/plugins")
    assert r.status_code == 200
    body = r.json()
    assert body["tools"][0]["values"] == {"max_results": 8}
    assert body["tools"][0]["enabled"] is True
    assert body["tools"][0]["config_schema"][0]["field"] == "max_results"


async def test_report_503_when_no_runtime():
    async with _client(plugins_service=None) as c:
        r = await c.get("/api/plugins")
    assert r.status_code == 503


# ---------------- POST /api/plugins/<project>/config ----------------


async def test_update_config_persists_to_file(tmp_path):
    svc = _FakePluginsService({"tools": [], "channels": []},
                                  tmp_path / "cfgs")
    async with _client(svc) as c:
        r = await c.post("/api/plugins/search/config",
                            json={"enabled": True,
                                   "values": {"max_results": 12}})
    assert r.status_code == 200
    path = tmp_path / "cfgs" / "search.yaml"
    assert path.exists()
    on_disk = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert on_disk == {"enabled": True, "max_results": 12}


async def test_update_config_strips_enabled_from_values(tmp_path):
    """Loader owns `enabled`; clients mustn't double-write it inside
    values."""
    svc = _FakePluginsService({"tools": [], "channels": []},
                                  tmp_path / "cfgs")
    async with _client(svc) as c:
        r = await c.post(
            "/api/plugins/search/config",
            json={"enabled": True,
                   "values": {"enabled": False, "max_results": 5}},
        )
    assert r.status_code == 200
    on_disk = yaml.safe_load(
        (tmp_path / "cfgs" / "search.yaml").read_text(encoding="utf-8")
    )
    # Top-level `enabled: true` wins; the nested one is silently dropped.
    assert on_disk["enabled"] is True
    assert on_disk["max_results"] == 5


async def test_update_config_succeeds_without_runtime(tmp_path, monkeypatch):
    """Plugin-config writes go straight to the on-disk file via the
    dashboard's own FilePluginConfigStore — runtime is not on the
    write path. Expect a successful save even when runtime=None."""
    monkeypatch.chdir(tmp_path)
    async with _client(plugins_service=None) as c:
        r = await c.post(
            "/api/plugins/search/config",
            json={"enabled": True, "values": {"max_results": 9}},
        )
    assert r.status_code == 200
    on_disk = yaml.safe_load(
        (tmp_path / "workspace" / "plugins" / "search"
         / "config.yaml").read_text(encoding="utf-8")
    )
    # `enabled` is dropped (central config.yaml owns enable/disable).
    assert on_disk == {"max_results": 9}
