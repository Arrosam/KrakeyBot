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

    def __init__(
        self,
        report: dict,
        cfg_root: Path,
        deps_status_payload: dict | None = None,
        install_payload: dict | None = None,
    ):
        self._report = report
        self._cfg_root = cfg_root
        self._deps_status = deps_status_payload or {
            "pending": False, "plugins": {}, "state": {},
        }
        self._install_payload = install_payload or {
            "rc": 0, "stdout": "", "stderr": "",
        }
        self.install_calls: list[dict] = []

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

    def deps_status(self):
        return self._deps_status

    def install(self, body):
        self.install_calls.append(dict(body or {}))
        return self._install_payload


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


# =====================================================================
# /api/plugins/deps_status — drives the plugin-list "needs install"
# badges
# =====================================================================


async def test_deps_status_passes_through_payload(tmp_path):
    payload = {
        "pending": True,
        "plugins": {
            "browser_exec": {
                "dependencies": ["playwright>=1.40"],
                "post_install": [
                    {"args": ["{python}", "-m", "playwright", "install",
                              "chromium"],
                     "description": "browser binary",
                     "optional": False},
                ],
                "installed": False,
                "satisfied": False,
            },
            "cli_exec": {
                "dependencies": [],
                "post_install": [],
                "installed": False,
                "satisfied": True,  # no deps → trivially satisfied
            },
        },
        "state": {
            "installed_at": None,
            "deps_hash": None,
            "live_hash": "abc123",
        },
    }
    svc = _FakePluginsService(
        {"tools": [], "channels": []},
        tmp_path / "cfgs",
        deps_status_payload=payload,
    )
    async with _client(svc) as c:
        r = await c.get("/api/plugins/deps_status")
    assert r.status_code == 200
    body = r.json()
    assert body["pending"] is True
    assert body["plugins"]["browser_exec"]["satisfied"] is False
    assert body["plugins"]["cli_exec"]["satisfied"] is True


# =====================================================================
# /api/plugins/install — one-click install button backend
# =====================================================================


async def test_install_endpoint_invokes_service_and_returns_rc(tmp_path):
    svc = _FakePluginsService(
        {"tools": [], "channels": []},
        tmp_path / "cfgs",
        install_payload={
            "rc": 0,
            "stdout": "Successfully installed playwright-1.40.0\n",
            "stderr": "",
        },
    )
    async with _client(svc) as c:
        r = await c.post(
            "/api/plugins/install",
            json={"upgrade": False},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["rc"] == 0
    assert "playwright" in body["stdout"]
    assert svc.install_calls == [{"upgrade": False}]


async def test_install_endpoint_propagates_pip_failure(tmp_path):
    svc = _FakePluginsService(
        {"tools": [], "channels": []},
        tmp_path / "cfgs",
        install_payload={
            "rc": 1,
            "stdout": "",
            "stderr": "ERROR: Could not find a version that satisfies "
                      "the requirement playwright>=999\n",
        },
    )
    async with _client(svc) as c:
        r = await c.post("/api/plugins/install", json={})
    # HTTP success — the failure is in the body's rc field, not
    # the HTTP status. (rc != 0 isn't a server error; the
    # endpoint did its job and reports what pip said.)
    assert r.status_code == 200
    body = r.json()
    assert body["rc"] == 1
    assert "ERROR" in body["stderr"]


async def test_install_endpoint_threads_upgrade_flag(tmp_path):
    svc = _FakePluginsService(
        {"tools": [], "channels": []}, tmp_path / "cfgs",
    )
    async with _client(svc) as c:
        await c.post("/api/plugins/install", json={"upgrade": True})
    assert svc.install_calls == [{"upgrade": True}]


async def test_install_endpoint_accepts_empty_body(tmp_path):
    """Operator hits "Install" without any payload — should still
    work, defaulting upgrade=false."""
    svc = _FakePluginsService(
        {"tools": [], "channels": []}, tmp_path / "cfgs",
    )
    async with _client(svc) as c:
        r = await c.post("/api/plugins/install")
    assert r.status_code == 200
    assert svc.install_calls == [{}]
