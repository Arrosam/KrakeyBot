"""Stale-plugin-config detection + deletion.

A "stale" plugin config = a directory under
``workspace/plugins/<name>/`` where:

  * the folder is NOT itself a workspace plugin (no ``meta.yaml``), AND
  * the name is not in the live plugin catalogue
    (``list_available_plugins()``).

These are leftovers from plugins that have been removed / renamed
since the operator last edited them via the dashboard. They sit in
the workspace dir indefinitely (gitignored) and silently confuse the
"per-plugin config" UI: a list of fields with no plugin to bind to.

Tests cover:
  * the HTTP wrapper (route → service);
  * the adapter's filesystem walk + safety rules around the deletion
    call (no path traversal, no deleting active plugins, no deleting
    workspace plugins that just have a broken meta.yaml).

Run from repo root:

    pytest krakey/plugins/dashboard
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from krakey.plugins.dashboard.app_factory import create_app
from krakey.plugins.dashboard.services.adapters import (
    RuntimePluginsService,
)


# ---------------------------------------------------------------------
# HTTP wrapper — routes/plugins.py + Protocol surface
# ---------------------------------------------------------------------


class _FakePluginsService:
    """Stand-in PluginsService for HTTP tests. Records every method
    call's args. Other Protocol methods are no-ops the routes under
    test never reach."""

    def __init__(self, stale=None, delete_result=None, raises=None):
        self._stale = list(stale or [])
        self._delete_result = delete_result
        self._raises = raises
        self.delete_calls: list[str] = []

    # ---- methods exercised here ----

    def find_stale_configs(self):
        return list(self._stale)

    def delete_stale_config(self, name):
        self.delete_calls.append(name)
        if self._raises is not None:
            raise self._raises
        return self._delete_result or {
            "name": name,
            "path": f"workspace/plugins/{name}",
            "deleted": True,
        }

    # ---- Protocol no-ops (routes never call these in stale tests) ----

    def report(self):
        return {"tools": [], "channels": []}

    def update_config(self, project, body):
        return {"project": project, "path": "", "config": {}}

    def deps_status(self):
        return {"pending": False, "plugins": {}, "state": {}}

    def install(self, body):
        return {"rc": 0, "stdout": "", "stderr": ""}

    async def hot_reload(self):
        return {
            "reloaded": [], "added": [], "removed": [],
            "skipped": [], "errors": [],
        }


def _client(svc):
    app = create_app(runtime=None, plugins_service=svc)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(
        transport=transport, base_url="http://test",
    )


async def test_list_endpoint_returns_records():
    svc = _FakePluginsService(stale=[
        {
            "name": "hypothalamus",
            "path": "workspace/plugins/hypothalamus",
            "has_config": True,
        },
    ])
    async with _client(svc) as c:
        r = await c.get("/api/plugins/stale_configs")
    assert r.status_code == 200
    body = r.json()
    assert body["stale"][0]["name"] == "hypothalamus"
    assert body["stale"][0]["has_config"] is True


async def test_list_endpoint_returns_empty_when_none():
    svc = _FakePluginsService(stale=[])
    async with _client(svc) as c:
        r = await c.get("/api/plugins/stale_configs")
    assert r.status_code == 200
    assert r.json() == {"stale": []}


async def test_delete_endpoint_invokes_service_with_name():
    svc = _FakePluginsService()
    async with _client(svc) as c:
        r = await c.post(
            "/api/plugins/stale_configs/delete",
            json={"name": "hypothalamus"},
        )
    assert r.status_code == 200
    assert svc.delete_calls == ["hypothalamus"]
    body = r.json()
    assert body["deleted"] is True
    assert body["name"] == "hypothalamus"


@pytest.mark.parametrize("payload", [
    {},                       # missing
    {"name": ""},             # empty
    {"name": None},           # null
    {"name": "  "},           # whitespace
])
async def test_delete_endpoint_400_on_invalid_name(payload):
    svc = _FakePluginsService()
    async with _client(svc) as c:
        r = await c.post(
            "/api/plugins/stale_configs/delete", json=payload,
        )
    assert r.status_code == 400
    # Service was never invoked.
    assert svc.delete_calls == []


async def test_delete_endpoint_400_on_value_error_from_service():
    svc = _FakePluginsService(
        raises=ValueError("plugin still in catalogue"),
    )
    async with _client(svc) as c:
        r = await c.post(
            "/api/plugins/stale_configs/delete",
            json={"name": "active"},
        )
    assert r.status_code == 400
    assert "catalogue" in r.json()["detail"]


async def test_delete_endpoint_404_on_lookup_error_from_service():
    svc = _FakePluginsService(raises=LookupError("no such folder"))
    async with _client(svc) as c:
        r = await c.post(
            "/api/plugins/stale_configs/delete",
            json={"name": "ghost"},
        )
    assert r.status_code == 404


# ---------------------------------------------------------------------
# Adapter — RuntimePluginsService.find_stale_configs / delete_stale_config
# ---------------------------------------------------------------------


_CATALOGUE_PATCH = (
    "krakey.plugins.dashboard.services.adapters.list_available_plugins"
)


def _write(path: Path, content: str = "max_results: 10\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class _StubMeta:
    """Stand-in for PluginMetadata — adapter only checks dict
    membership by name."""


def test_adapter_finds_stale_config_dirs(tmp_path, monkeypatch):
    plugins_root = tmp_path / "workspace" / "plugins"
    _write(plugins_root / "ghost_plugin" / "config.yaml")
    monkeypatch.setattr(_CATALOGUE_PATCH, lambda: {})

    svc = RuntimePluginsService(
        runtime=None, plugin_configs_root=plugins_root,
    )
    stale = svc.find_stale_configs()

    names = sorted(s["name"] for s in stale)
    assert names == ["ghost_plugin"]
    assert stale[0]["has_config"] is True


def test_adapter_skips_active_plugins(tmp_path, monkeypatch):
    plugins_root = tmp_path / "workspace" / "plugins"
    _write(plugins_root / "active" / "config.yaml")
    _write(plugins_root / "stale_one" / "config.yaml")
    monkeypatch.setattr(
        _CATALOGUE_PATCH, lambda: {"active": _StubMeta()},
    )

    svc = RuntimePluginsService(
        runtime=None, plugin_configs_root=plugins_root,
    )
    names = sorted(s["name"] for s in svc.find_stale_configs())
    assert names == ["stale_one"]


def test_adapter_skips_workspace_plugins_with_meta(tmp_path, monkeypatch):
    """A folder with meta.yaml is potentially a workspace plugin
    whose meta failed to parse. NEVER auto-delete."""
    plugins_root = tmp_path / "workspace" / "plugins"
    _write(plugins_root / "broken_workspace" / "config.yaml")
    _write(
        plugins_root / "broken_workspace" / "meta.yaml",
        content="malformed: [\n",
    )
    monkeypatch.setattr(_CATALOGUE_PATCH, lambda: {})

    svc = RuntimePluginsService(
        runtime=None, plugin_configs_root=plugins_root,
    )
    assert svc.find_stale_configs() == []


def test_adapter_returns_empty_when_root_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(_CATALOGUE_PATCH, lambda: {})
    svc = RuntimePluginsService(
        runtime=None,
        plugin_configs_root=tmp_path / "does" / "not" / "exist",
    )
    assert svc.find_stale_configs() == []


def test_adapter_lists_no_config_folder_too(tmp_path, monkeypatch):
    """A leftover empty folder still gets reported — the operator may
    want it gone even without a config.yaml. ``has_config`` is False
    so the UI can label it accordingly."""
    plugins_root = tmp_path / "workspace" / "plugins"
    (plugins_root / "empty_ghost").mkdir(parents=True)
    monkeypatch.setattr(_CATALOGUE_PATCH, lambda: {})

    svc = RuntimePluginsService(
        runtime=None, plugin_configs_root=plugins_root,
    )
    stale = svc.find_stale_configs()
    assert [s["name"] for s in stale] == ["empty_ghost"]
    assert stale[0]["has_config"] is False


def test_adapter_delete_removes_directory(tmp_path, monkeypatch):
    plugins_root = tmp_path / "workspace" / "plugins"
    _write(plugins_root / "ghost" / "config.yaml")
    _write(plugins_root / "ghost" / "subfile.txt", content="x")
    monkeypatch.setattr(_CATALOGUE_PATCH, lambda: {})

    svc = RuntimePluginsService(
        runtime=None, plugin_configs_root=plugins_root,
    )
    out = svc.delete_stale_config("ghost")

    assert out["deleted"] is True
    assert out["name"] == "ghost"
    assert not (plugins_root / "ghost").exists()


@pytest.mark.parametrize("bad_name", [
    "..", "../foo", "foo/bar", "foo\\bar", "", "  ",
    "/abs", "name.with.dot", "x" * 200,
])
def test_adapter_delete_rejects_bad_names(
    tmp_path, monkeypatch, bad_name,
):
    plugins_root = tmp_path / "workspace" / "plugins"
    plugins_root.mkdir(parents=True)
    monkeypatch.setattr(_CATALOGUE_PATCH, lambda: {})

    svc = RuntimePluginsService(
        runtime=None, plugin_configs_root=plugins_root,
    )
    with pytest.raises(ValueError):
        svc.delete_stale_config(bad_name)


def test_adapter_delete_rejects_active_plugin(tmp_path, monkeypatch):
    plugins_root = tmp_path / "workspace" / "plugins"
    _write(plugins_root / "active" / "config.yaml")
    monkeypatch.setattr(
        _CATALOGUE_PATCH, lambda: {"active": _StubMeta()},
    )

    svc = RuntimePluginsService(
        runtime=None, plugin_configs_root=plugins_root,
    )
    with pytest.raises(ValueError):
        svc.delete_stale_config("active")
    # The folder must still be on disk — refusal must be safe.
    assert (plugins_root / "active" / "config.yaml").exists()


def test_adapter_delete_rejects_workspace_plugin_with_meta(
    tmp_path, monkeypatch,
):
    plugins_root = tmp_path / "workspace" / "plugins"
    _write(plugins_root / "wp" / "meta.yaml", content="name: wp\n")
    _write(plugins_root / "wp" / "config.yaml")
    monkeypatch.setattr(_CATALOGUE_PATCH, lambda: {})

    svc = RuntimePluginsService(
        runtime=None, plugin_configs_root=plugins_root,
    )
    with pytest.raises(ValueError):
        svc.delete_stale_config("wp")
    assert (plugins_root / "wp" / "meta.yaml").exists()


def test_adapter_delete_404_when_folder_missing(tmp_path, monkeypatch):
    plugins_root = tmp_path / "workspace" / "plugins"
    plugins_root.mkdir(parents=True)
    monkeypatch.setattr(_CATALOGUE_PATCH, lambda: {})

    svc = RuntimePluginsService(
        runtime=None, plugin_configs_root=plugins_root,
    )
    with pytest.raises(LookupError):
        svc.delete_stale_config("nope")
