"""Phase 3.F.6: Settings REST + config backup."""
import time
from pathlib import Path

import httpx
import pytest
import yaml

from src.dashboard.app_factory import create_app
from src.models.config_backup import (
    BACKUP_FILENAME_PREFIX, backup_config, list_backups,
)


# ---------------- backup helpers ----------------

def test_backup_creates_timestamped_file(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("hello: world\n", encoding="utf-8")
    backup_dir = tmp_path / "backups"

    out_path = backup_config(cfg, backup_dir)
    assert out_path.exists()
    assert out_path.parent == backup_dir
    assert out_path.name.startswith(BACKUP_FILENAME_PREFIX)
    assert out_path.read_text(encoding="utf-8") == "hello: world\n"


def test_backup_missing_source_returns_none(tmp_path):
    out = backup_config(tmp_path / "missing.yaml",
                          tmp_path / "backups")
    assert out is None


def test_list_backups_returns_newest_first(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("v: 1\n", encoding="utf-8")
    backups = tmp_path / "backups"
    p1 = backup_config(cfg, backups)
    time.sleep(1.1)  # filename timestamps are second-resolution
    cfg.write_text("v: 2\n", encoding="utf-8")
    p2 = backup_config(cfg, backups)

    listed = list_backups(backups)
    assert [b.name for b in listed][:2] == [p2.name, p1.name]


def test_backup_caps_retention_at_keep_last(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")
    backups = tmp_path / "backups"
    paths = []
    for i in range(5):
        time.sleep(1.05)
        cfg.write_text(f"x: {i}\n", encoding="utf-8")
        paths.append(backup_config(cfg, backups, keep_last=3))
    surviving = list_backups(backups)
    assert len(surviving) == 3


# ---------------- Settings REST ----------------

def _client(*, runtime=None, config_path: Path | None = None,
             on_restart=None):
    app = create_app(runtime=runtime, config_path=config_path,
                       on_restart=on_restart)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_get_settings_returns_raw_text(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("a: 1\nb: hello\n", encoding="utf-8")
    async with _client(config_path=p) as c:
        r = await c.get("/api/settings")
    assert r.status_code == 200
    assert r.json()["raw"] == "a: 1\nb: hello\n"


async def test_get_settings_503_when_no_path():
    async with _client() as c:
        r = await c.get("/api/settings")
    assert r.status_code == 503


async def test_post_settings_writes_and_backs_up(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("a: 1\n", encoding="utf-8")
    backups = tmp_path / "backups"

    async with _client(config_path=p) as c:
        r = await c.post("/api/settings",
                            json={"raw": "a: 2\nb: new\n",
                                   "backup_dir": str(backups)})
    assert r.status_code == 200
    assert p.read_text(encoding="utf-8") == "a: 2\nb: new\n"
    # backup of OLD content was made before overwrite
    bks = list(backups.glob("*.yaml"))
    assert len(bks) == 1
    assert bks[0].read_text(encoding="utf-8") == "a: 1\n"


async def test_post_settings_invalid_yaml_rejected(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("a: 1\n", encoding="utf-8")
    async with _client(config_path=p) as c:
        r = await c.post("/api/settings",
                            json={"raw": "a: : : invalid"})
    assert r.status_code == 400
    assert p.read_text(encoding="utf-8") == "a: 1\n"  # untouched


async def test_post_restart_invokes_callback():
    triggered = {"count": 0}

    def on_restart():
        triggered["count"] += 1

    async with _client(on_restart=on_restart) as c:
        r = await c.post("/api/restart")
    assert r.status_code == 200
    assert triggered["count"] == 1


async def test_post_restart_503_when_no_callback():
    async with _client() as c:
        r = await c.post("/api/restart")
    assert r.status_code == 503


# ---------------- Structured POST + upload ----------------

async def test_post_settings_accepts_structured_parsed(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("a: 1\n", encoding="utf-8")
    async with _client(config_path=p) as c:
        r = await c.post("/api/settings",
                            json={"parsed": {"a": 2, "nested": {"k": "v"}},
                                   "backup_dir": str(tmp_path / "bk")})
    assert r.status_code == 200
    written = p.read_text(encoding="utf-8")
    assert yaml.safe_load(written) == {"a": 2, "nested": {"k": "v"}}


async def test_post_settings_requires_parsed_or_raw(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("a: 1\n", encoding="utf-8")
    async with _client(config_path=p) as c:
        r = await c.post("/api/settings", json={})
    assert r.status_code == 400


# ---------------- /api/config/schema ----------------


async def test_config_schema_lists_llm_params(tmp_path):
    """The dashboard renders tag params from this endpoint. It must
    expose every field on LLMParams so a future field addition doesn't
    silently drop off the UI."""
    p = tmp_path / "config.yaml"
    p.write_text("a: 1\n", encoding="utf-8")
    async with _client(config_path=p) as c:
        r = await c.get("/api/config/schema")
    assert r.status_code == 200
    body = r.json()
    assert "llm_params" in body
    names = {e["field"] for e in body["llm_params"]}
    assert {"max_output_tokens", "max_input_tokens", "temperature",
            "reasoning_mode", "reasoning_budget_tokens",
            "timeout_seconds", "max_retries", "retry_on_status"} <= names
    # Old ambiguous name must not resurface in the schema; it lives
    # only as a YAML read-alias.
    assert "max_tokens" not in names
    # `llm_role_defaults` was removed in the tag-based refactor —
    # roles don't exist any more, defaults come from LLMParams itself.
    assert "llm_role_defaults" not in body


async def test_config_schema_reasoning_mode_has_choices(tmp_path):
    """reasoning_mode is an enum — UI needs the choice list."""
    p = tmp_path / "config.yaml"
    p.write_text("a: 1\n", encoding="utf-8")
    async with _client(config_path=p) as c:
        r = await c.get("/api/config/schema")
    by_name = {e["field"]: e for e in r.json()["llm_params"]}
    entry = by_name["reasoning_mode"]
    assert entry["type"] == "enum"
    assert set(entry["choices"]) == {"off", "low", "medium", "high"}


async def test_upload_endpoint_saves_files_and_serves_them(tmp_path, monkeypatch):
    # Re-point the upload dir at tmp so we don't litter the repo workspace.
    import src.dashboard.routes.uploads as uploads_route
    monkeypatch.setattr(uploads_route, "_UPLOAD_DIR", tmp_path / "uploads")

    async with _client() as c:
        files = [("files", ("hello.txt", b"hi there", "text/plain"))]
        r = await c.post("/api/chat/upload", files=files)
        assert r.status_code == 200
        body = r.json()
        assert len(body["files"]) == 1
        f = body["files"][0]
        assert f["name"] == "hello.txt"
        assert f["type"] == "text/plain"
        assert f["size"] == 8
        assert f["url"].startswith("/uploads/")

        # And the served file is fetchable
        r2 = await c.get(f["url"])
        assert r2.status_code == 200
        assert r2.content == b"hi there"


# ---- Reflects discovery + per-plugin config endpoints ----------------


async def test_reflects_available_lists_metadata(tmp_path):
    """The /api/reflects/available endpoint exposes every Reflect's
    static metadata so the dashboard can render the available-plugin
    list without importing plugin code."""
    p = tmp_path / "config.yaml"
    p.write_text("a: 1\n", encoding="utf-8")
    async with _client(config_path=p) as c:
        r = await c.get("/api/reflects/available")
    assert r.status_code == 200
    names = {entry["name"] for entry in r.json()["reflects"]}
    # All three in-tree built-ins must be discovered
    assert {"default_hypothalamus", "default_recall_anchor",
            "default_in_mind"} <= names

    # Hypothalamus declares its `translator` purpose
    by_name = {e["name"]: e for e in r.json()["reflects"]}
    purposes = by_name["default_hypothalamus"]["llm_purposes"]
    assert any(p.get("name") == "translator" for p in purposes)


async def test_reflect_config_get_missing_returns_empty(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("a: 1\n", encoding="utf-8")
    async with _client(config_path=p) as c:
        r = await c.get("/api/reflects/some_reflect/config")
    assert r.status_code == 200
    assert r.json()["config"] == {}


async def test_reflect_config_save_and_read_back(tmp_path, monkeypatch):
    """POST /api/reflects/<name>/config writes the workspace file;
    a follow-up GET reads it back. Path is rooted under
    `workspace/plugins/`; isolate to tmp_path so the test doesn't
    write into the real workspace."""
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("a: 1\n", encoding="utf-8")

    async with _client(config_path=cfg_path) as c:
        save = await c.post(
            "/api/reflects/default_hypothalamus/config",
            json={"config": {
                "llm_purposes": {"translator": "fast_generation"},
            }},
        )
        assert save.status_code == 200
        assert save.json()["restart_required"] is True

        read = await c.get("/api/reflects/default_hypothalamus/config")
        assert read.status_code == 200
        body = read.json()["config"]
        assert body["llm_purposes"] == {"translator": "fast_generation"}

    # File actually landed under workspace/plugins/
    written = (tmp_path / "workspace" / "plugins"
               / "default_hypothalamus" / "config.yaml")
    assert written.exists()
    parsed = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert parsed["llm_purposes"]["translator"] == "fast_generation"
