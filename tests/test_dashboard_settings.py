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
