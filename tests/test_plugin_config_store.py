"""Per-plugin config file store.

Covers:
  - first-run file generation from schema defaults
  - subsequent loads return file contents verbatim
  - empty-schema plugins never touch the filesystem
  - legacy central-config values migrate once on first discovery
  - dashboard-style write round-trips through load
  - loader integrates with a file-backed store
"""
from __future__ import annotations

from pathlib import Path

import yaml

from src.plugins.loader import discover_plugins
from src.plugins.plugin_config import (
    DictPluginConfigStore,
    FilePluginConfigStore,
)


def _write_single(dir_path: Path, name: str, body: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{name}.py").write_text(body, encoding="utf-8")


# ---------------- FilePluginConfigStore semantics ----------------


def test_load_or_init_creates_file_from_schema_defaults(tmp_path):
    store = FilePluginConfigStore(root=tmp_path / "cfgs")
    schema = [
        {"field": "greeting", "type": "text", "default": "hi"},
        {"field": "count", "type": "number", "default": 3},
    ]
    cfg = store.load_or_init("hello", schema)
    assert cfg == {"enabled": False, "greeting": "hi", "count": 3}
    path = tmp_path / "cfgs" / "hello.yaml"
    assert path.exists()
    on_disk = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert on_disk["greeting"] == "hi"


def test_load_or_init_reads_existing_file_unchanged(tmp_path):
    root = tmp_path / "cfgs"
    root.mkdir()
    (root / "existing.yaml").write_text(
        "enabled: true\ngreeting: howdy\n", encoding="utf-8",
    )
    store = FilePluginConfigStore(root=root)
    cfg = store.load_or_init("existing",
                                [{"field": "greeting", "default": "hi"}])
    # Existing file wins — default "hi" must NOT overwrite "howdy".
    assert cfg["greeting"] == "howdy"
    assert cfg["enabled"] is True


def test_empty_schema_creates_no_file(tmp_path):
    """User rule: if a plugin declares no config_schema, leave it alone."""
    root = tmp_path / "cfgs"
    store = FilePluginConfigStore(root=root)
    cfg = store.load_or_init("no_schema", [])
    # Returned config is minimal and purely in-memory.
    assert cfg == {"enabled": False}
    # No file was written.
    assert not (root / "no_schema.yaml").exists()
    assert not root.exists() or not any(root.iterdir())


def test_legacy_values_migrate_on_first_init(tmp_path):
    """Old config.yaml plugins.<project> dict values carry across once."""
    store = FilePluginConfigStore(
        root=tmp_path / "cfgs",
        legacy_plugins={"search": {"enabled": True, "max_results": 9}},
    )
    schema = [{"field": "max_results", "default": 5}]
    cfg = store.load_or_init("search", schema)
    # Legacy value trumps schema default.
    assert cfg == {"enabled": True, "max_results": 9}
    # And the migration was persisted.
    on_disk = yaml.safe_load(
        (tmp_path / "cfgs" / "search.yaml").read_text(encoding="utf-8"),
    )
    assert on_disk["max_results"] == 9


def test_legacy_respected_for_schemaless_plugin(tmp_path):
    """Even with no schema, `enabled: true` from legacy should survive."""
    store = FilePluginConfigStore(
        root=tmp_path / "cfgs",
        legacy_plugins={"p": {"enabled": True, "anything": "x"}},
    )
    cfg = store.load_or_init("p", [])
    assert cfg["enabled"] is True
    assert cfg["anything"] == "x"
    # Still no file, per the "don't write schemaless configs" rule.
    assert not (tmp_path / "cfgs" / "p.yaml").exists()


def test_write_roundtrips_through_load(tmp_path):
    """Dashboard save → next start reads same values."""
    store = FilePluginConfigStore(root=tmp_path / "cfgs")
    path = store.write("dash_edit", {"enabled": True, "knob": 42})
    assert path == tmp_path / "cfgs" / "dash_edit.yaml"
    cfg = store.load_or_init("dash_edit",
                                [{"field": "knob", "default": 0}])
    # File wins over schema default.
    assert cfg == {"enabled": True, "knob": 42}


def test_peek_config_returns_file_when_present(tmp_path):
    root = tmp_path / "cfgs"
    root.mkdir()
    (root / "w.yaml").write_text(
        "history_path: x.jsonl\n", encoding="utf-8",
    )
    store = FilePluginConfigStore(root=root)
    assert store.peek_config("w") == {"history_path": "x.jsonl"}


def test_peek_config_falls_back_to_legacy_when_file_missing(tmp_path):
    store = FilePluginConfigStore(
        root=tmp_path / "cfgs",
        legacy_plugins={"w": {"history_path": "legacy.jsonl"}},
    )
    # peek_config does NOT materialize the file — no schema is known.
    assert store.peek_config("w") == {"history_path": "legacy.jsonl"}
    assert not (tmp_path / "cfgs" / "w.yaml").exists()


def test_peek_config_returns_empty_when_unknown(tmp_path):
    store = FilePluginConfigStore(root=tmp_path / "cfgs")
    assert store.peek_config("ghost") == {}


# ---------------- Loader integration via store ----------------


def test_loader_generates_per_plugin_file_on_first_run(tmp_path):
    body = """
from src.interfaces.tentacle import Tentacle
from src.models.stimulus import Stimulus
from datetime import datetime

MANIFEST = {
    "config_schema": [
        {"field": "greeting", "type": "text", "default": "hi"},
    ],
}

class T(Tentacle):
    def __init__(self, greeting): self.greeting = greeting
    @property
    def name(self): return "gen"
    @property
    def description(self): return ""
    @property
    def parameters_schema(self): return {}
    async def execute(self, intent, params):
        return Stimulus(type="tentacle_feedback", source="t:gen",
                        content=self.greeting, timestamp=datetime.now())

def create_tentacle(config, deps):
    return T(greeting=config["greeting"])
"""
    plug_dir = tmp_path / "plugins"
    _write_single(plug_dir, "gen", body)
    cfg_root = tmp_path / "cfgs"

    store = FilePluginConfigStore(root=cfg_root)
    out = discover_plugins(plug_dir, deps={}, config_store=store)
    assert len(out) == 1
    info = out[0]
    # enabled defaults to False → factory should NOT have run,
    # but the config file must have been generated anyway.
    assert info.enabled is False
    assert info.instance is None
    gen_file = cfg_root / "gen.yaml"
    assert gen_file.exists()
    on_disk = yaml.safe_load(gen_file.read_text(encoding="utf-8"))
    assert on_disk == {"enabled": False, "greeting": "hi"}


def test_loader_respects_edits_to_per_plugin_file(tmp_path):
    body = """
from src.interfaces.tentacle import Tentacle
from src.models.stimulus import Stimulus
from datetime import datetime

MANIFEST = {"config_schema": [
    {"field": "greeting", "type": "text", "default": "hi"},
]}

class T(Tentacle):
    def __init__(self, greeting): self.greeting = greeting
    @property
    def name(self): return "gen"
    @property
    def description(self): return ""
    @property
    def parameters_schema(self): return {}
    async def execute(self, intent, params):
        return Stimulus(type="tentacle_feedback", source="t:gen",
                        content=self.greeting, timestamp=datetime.now())

def create_tentacle(config, deps): return T(greeting=config["greeting"])
"""
    plug_dir = tmp_path / "plugins"
    _write_single(plug_dir, "gen", body)
    cfg_root = tmp_path / "cfgs"
    cfg_root.mkdir()
    (cfg_root / "gen.yaml").write_text(
        "enabled: true\ngreeting: howdy\n", encoding="utf-8",
    )

    store = FilePluginConfigStore(root=cfg_root)
    out = discover_plugins(plug_dir, deps={}, config_store=store)
    info = out[0]
    assert info.enabled is True
    assert info.instance is not None
    assert info.instance.greeting == "howdy"


def test_loader_uses_dict_store_shim_when_no_store_kwarg(tmp_path):
    """Legacy call path: `configs={...}` still works without a store."""
    body = """
from src.interfaces.tentacle import Tentacle
from src.models.stimulus import Stimulus
from datetime import datetime

MANIFEST = {"config_schema": [
    {"field": "greeting", "default": "hi"},
]}

class T(Tentacle):
    def __init__(self, greeting): self.greeting = greeting
    @property
    def name(self): return "legacy"
    @property
    def description(self): return ""
    @property
    def parameters_schema(self): return {}
    async def execute(self, intent, params):
        return Stimulus(type="tentacle_feedback", source="t:legacy",
                        content="", timestamp=datetime.now())

def create_tentacle(config, deps): return T(greeting=config["greeting"])
"""
    plug_dir = tmp_path / "plugins"
    _write_single(plug_dir, "legacy", body)
    out = discover_plugins(
        plug_dir,
        deps={},
        configs={"legacy": {"enabled": True, "greeting": "dict"}},
    )
    assert out[0].instance.greeting == "dict"


def test_dict_store_standalone():
    """Direct smoke test of DictPluginConfigStore."""
    s = DictPluginConfigStore({"x": {"enabled": True, "v": 1}})
    assert s.peek_config("x") == {"enabled": True, "v": 1}
    # Schema is ignored by the dict shim; returns raw entry.
    assert s.load_or_init("x", [{"field": "v", "default": 99}]) == {
        "enabled": True, "v": 1,
    }
    # Unknown project → empty dict (plus loader will apply defaults).
    assert s.load_or_init("missing", []) == {}
