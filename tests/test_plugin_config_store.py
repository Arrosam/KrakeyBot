"""Per-plugin config file store — minimal in-folder layout.

Covers the small surface of FilePluginConfigStore after the
plugin-configs unification (Samuel 2026-04-27):
  - file lives at <root>/<name>/config.yaml
  - read returns {} when missing
  - write creates parent dirs + writes YAML
  - read after write round-trips the same dict
"""
from __future__ import annotations

import yaml

from krakey.plugin_system.config import FilePluginConfigStore


def test_path_for_uses_in_folder_layout(tmp_path):
    store = FilePluginConfigStore(root=tmp_path)
    assert store.path_for("greeter") == tmp_path / "greeter" / "config.yaml"


def test_read_returns_empty_dict_when_missing(tmp_path):
    store = FilePluginConfigStore(root=tmp_path)
    assert store.read("ghost") == {}


def test_read_returns_empty_dict_when_yaml_not_a_mapping(tmp_path):
    """File exists but contains a list / scalar — defensive return."""
    p = tmp_path / "weird" / "config.yaml"
    p.parent.mkdir(parents=True)
    p.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    store = FilePluginConfigStore(root=tmp_path)
    assert store.read("weird") == {}


def test_write_creates_dir_and_persists_yaml(tmp_path):
    store = FilePluginConfigStore(root=tmp_path)
    out = store.write("hello", {"greeting": "hi", "count": 3})
    assert out == tmp_path / "hello" / "config.yaml"
    assert out.exists()
    on_disk = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert on_disk == {"greeting": "hi", "count": 3}


def test_read_after_write_roundtrips(tmp_path):
    store = FilePluginConfigStore(root=tmp_path)
    store.write("p", {"a": 1, "b": "two"})
    assert store.read("p") == {"a": 1, "b": "two"}


def test_write_overwrites_existing_file(tmp_path):
    store = FilePluginConfigStore(root=tmp_path)
    store.write("p", {"a": 1})
    store.write("p", {"a": 2, "b": 3})
    assert store.read("p") == {"a": 2, "b": 3}
