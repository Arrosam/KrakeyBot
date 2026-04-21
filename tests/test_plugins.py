"""Plugin loader — discover + instantiate + fail-soft."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.plugins.loader import (
    discover_sensories, discover_tentacles,
)


# ---------------- helpers ----------------


def _write_plugin(dir_path: Path, name: str, body: str,
                     manifest_yaml: str | None = None,
                     as_package: bool = False) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    if as_package:
        pkg = dir_path / name
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").write_text(body, encoding="utf-8")
        if manifest_yaml is not None:
            (pkg / "manifest.yaml").write_text(manifest_yaml, encoding="utf-8")
    else:
        (dir_path / f"{name}.py").write_text(body, encoding="utf-8")
        if manifest_yaml is not None:
            # yaml next to single-file plugin lives in a sibling dir matching
            # the module name; for single-file, keep yaml inline via MANIFEST
            raise AssertionError("manifest.yaml only for package plugins")


# ---------------- discovery ----------------


def test_discover_nothing_when_dir_missing(tmp_path):
    out = discover_tentacles(tmp_path / "nope", deps={}, config={})
    assert out == []


def test_single_file_tentacle_with_factory(tmp_path):
    body = """
from src.interfaces.tentacle import Tentacle
from src.models.stimulus import Stimulus
from datetime import datetime

MANIFEST = {
    "description": "t",
    "is_internal": True,
    "config_schema": [
        {"field": "greeting", "type": "text", "default": "hi"},
    ],
}

class EchoT(Tentacle):
    def __init__(self, greeting):
        self.greeting = greeting
    @property
    def name(self): return "echo"
    @property
    def description(self): return "e"
    @property
    def parameters_schema(self): return {}
    @property
    def is_internal(self): return True
    async def execute(self, intent, params):
        return Stimulus(type="tentacle_feedback", source="tentacle:echo",
                        content=self.greeting, timestamp=datetime.now())

def create_tentacle(config, deps):
    return EchoT(greeting=config["greeting"])
"""
    d = tmp_path / "tentacles"
    _write_plugin(d, "echo", body)

    out = discover_tentacles(d, deps={}, config={})
    assert len(out) == 1
    info = out[0]
    assert info.error is None, info.error
    assert info.name == "echo"
    assert info.is_internal is True
    assert info.config_schema[0]["field"] == "greeting"
    assert info.instance is not None
    assert info.instance.name == "echo"
    # Default from schema applied (user cfg is empty)
    assert info.instance.greeting == "hi"


def test_package_plugin_with_manifest_yaml_overrides_inline(tmp_path):
    body = """
from src.interfaces.tentacle import Tentacle
from src.models.stimulus import Stimulus
from datetime import datetime

MANIFEST = {"description": "inline-desc", "is_internal": False,
            "config_schema": []}

class P(Tentacle):
    @property
    def name(self): return "p"
    @property
    def description(self): return "p"
    @property
    def parameters_schema(self): return {}
    async def execute(self, intent, params):
        return Stimulus(type="tentacle_feedback", source="tentacle:p",
                        content="", timestamp=datetime.now())

TENTACLE_CLASS = P
"""
    yaml_override = """
description: yaml-desc
is_internal: true
"""
    d = tmp_path / "tentacles"
    _write_plugin(d, "p", body, manifest_yaml=yaml_override, as_package=True)

    out = discover_tentacles(d, deps={}, config={})
    assert len(out) == 1
    info = out[0]
    assert info.error is None
    # yaml wins
    assert info.description == "yaml-desc"
    assert info.is_internal is True


def test_broken_plugin_is_reported_not_raised(tmp_path):
    body = "import this_module_does_not_exist_xyz123\n"
    d = tmp_path / "tentacles"
    _write_plugin(d, "bad", body)

    out = discover_tentacles(d, deps={}, config={})
    assert len(out) == 1
    info = out[0]
    assert info.error is not None
    assert "xyz123" in info.error or "ModuleNotFound" in info.error
    assert info.instance is None


def test_disabled_plugin_skips_instantiation(tmp_path):
    body = """
from src.interfaces.tentacle import Tentacle
from src.models.stimulus import Stimulus
from datetime import datetime

MANIFEST = {
    "config_schema": [
        {"field": "enabled", "type": "bool", "default": True},
    ],
}

class Never(Tentacle):
    def __init__(self):
        raise RuntimeError("must not build when disabled")
    @property
    def name(self): return "never"
    @property
    def description(self): return ""
    @property
    def parameters_schema(self): return {}
    async def execute(self, intent, params):
        raise NotImplementedError

def create_tentacle(config, deps):
    return Never()
"""
    d = tmp_path / "tentacles"
    _write_plugin(d, "never", body)
    # User config disables it
    out = discover_tentacles(d, deps={}, config={"never": {"enabled": False}})
    assert len(out) == 1
    assert out[0].error is None
    assert out[0].instance is None   # not built


def test_hidden_and_pycache_dirs_skipped(tmp_path):
    d = tmp_path / "tentacles"
    d.mkdir()
    (d / "__pycache__").mkdir()
    (d / ".hidden").mkdir()
    (d / "_private.py").write_text("MANIFEST = {}", encoding="utf-8")
    assert discover_tentacles(d, deps={}, config={}) == []


def test_sensory_discover_uses_same_contract(tmp_path):
    body = """
from src.interfaces.sensory import Sensory

class S(Sensory):
    @property
    def name(self): return "s"
    async def start(self, buffer): pass
    async def stop(self): pass

def create_sensory(config, deps):
    return S()
"""
    d = tmp_path / "sensories"
    _write_plugin(d, "s_plugin", body)
    out = discover_sensories(d, deps={}, config={})
    assert len(out) == 1
    assert out[0].instance is not None
    assert out[0].instance.name == "s"


def test_config_schema_defaults_fill_user_overrides(tmp_path):
    body = """
from src.interfaces.tentacle import Tentacle
from src.models.stimulus import Stimulus
from datetime import datetime

MANIFEST = {
    "config_schema": [
        {"field": "k", "type": "number", "default": 10},
    ],
}

class P(Tentacle):
    def __init__(self, k):
        self.k = k
    @property
    def name(self): return "p"
    @property
    def description(self): return ""
    @property
    def parameters_schema(self): return {}
    async def execute(self, intent, params):
        return Stimulus(type="tentacle_feedback", source="t:p", content="",
                        timestamp=datetime.now())

def create_tentacle(config, deps):
    return P(k=config["k"])
"""
    d = tmp_path / "tentacles"
    _write_plugin(d, "p", body)
    # No user override → default 10
    out1 = discover_tentacles(d, deps={}, config={})
    assert out1[0].instance.k == 10
    # User override → 42
    out2 = discover_tentacles(d, deps={}, config={"p": {"k": 42}})
    assert out2[0].instance.k == 42
