"""Plugin loader — discover + instantiate + fail-soft + multi-component."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.plugins.loader import discover_plugins


def _write_single(dir_path: Path, name: str, body: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{name}.py").write_text(body, encoding="utf-8")


def _write_pkg(dir_path: Path, name: str, body: str,
                  manifest_yaml: str | None = None) -> None:
    pkg = dir_path / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(body, encoding="utf-8")
    if manifest_yaml is not None:
        (pkg / "manifest.yaml").write_text(manifest_yaml, encoding="utf-8")


# ---------------- single-component shortcuts ----------------


def test_discover_nothing_when_dir_missing(tmp_path):
    assert discover_plugins(tmp_path / "nope", deps={}, configs={}) == []


def test_single_file_tentacle_shortcut(tmp_path):
    body = """
from src.interfaces.tentacle import Tentacle
from src.models.stimulus import Stimulus
from datetime import datetime

MANIFEST = {
    "description": "t",
    "config_schema": [
        {"field": "greeting", "type": "text", "default": "hi"},
    ],
}

class EchoT(Tentacle):
    def __init__(self, greeting): self.greeting = greeting
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
    d = tmp_path / "plugins"
    _write_single(d, "echo", body)

    # Must explicitly enable; enabled defaults to False.
    out = discover_plugins(d, deps={},
                                configs={"echo": {"enabled": True}})
    assert len(out) == 1
    info = out[0]
    assert info.error is None
    assert info.name == "echo"
    assert info.kind == "tentacle"
    assert info.project == "echo"
    assert info.enabled is True
    assert info.instance is not None and info.instance.greeting == "hi"


def test_single_sensory_shortcut(tmp_path):
    body = """
from src.interfaces.sensory import Sensory

class S(Sensory):
    @property
    def name(self): return "s"
    async def start(self, buffer): pass
    async def stop(self): pass

def create_sensory(config, deps): return S()
"""
    d = tmp_path / "plugins"
    _write_single(d, "feed", body)
    out = discover_plugins(d, deps={},
                                configs={"feed": {"enabled": True}})
    assert len(out) == 1
    assert out[0].kind == "sensory"
    assert out[0].enabled is True
    assert out[0].instance is not None


# ---------------- multi-component projects ----------------


def test_create_plugins_emits_one_info_per_component(tmp_path):
    body = """
from src.interfaces.tentacle import Tentacle
from src.interfaces.sensory import Sensory
from src.models.stimulus import Stimulus
from datetime import datetime

MANIFEST = {
    "description": "paired",
    "components": [
        {"kind": "tentacle", "name": "paired_send", "is_internal": False,
         "description": "send half"},
        {"kind": "sensory",  "name": "paired_recv",
         "description": "recv half"},
    ],
    "config_schema": [
        {"field": "endpoint", "type": "text", "default": "x"},
    ],
}

class T(Tentacle):
    def __init__(self, ep): self.ep = ep
    @property
    def name(self): return "paired_send"
    @property
    def description(self): return "send"
    @property
    def parameters_schema(self): return {}
    async def execute(self, intent, params):
        return Stimulus(type="tentacle_feedback", source="t:paired_send",
                        content="", timestamp=datetime.now())

class S(Sensory):
    def __init__(self, ep): self.ep = ep
    @property
    def name(self): return "paired_recv"
    async def start(self, buffer): pass
    async def stop(self): pass

def create_plugins(config, deps):
    ep = config["endpoint"]
    return {"tentacles": [T(ep)], "sensories": [S(ep)]}
"""
    d = tmp_path / "plugins"
    _write_pkg(d, "paired", body)
    out = discover_plugins(d, deps={},
                                configs={"paired": {"enabled": True}})
    assert len(out) == 2
    # Same project name on both
    assert {i.project for i in out} == {"paired"}
    # Kinds split correctly
    assert {i.kind for i in out} == {"tentacle", "sensory"}
    # Metadata from components entries used
    names = {i.name for i in out}
    assert names == {"paired_send", "paired_recv"}
    # Shared config_schema
    for i in out:
        assert len(i.config_schema) == 1
        assert i.config_schema[0]["field"] == "endpoint"
    # Shared state (both saw the same endpoint value)
    t_inst = next(i.instance for i in out if i.kind == "tentacle")
    s_inst = next(i.instance for i in out if i.kind == "sensory")
    assert t_inst.ep == s_inst.ep == "x"


def test_create_plugins_shared_client_between_components(tmp_path):
    """Classic telegram-style: one client shared across tentacle + sensory."""
    body = """
from src.interfaces.tentacle import Tentacle
from src.interfaces.sensory import Sensory
from src.models.stimulus import Stimulus
from datetime import datetime

MANIFEST = {"components": [
    {"kind": "tentacle", "name": "x_send"},
    {"kind": "sensory",  "name": "x_recv"},
]}

class Client:
    pass

class T(Tentacle):
    def __init__(self, c): self.client = c
    @property
    def name(self): return "x_send"
    @property
    def description(self): return ""
    @property
    def parameters_schema(self): return {}
    async def execute(self, intent, params):
        return Stimulus(type="tentacle_feedback", source="t:x_send",
                        content="", timestamp=datetime.now())

class S(Sensory):
    def __init__(self, c): self.client = c
    @property
    def name(self): return "x_recv"
    async def start(self, buffer): pass
    async def stop(self): pass

def create_plugins(config, deps):
    c = Client()
    return {"tentacles": [T(c)], "sensories": [S(c)]}
"""
    d = tmp_path / "plugins"
    _write_pkg(d, "x_proj", body)
    out = discover_plugins(d, deps={},
                                configs={"x_proj": {"enabled": True}})
    t = next(i.instance for i in out if i.kind == "tentacle")
    s = next(i.instance for i in out if i.kind == "sensory")
    assert t.client is s.client  # same object — the whole point


# ---------------- error + disabled + filter paths ----------------


def test_broken_plugin_is_reported_not_raised(tmp_path):
    d = tmp_path / "plugins"
    _write_single(d, "bad", "import this_module_does_not_exist_xyz123\n")
    out = discover_plugins(d, deps={}, configs={})
    assert len(out) == 1
    assert out[0].error is not None
    assert "xyz123" in out[0].error or "ModuleNotFound" in out[0].error
    assert out[0].instance is None


def test_disabled_project_reports_but_does_not_instantiate(tmp_path):
    body = """
from src.interfaces.tentacle import Tentacle

MANIFEST = {}

class Never(Tentacle):
    def __init__(self):
        raise RuntimeError("must not build when disabled")
    @property
    def name(self): return "never"
    @property
    def description(self): return ""
    @property
    def parameters_schema(self): return {}
    async def execute(self, intent, params): raise NotImplementedError

def create_tentacle(config, deps): return Never()
"""
    d = tmp_path / "plugins"
    _write_single(d, "never", body)
    out = discover_plugins(d, deps={}, configs={"never": {"enabled": False}})
    assert len(out) == 1
    assert out[0].error is None
    assert out[0].instance is None
    assert out[0].enabled is False


def test_default_enabled_is_false_factory_not_called(tmp_path):
    """No config entry + no `enabled: true` → factory does not run,
    even though the plugin module is otherwise valid."""
    body = """
from src.interfaces.tentacle import Tentacle

MANIFEST = {"description": "default-off plugin"}

class BoomOnBuild(Tentacle):
    def __init__(self):
        raise RuntimeError("factory must not run — plugin is disabled by default")
    @property
    def name(self): return "boom"
    @property
    def description(self): return ""
    @property
    def parameters_schema(self): return {}
    async def execute(self, intent, params): raise NotImplementedError

def create_tentacle(config, deps): return BoomOnBuild()
"""
    d = tmp_path / "plugins"
    _write_single(d, "boom", body)
    # configs={} → enabled defaults to False → factory is NOT called
    out = discover_plugins(d, deps={}, configs={})
    assert len(out) == 1
    assert out[0].error is None
    assert out[0].instance is None
    assert out[0].enabled is False


def test_author_declared_enabled_is_stripped_from_schema(tmp_path):
    """`enabled` is loader-owned; anything a plugin author writes for it
    in config_schema is stripped before the dashboard sees it."""
    body = """
from src.interfaces.tentacle import Tentacle

MANIFEST = {
    "config_schema": [
        {"field": "enabled", "type": "bool", "default": True,
         "help": "this should be stripped"},
        {"field": "knob", "type": "text", "default": "x"},
    ],
}

class T(Tentacle):
    @property
    def name(self): return "stripper"
    @property
    def description(self): return ""
    @property
    def parameters_schema(self): return {}
    async def execute(self, intent, params): raise NotImplementedError

def create_tentacle(config, deps): return T()
"""
    d = tmp_path / "plugins"
    _write_single(d, "stripper", body)
    out = discover_plugins(d, deps={},
                                configs={"stripper": {"enabled": True}})
    assert len(out) == 1
    info = out[0]
    fields = [f["field"] for f in info.config_schema]
    # `enabled` stripped, author's other knob survives
    assert "enabled" not in fields
    assert "knob" in fields
    # Even though author set default=True, the loader-owned default is
    # False — only the user's config flips it on.
    assert info.enabled is True


def test_hidden_dirs_and_non_py_files_skipped(tmp_path):
    d = tmp_path / "plugins"
    d.mkdir()
    (d / "__pycache__").mkdir()
    (d / ".hidden").mkdir()
    (d / "_private.py").write_text("MANIFEST = {}", encoding="utf-8")
    (d / "README.md").write_text("# readme", encoding="utf-8")
    assert discover_plugins(d, deps={}, configs={}) == []


def test_yaml_manifest_overrides_inline(tmp_path):
    body = """
from src.interfaces.tentacle import Tentacle
MANIFEST = {"description": "inline"}

class P(Tentacle):
    @property
    def name(self): return "p"
    @property
    def description(self): return "p"
    @property
    def parameters_schema(self): return {}
    async def execute(self, intent, params): raise NotImplementedError

def create_tentacle(config, deps): return P()
"""
    d = tmp_path / "plugins"
    _write_pkg(d, "p", body, manifest_yaml="description: yaml-wins\n")
    out = discover_plugins(d, deps={}, configs={})
    assert len(out) == 1
    assert out[0].description == "yaml-wins"


def test_config_defaults_fill_user_missing_keys(tmp_path):
    body = """
from src.interfaces.tentacle import Tentacle
from src.models.stimulus import Stimulus
from datetime import datetime

MANIFEST = {"config_schema": [
    {"field": "k", "type": "number", "default": 10},
]}

class P(Tentacle):
    def __init__(self, k): self.k = k
    @property
    def name(self): return "p"
    @property
    def description(self): return ""
    @property
    def parameters_schema(self): return {}
    async def execute(self, intent, params):
        return Stimulus(type="tentacle_feedback", source="t:p",
                        content="", timestamp=datetime.now())

def create_tentacle(config, deps): return P(k=config["k"])
"""
    d = tmp_path / "plugins"
    _write_single(d, "p", body)
    # Missing keys fall back to schema defaults, but `enabled` still
    # has to be set explicitly — it defaults to False regardless.
    r1 = discover_plugins(d, deps={},
                             configs={"p": {"enabled": True}})[0]
    assert r1.instance.k == 10
    r2 = discover_plugins(d, deps={},
                             configs={"p": {"enabled": True, "k": 42}})[0]
    assert r2.instance.k == 42
