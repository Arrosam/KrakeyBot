"""Safe plugin discovery + import.

Layout each plugin under `workspace/tentacles/<name>/` or
`workspace/sensories/<name>/` with either

    workspace/tentacles/<name>.py              # single-file
    workspace/tentacles/<name>/__init__.py     # package

The module must expose one of:

    create_tentacle(config: dict, deps: dict) -> Tentacle     # factory
    TENTACLE_CLASS = MyTentacle                               # direct class
    Sensory variant: create_sensory / SENSORY_CLASS

and optionally a `MANIFEST` dict:

    MANIFEST = {
        "name":        "my_plugin",            # defaults to filename
        "description": "one-liner",
        "is_internal": True,                    # tentacles only
        "config_schema": [
            {"field": "api_key", "type": "password", "default": "",
             "help": "..."},
            {"field": "max_results", "type": "number", "default": 5},
        ],
    }

A standalone `manifest.yaml` next to the module is read if present and
merged on top of in-module `MANIFEST` (yaml wins).

Safety: each plugin loads in its own spec_from_file_location. ImportErrors
and any exception during import are logged and the plugin is skipped —
one bad plugin must not break Krakey's boot.
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml


@dataclass
class PluginInfo:
    """Descriptor returned by discovery for REST/UI consumption."""
    name: str
    kind: str                           # "tentacle" | "sensory"
    source: str                         # "builtin" | "plugin"
    path: str                           # absolute path on disk, or "" for builtin
    description: str = ""
    is_internal: bool = False
    config_schema: list[dict[str, Any]] = field(default_factory=list)
    # Errors encountered at import time (if any). Plugin with errors is
    # NOT registered but IS reported so the UI can surface the problem.
    error: str | None = None
    # The instance or factory-result; attached by the loader after a
    # successful build_*(). Never serialised to JSON.
    instance: Any = None


# ---------------- discovery entrypoints ----------------


def discover_tentacles(dir_path: Path, deps: dict[str, Any],
                          config: dict[str, dict[str, Any]]
                          ) -> list[PluginInfo]:
    """Walk dir_path for tentacle plugins. Returns PluginInfo list (one
    per plugin directory / file). Successful ones have .instance set to
    a Tentacle; failures have .error populated and .instance=None.

    `config` is the runtime's `tentacle:` config dict. Per-plugin entry
    (config[name]) is passed to the plugin factory.
    """
    return _discover(dir_path, kind="tentacle", deps=deps, config=config)


def discover_sensories(dir_path: Path, deps: dict[str, Any],
                          config: dict[str, dict[str, Any]]
                          ) -> list[PluginInfo]:
    return _discover(dir_path, kind="sensory", deps=deps, config=config)


# ---------------- internals ----------------


def _discover(dir_path: Path, *, kind: str, deps: dict[str, Any],
                 config: dict[str, dict[str, Any]]) -> list[PluginInfo]:
    out: list[PluginInfo] = []
    if not dir_path.exists():
        return out
    for entry in sorted(dir_path.iterdir()):
        if entry.name.startswith((".", "_")) or entry.name == "__pycache__":
            continue
        if entry.is_file() and entry.suffix == ".py":
            module_path = entry
            pkg_dir = None
        elif entry.is_dir() and (entry / "__init__.py").exists():
            module_path = entry / "__init__.py"
            pkg_dir = entry
        else:
            continue

        default_name = pkg_dir.name if pkg_dir else entry.stem
        info = PluginInfo(name=default_name, kind=kind, source="plugin",
                           path=str(module_path))
        try:
            module = _import_module(module_path, default_name, kind)
            manifest = _read_manifest(module, pkg_dir or module_path.parent)
            _fill_info_from_manifest(info, manifest, default_name)
            # User config for THIS plugin (may be None)
            user_cfg = (config or {}).get(info.name, {}) or {}
            # Default-merge so plugin sees declared defaults unless
            # overridden
            merged_cfg = _apply_defaults(user_cfg, info.config_schema)
            # Skip disabled plugins — but still report them in the list
            if not merged_cfg.get("enabled", True):
                info.instance = None
                out.append(info)
                continue
            instance = _instantiate(module, kind, merged_cfg, deps)
            info.instance = instance
        except Exception as e:  # noqa: BLE001
            info.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            info.instance = None
        out.append(info)
    return out


def _import_module(module_path: Path, module_name: str, kind: str):
    # Namespace plugin modules so they don't collide with site-packages
    # and can be re-imported (pytest, live config reload).
    full_name = f"krakey_plugin_{kind}_{module_name}"
    spec = importlib.util.spec_from_file_location(full_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load plugin at {module_path}")
    module = importlib.util.module_from_spec(spec)
    # Expose under sys.modules so relative imports inside the plugin work
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def _read_manifest(module, plugin_dir: Path) -> dict[str, Any]:
    # Merge: module MANIFEST dict → overridden by yaml file if present
    manifest: dict[str, Any] = dict(getattr(module, "MANIFEST", {}) or {})
    yaml_path = plugin_dir / "manifest.yaml"
    if yaml_path.exists():
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                manifest.update(data)
        except yaml.YAMLError as e:
            raise ValueError(f"manifest.yaml parse error: {e}") from e
    return manifest


def _fill_info_from_manifest(info: PluginInfo, manifest: dict[str, Any],
                                default_name: str) -> None:
    info.name = str(manifest.get("name") or default_name)
    info.description = str(manifest.get("description") or "")
    info.is_internal = bool(manifest.get("is_internal", False))
    raw_schema = manifest.get("config_schema") or []
    if not isinstance(raw_schema, list):
        raise ValueError("config_schema must be a list of field dicts")
    info.config_schema = [_normalize_field(f) for f in raw_schema]


def _normalize_field(f: dict[str, Any]) -> dict[str, Any]:
    if "field" not in f:
        raise ValueError(f"config_schema entry missing 'field': {f}")
    out = {
        "field": str(f["field"]),
        "type": str(f.get("type", "text")),
        "default": f.get("default"),
        "help": str(f.get("help", "")),
    }
    return out


def _apply_defaults(user_cfg: dict[str, Any],
                      schema: list[dict[str, Any]]) -> dict[str, Any]:
    merged = dict(user_cfg)
    for field_def in schema:
        name = field_def["field"]
        if name not in merged and field_def.get("default") is not None:
            merged[name] = field_def["default"]
    return merged


def _instantiate(module, kind: str, config: dict[str, Any],
                    deps: dict[str, Any]):
    factory_name = f"create_{kind}"
    class_name = f"{kind.upper()}_CLASS"
    factory: Callable | None = getattr(module, factory_name, None)
    if factory is not None:
        return factory(config, deps)
    cls = getattr(module, class_name, None)
    if cls is None:
        # Fall back: look for any class that subclasses the base Tentacle/
        # Sensory so plugin authors can omit the export. Inspect module.
        from src.interfaces.tentacle import Tentacle
        from src.interfaces.sensory import Sensory
        base = Tentacle if kind == "tentacle" else Sensory
        for val in vars(module).values():
            if isinstance(val, type) and val is not base \
                    and issubclass(val, base):
                cls = val
                break
    if cls is None:
        raise ImportError(
            f"plugin module has no create_{kind}() factory, "
            f"{class_name}, or subclass of {kind.capitalize()}"
        )
    # Try calling with config kw, then bare
    try:
        return cls(**config) if config else cls()
    except TypeError:
        return cls()
