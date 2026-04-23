"""Plugin discovery and safe import.

Layout: each plugin is a *project* folder containing zero or more
tentacles + zero or more sensories that may share state.

    <root>/<project_name>/__init__.py

Two roots are scanned:

    src/plugins/builtin/       source = "builtin"   (ships with Krakey)
    workspace/plugins/         source = "plugin"    (user-dropped)

Each project module must expose ONE of:

    create_plugins(config, deps) -> dict:
        # returns {"tentacles": [Tentacle, ...], "sensories": [Sensory, ...]}
        # Omit either key for single-kind projects. Preferred because
        # multi-component projects (e.g. telegram = sensory+tentacle
        # sharing a client) need the same `config` + `deps` in ONE
        # factory to keep shared state sane.

    create_tentacle(config, deps) -> Tentacle      # single-tentacle shortcut
    create_sensory(config, deps)  -> Sensory       # single-sensory shortcut

and optionally a MANIFEST dict (or manifest.yaml next to the module):

    MANIFEST = {
        "name":          "my_project",     # defaults to folder name
        "description":   "one-liner",
        "components": [                      # REQUIRED for create_plugins
            {"kind": "tentacle", "name": "my_thing", "is_internal": True,
             "description": "optional per-component blurb"},
            {"kind": "sensory",  "name": "my_thing_feed"},
        ],
        "config_schema": [                   # drives dashboard Settings form
            # DO NOT declare "enabled" here — it is reserved and managed
            # by the loader (default: False). The dashboard renders a
            # dedicated toggle above your schema rows.
            {"field": "api_key", "type": "password", "default": ""},
            ...
        ],
    }

For single-component shortcuts (create_tentacle / create_sensory),
`components` may be omitted — the component inherits the project name.

**Enabled is loader-owned.** Every project gets an implicit
`plugins.<project>.enabled: bool` config key with default `False`.
The factory never runs unless the user sets it to `true`. User-declared
`enabled` fields in `config_schema` are silently stripped so plugin
authors cannot override the default or the widget.

Safety: each module loads in isolation via importlib.util. Any import
or factory exception is caught, recorded on the PluginInfo.error
field, and the plugin is skipped — a broken plugin must never block
Krakey's boot.
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PluginInfo:
    """Descriptor returned by discovery for REST / UI consumption.

    One PluginInfo corresponds to one *component* (a tentacle OR a
    sensory). A multi-component project produces multiple PluginInfos
    linked by the same `project` string.
    """
    name: str                           # component name
    kind: str                           # "tentacle" | "sensory"
    source: str                         # "builtin" | "plugin"
    path: str                           # module path on disk, "" for core
    project: str = ""                   # containing project folder name
    description: str = ""
    is_internal: bool = False
    config_schema: list[dict[str, Any]] = field(default_factory=list)
    # Loader-owned. True iff `plugins.<project>.enabled: true` in
    # config.yaml. Default is False — plugins never self-enable.
    enabled: bool = False
    error: str | None = None
    # Set by the loader on success, None otherwise. Never JSON-serialised.
    instance: Any = None


def discover_plugins(dir_path: Path, deps: dict[str, Any],
                        configs: dict[str, dict[str, Any]],
                        *, source: str = "plugin"
                        ) -> list[PluginInfo]:
    """Scan `dir_path` for project folders / files and return one
    PluginInfo per *component* (tentacle or sensory) produced.

    `configs` is the runtime's `plugins:` section (mapping project name
    to its config dict). Each project's factory receives its own
    config slice.
    """
    out: list[PluginInfo] = []
    if not dir_path.exists():
        return out
    for entry in sorted(dir_path.iterdir()):
        name = entry.name
        if name.startswith((".", "_")) or name == "__pycache__":
            continue
        if entry.is_file() and entry.suffix == ".py":
            module_path = entry
            pkg_dir = None
        elif entry.is_dir() and (entry / "__init__.py").exists():
            module_path = entry / "__init__.py"
            pkg_dir = entry
        else:
            continue
        project = pkg_dir.name if pkg_dir else entry.stem
        out.extend(_load_project(
            module_path=module_path,
            pkg_dir=pkg_dir,
            project=project,
            source=source,
            deps=deps,
            project_config=configs.get(project) or {},
        ))
    return out


# ---------------- internals ----------------


def _load_project(*, module_path: Path, pkg_dir: Path | None,
                     project: str, source: str, deps: dict[str, Any],
                     project_config: dict[str, Any]) -> list[PluginInfo]:
    """Import one project module, run its factory, and return
    PluginInfo for each produced component."""
    # Base info shared by every component — gets cloned per component below.
    base = PluginInfo(
        name=project, kind="?", source=source, path=str(module_path),
        project=project,
    )
    # Resolve `enabled` BEFORE import so even a plugin that would crash
    # on import is gated by it. A user who sets enabled=false never even
    # pays the import cost of a flaky plugin.
    base.enabled = bool(project_config.get("enabled", False))
    try:
        module = _import_module(module_path, project)
        manifest = _read_manifest(module, pkg_dir or module_path.parent)
        _apply_manifest_project_fields(base, manifest, project)
        merged_cfg = _apply_defaults(project_config, base.config_schema)
        # `enabled` is loader-owned; never let a user-supplied schema or
        # project config resurrect a bogus value.
        merged_cfg["enabled"] = base.enabled
        components_meta = manifest.get("components") or []

        # enabled=false short-circuits: report everything as "present but
        # not instantiated" without running the factory. Default is
        # false — plugins must be explicitly enabled in config.yaml.
        if not base.enabled:
            return _describe_components(base, manifest, components_meta,
                                           skipped=True)

        instances = _run_factory(module, merged_cfg, deps)
    except Exception as e:  # noqa: BLE001
        base.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        base.instance = None
        # Even on error, report one PluginInfo per declared component
        # (so the dashboard knows what SHOULD have loaded).
        return _describe_components(base,
                                        {"components": _stub_components(base)},
                                        _stub_components(base), error=True)

    return _zip_components(base, manifest.get("components") or [], instances)


def _import_module(module_path: Path, project: str):
    full_name = f"krakey_plugin_{project}"
    # For package-style plugins (`<project>/__init__.py`) set
    # `submodule_search_locations` so intra-package relative imports
    # like `from .client import HttpTelegramClient` resolve correctly.
    submodule_locations: list[str] | None = None
    if module_path.name == "__init__.py":
        submodule_locations = [str(module_path.parent)]
    spec = importlib.util.spec_from_file_location(
        full_name, module_path,
        submodule_search_locations=submodule_locations,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load plugin at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def _read_manifest(module, plugin_dir: Path) -> dict[str, Any]:
    """Merge inline MANIFEST with an optional manifest.yaml (yaml wins)."""
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


def _apply_manifest_project_fields(info: PluginInfo,
                                        manifest: dict[str, Any],
                                        default_name: str) -> None:
    info.name = str(manifest.get("name") or default_name)
    info.project = info.name
    info.description = str(manifest.get("description") or "")
    info.is_internal = bool(manifest.get("is_internal", False))
    raw_schema = manifest.get("config_schema") or []
    if not isinstance(raw_schema, list):
        raise ValueError("config_schema must be a list of field dicts")
    # `enabled` is reserved; strip anything a plugin author tried to
    # declare for it. The loader owns that field (default False) and
    # the dashboard renders a dedicated widget.
    info.config_schema = [
        _normalize_field(f) for f in raw_schema
        if str(f.get("field", "")).strip() != "enabled"
    ]


def _normalize_field(f: dict[str, Any]) -> dict[str, Any]:
    if "field" not in f:
        raise ValueError(f"config_schema entry missing 'field': {f}")
    return {
        "field": str(f["field"]),
        "type": str(f.get("type", "text")),
        "default": f.get("default"),
        "help": str(f.get("help", "")),
    }


def _apply_defaults(user_cfg: dict[str, Any],
                      schema: list[dict[str, Any]]) -> dict[str, Any]:
    merged = dict(user_cfg)
    for field_def in schema:
        name = field_def["field"]
        if name not in merged and field_def.get("default") is not None:
            merged[name] = field_def["default"]
    return merged


def _run_factory(module, config: dict[str, Any],
                    deps: dict[str, Any]) -> dict[str, list]:
    """Call the project's factory and normalize to
    `{"tentacles": [...], "sensories": [...]}`."""
    if hasattr(module, "create_plugins"):
        out = module.create_plugins(config, deps) or {}
        if not isinstance(out, dict):
            raise TypeError("create_plugins must return a dict")
        return {
            "tentacles": list(out.get("tentacles") or []),
            "sensories": list(out.get("sensories") or []),
        }
    if hasattr(module, "create_tentacle"):
        return {"tentacles": [module.create_tentacle(config, deps)],
                "sensories": []}
    if hasattr(module, "create_sensory"):
        return {"tentacles": [],
                "sensories": [module.create_sensory(config, deps)]}

    # Fallback: find any Tentacle / Sensory subclass and instantiate it.
    from src.interfaces.sensory import Sensory
    from src.interfaces.tentacle import Tentacle
    tentacles, sensories = [], []
    for val in vars(module).values():
        if not isinstance(val, type):
            continue
        if val is Tentacle or val is Sensory:
            continue
        if issubclass(val, Tentacle):
            tentacles.append(_safe_construct(val, config))
        elif issubclass(val, Sensory):
            sensories.append(_safe_construct(val, config))
    if not (tentacles or sensories):
        raise ImportError(
            "plugin module exposes neither create_plugins / "
            "create_tentacle / create_sensory nor any Tentacle / Sensory "
            "subclass"
        )
    return {"tentacles": tentacles, "sensories": sensories}


def _safe_construct(cls, config):
    try:
        return cls(**config) if config else cls()
    except TypeError:
        return cls()


# ---------------- reporting helpers ----------------


def _zip_components(base: PluginInfo,
                      components_meta: list[dict[str, Any]],
                      instances: dict[str, list]) -> list[PluginInfo]:
    """Pair factory-produced instances with optional component metadata
    and emit one PluginInfo per component."""
    out: list[PluginInfo] = []
    tent_meta = [m for m in components_meta if m.get("kind") == "tentacle"]
    sens_meta = [m for m in components_meta if m.get("kind") == "sensory"]
    out.extend(_pair(base, "tentacle", instances["tentacles"], tent_meta))
    out.extend(_pair(base, "sensory",  instances["sensories"], sens_meta))
    return out


def _pair(base: PluginInfo, kind: str, instances: list,
             meta_list: list[dict[str, Any]]) -> list[PluginInfo]:
    """Build PluginInfo for each instance, preferring metadata for the
    component name / description / is_internal when provided."""
    out = []
    for i, inst in enumerate(instances):
        meta = meta_list[i] if i < len(meta_list) else {}
        inst_name = getattr(inst, "name", None) if hasattr(inst, "name") else None
        # "name" on Tentacle / Sensory is a @property; safe to read.
        try:
            inst_name = inst.name if inst_name is None else inst_name
        except Exception:  # noqa: BLE001
            inst_name = None
        info = PluginInfo(
            name=str(meta.get("name") or inst_name or base.name),
            kind=kind,
            source=base.source,
            path=base.path,
            project=base.project,
            description=str(meta.get("description") or base.description),
            is_internal=bool(meta.get("is_internal",
                                         getattr(inst, "is_internal", False))),
            config_schema=base.config_schema,
            enabled=base.enabled,
            error=None,
            instance=inst,
        )
        out.append(info)
    return out


def _describe_components(base: PluginInfo, manifest: dict[str, Any],
                           components_meta: list[dict[str, Any]],
                           *, skipped: bool = False,
                           error: bool = False) -> list[PluginInfo]:
    """Emit PluginInfos with instance=None for reporting purposes —
    used when enabled=false (skipped) or when the factory crashed."""
    out = []
    meta_iter = components_meta or [{"kind": "tentacle",
                                     "name": base.name}]
    for meta in meta_iter:
        info = PluginInfo(
            name=str(meta.get("name") or base.name),
            kind=str(meta.get("kind", "tentacle")),
            source=base.source,
            path=base.path,
            project=base.project,
            description=base.description,
            is_internal=bool(meta.get("is_internal", base.is_internal)),
            config_schema=base.config_schema,
            enabled=base.enabled,
            error=base.error if error else None,
            instance=None,
        )
        out.append(info)
    return out


def _stub_components(base: PluginInfo) -> list[dict[str, Any]]:
    """When a manifest failed to parse, produce a single-tentacle stub
    so the dashboard still sees the project name and error."""
    return [{"kind": "tentacle", "name": base.name}]
