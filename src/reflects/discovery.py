"""Reflect discovery — pure-text-only metadata scan.

**Architectural invariant** (Samuel 2026-04-25): a plugin's *code*
must not be imported until the user explicitly enables it. Web UI
display, config-form rendering, "what plugins are available" lists
— all of these may only consult the plugin's pure-text metadata
file. The Python module behind it stays untouched until runtime
startup walks `config.reflects` and calls ``load_reflect(name)``.

Why: enabling a plugin is a deliberate user act with consent
implications. Code execution at scan time means a malicious plugin
could side-effect just by being present on disk. By forcing a
text-only scan path, we make the contract auditable: "available
plugin list" is a pure file-walk + YAML parse, never an import.

## File layout

Each Reflect lives in its own folder under ``src/reflects/builtin/``
(in-tree) or — eventually — ``workspace/reflects/`` (user-installed):

    src/reflects/builtin/<name>/
        meta.yaml      ← pure text manifest (name, kind, description,
                         factory module path, config schema)
        __init__.py    ← empty marker; we never import this
        reflect.py     ← actual implementation, imported lazily

Discovery (``discover_reflects``) walks for ``meta.yaml``. It never
imports the sibling Python files. ``load_reflect(name, deps)`` is
the only entry that imports the module + invokes the factory.

## meta.yaml schema

```yaml
name: default_in_mind          # unique id; what users write in config.yaml
kind: in_mind                  # which dispatch slot in ReflectRegistry
description: "..."             # human-readable, shown in Web UI
factory_module: src.reflects.builtin.default_in_mind.reflect
factory_attr: build_reflect    # callable taking RuntimeDeps → Reflect
config_schema: []              # list of {field, type, default, help}
                               # entries; same shape as plugin schemas
```
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from src.reflects.protocol import Reflect


# Where to scan. Built-ins ship with Krakey. Workspace path is
# reserved for user-installed Reflects (not implemented yet but
# wired in here so the discovery never has to be touched again
# when that lands).
_BUILTIN_ROOT = Path(__file__).resolve().parent / "builtin"
_WORKSPACE_ROOT = Path("workspace") / "reflects"


@dataclass
class ReflectMetadata:
    """Parsed contents of a single ``meta.yaml`` file."""
    name: str
    kind: str
    description: str
    factory_module: str
    factory_attr: str
    config_schema: list[dict[str, Any]] = field(default_factory=list)
    source_path: Path | None = None  # which meta.yaml this came from


def discover_reflects() -> dict[str, ReflectMetadata]:
    """Scan all known Reflect roots for ``meta.yaml`` files.

    Pure-text operation: no plugin module is imported. Returns a
    mapping of ``name → ReflectMetadata``. Duplicate names across
    roots prefer the workspace copy (so users can override built-ins
    by dropping a folder under ``workspace/reflects/<same_name>/``).

    Malformed metadata files are logged and skipped — discovery is
    best-effort and a bad plugin can't break the scan for the others.
    """
    out: dict[str, ReflectMetadata] = {}
    for root in (_BUILTIN_ROOT, _WORKSPACE_ROOT):
        if not root.exists():
            continue
        for meta_path in sorted(root.glob("*/meta.yaml")):
            try:
                meta = _parse_meta(meta_path)
            except Exception as e:  # noqa: BLE001
                # Bad meta.yaml shouldn't poison the rest of the scan.
                # Future: surface this in Web UI as "broken plugin"
                # so users notice. For now, stderr is enough.
                import sys
                print(
                    f"warning: failed to parse {meta_path}: {e}; "
                    "skipping", file=sys.stderr,
                )
                continue
            out[meta.name] = meta  # later root wins by iteration order
    return out


def load_reflect(name: str, deps: Any) -> "Reflect":
    """Look up ``name`` in discovery, lazily import its
    ``factory_module``, invoke ``factory_attr(deps)``, return the
    resulting Reflect instance.

    Raises ``KeyError`` if the name isn't in discovery (caller
    typically logs + skips per the strictly-additive plugin rule).
    Other exceptions bubble — the caller decides whether to skip
    the plugin or fail loud.
    """
    metadata = discover_reflects().get(name)
    if metadata is None:
        raise KeyError(name)
    module = importlib.import_module(metadata.factory_module)
    factory = getattr(module, metadata.factory_attr)
    return factory(deps)


def _parse_meta(path: Path) -> ReflectMetadata:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("meta.yaml must be a YAML mapping at top level")
    required = ("name", "kind", "description", "factory_module",
                 "factory_attr")
    missing = [k for k in required if not raw.get(k)]
    if missing:
        raise ValueError(f"meta.yaml missing required keys: {missing}")
    schema = raw.get("config_schema") or []
    if not isinstance(schema, list):
        raise ValueError("config_schema must be a list of field entries")
    return ReflectMetadata(
        name=str(raw["name"]),
        kind=str(raw["kind"]),
        description=str(raw["description"]),
        factory_module=str(raw["factory_module"]),
        factory_attr=str(raw["factory_attr"]),
        config_schema=list(schema),
        source_path=path,
    )
