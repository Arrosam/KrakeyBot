"""Unified plugin discovery — pure-text meta.yaml + components list.

A "plugin" is the unit of distribution + enable. A plugin can declare
any combination of components, each one of:

  * ``reflect``  — heartbeat hook (hypothalamus / recall_anchor /
                   in_mind / future kinds)
  * ``tentacle`` — outbound action
  * ``sensory``  — inbound stimulus producer

This unified loader replaces the old split between
``src/reflects/builtin/<name>/`` (Reflect-only) and
``src/plugins/builtin/<name>/__init__.py`` (Python ``MANIFEST = {}``
loader for tentacles + sensories — Phase 2 will fold the latter in too).

## Architectural rules (Samuel 2026-04-26)

1. **No code load before user enable.** Discovery walks ``meta.yaml``
   files only. Plugin Python modules are imported lazily on enable
   via ``load_plugin_components(name, ctx)``.
2. **Plugin granularity for enable** — checking a plugin in config
   loads ALL its components together. No per-component toggle.
3. **All plugins default OFF.** Empty ``config.plugins:`` list = zero
   components. The user must explicitly opt in.

## meta.yaml schema

```yaml
name: my_plugin
description: "..."
config_schema: []          # plugin-level config fields (UI hints)

components:
  - kind: reflect
    sub_kind: hypothalamus           # which Reflect kind
    factory_module: src.plugins.my_plugin.reflect
    factory_attr: build_reflect
    llm_purposes:                    # optional; what LLM purposes this
      - name: translator             # component declares
        description: "..."
        suggested_tag: fast_generation

  - kind: tentacle
    factory_module: src.plugins.my_plugin.tentacle
    factory_attr: build_tentacle
    # tentacle-specific fields (capabilities, sandboxed, etc.) can
    # live alongside; the loader passes them straight to the
    # tentacle's __init__ if it accepts them.

  - kind: sensory
    factory_module: src.plugins.my_plugin.sensory
    factory_attr: build_sensory
```
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from src.interfaces.plugin_context import PluginContext


# Roots scanned for plugin manifests. Built-in plugins ship with the
# Krakey repo at src/plugins/<name>/; workspace plugins live at
# workspace/plugins/<name>/ and are dropped in by the user. Both
# locations have identical structure (a folder with meta.yaml +
# component code) — the only distinction is "ships with code" vs
# "user-installed".
_BUILTIN_ROOT = Path(__file__).resolve().parent.parent / "plugins"
_WORKSPACE_ROOT = Path("workspace") / "plugins"


@dataclass
class ComponentMetadata:
    """One entry in a plugin's ``components:`` list."""
    kind: str  # "reflect" | "tentacle" | "sensory"
    factory_module: str
    factory_attr: str
    sub_kind: str | None = None  # for kind="reflect": hypothalamus / etc
    llm_purposes: list[dict[str, Any]] = field(default_factory=list)
    # Anything else from the component dict is preserved as `extra` so
    # plugin-specific options can ride along without schema changes.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginMetadata:
    """Parsed contents of a single ``meta.yaml`` file (unified
    plugin format, Samuel 2026-04-26)."""
    name: str
    description: str
    components: list[ComponentMetadata] = field(default_factory=list)
    config_schema: list[dict[str, Any]] = field(default_factory=list)
    # Plugin self-declares whether it depends on the sandbox VM. When
    # True and the user has the plugin enabled, Runtime preflights the
    # guest agent at startup. Lets us drop the hardcoded plugin-name
    # list that used to live in main._preflight_sandbox.
    requires_sandbox: bool = False
    source_path: Path | None = None


def discover_plugins() -> dict[str, PluginMetadata]:
    """Scan all known plugin roots for ``meta.yaml`` files.

    Pure-text — no plugin module is imported. Returns
    ``name → PluginMetadata``. Workspace plugins override built-ins
    with the same name (later iteration wins).

    Malformed manifests are logged and skipped — best-effort scan.
    """
    out: dict[str, PluginMetadata] = {}
    for root in (_BUILTIN_ROOT, _WORKSPACE_ROOT):
        if not root.exists():
            continue
        for meta_path in sorted(root.glob("*/meta.yaml")):
            try:
                meta = _parse_meta(meta_path)
            except Exception as e:  # noqa: BLE001
                print(
                    f"warning: failed to parse {meta_path}: {e}; skipping",
                    file=sys.stderr,
                )
                continue
            out[meta.name] = meta
    return out


def load_component(component: ComponentMetadata, ctx: "PluginContext") -> Any:
    """Lazily import + invoke one component's factory.

    ``ctx`` carries the per-plugin config + LLM resolutions; the
    factory may return ``None`` to opt out (e.g. unbound LLM purpose).
    """
    module = importlib.import_module(component.factory_module)
    factory = getattr(module, component.factory_attr)
    return factory(ctx)


def _parse_meta(path: Path) -> PluginMetadata:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("meta.yaml top level must be a YAML mapping")
    if not raw.get("name"):
        raise ValueError("meta.yaml missing required field: name")
    components_raw = raw.get("components") or []
    if not isinstance(components_raw, list):
        raise ValueError("meta.yaml `components:` must be a list")
    components = [_parse_component(c) for c in components_raw]
    schema = raw.get("config_schema") or []
    if not isinstance(schema, list):
        raise ValueError("meta.yaml `config_schema:` must be a list")
    return PluginMetadata(
        name=str(raw["name"]),
        description=str(raw.get("description", "")),
        components=components,
        config_schema=list(schema),
        requires_sandbox=bool(raw.get("requires_sandbox", False)),
        source_path=path,
    )


_KNOWN_COMPONENT_KINDS = {"reflect", "tentacle", "sensory"}


def _parse_component(c: Any) -> ComponentMetadata:
    if not isinstance(c, dict):
        raise ValueError("each component must be a mapping")
    kind = str(c.get("kind", "")).strip()
    if kind not in _KNOWN_COMPONENT_KINDS:
        raise ValueError(
            f"component kind {kind!r} not recognised; expected one of "
            f"{sorted(_KNOWN_COMPONENT_KINDS)}"
        )
    if not c.get("factory_module") or not c.get("factory_attr"):
        raise ValueError("component missing factory_module / factory_attr")
    purposes = c.get("llm_purposes") or []
    if not isinstance(purposes, list):
        raise ValueError("component `llm_purposes:` must be a list")
    # Stash the rest (sub_kind already pulled, factory_* already pulled)
    known = {"kind", "sub_kind", "factory_module", "factory_attr",
             "llm_purposes"}
    extra = {k: v for k, v in c.items() if k not in known}
    return ComponentMetadata(
        kind=kind,
        sub_kind=str(c["sub_kind"]) if c.get("sub_kind") else None,
        factory_module=str(c["factory_module"]),
        factory_attr=str(c["factory_attr"]),
        llm_purposes=list(purposes),
        extra=extra,
    )
