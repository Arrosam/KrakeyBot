"""Plugin loader — load-by-name + lazy component import (runtime side).

Pure-text meta.yaml parsing + ``importlib`` on enable. **Runtime never
scans plugin folders** — it only opens the meta.yaml files for the
names listed in ``config.yaml`` ``plugins:``. Catalogue scanning
("show me everything available") is a Web UI concern and lives in
``krakey/plugins/dashboard/services/plugin_catalogue.py``.

A "plugin" is the unit of distribution + enable. A plugin can declare
any combination of components, each one of:

  * ``modifier``  — heartbeat hook (hypothalamus / recall_anchor /
                   in_mind / future kinds)
  * ``tool`` — outbound action
  * ``channel``  — inbound stimulus producer

## Architectural rules (Samuel 2026-04-26)

1. **No code load before user enable.** ``parse_meta`` walks meta.yaml
   files only. Plugin Python modules are imported lazily on enable
   via ``load_component(component, ctx)``.
2. **Plugin granularity for enable** — checking a plugin in config
   loads ALL its components together. No per-component toggle.
3. **All plugins default OFF.** Empty ``config.plugins:`` list = zero
   components. The user must explicitly opt in.

## meta.yaml schema

```yaml
name: my_plugin
description: "..."
config_schema: []          # plugin-level config fields (UI hints)
dependencies:              # pip-installable spec strings
  - "some-package>=1.0"    # installed by ``krakey install``
post_install:              # secondary install steps after pip
  - args: ["{python}", "-m", "playwright", "install", "chromium"]
    description: "Download Chromium (~200MB, one-time)"
    optional: false        # true = swallow failure; false = abort install

components:
  - kind: modifier
    role: hypothalamus               # the Modifier's role (must be unique
                                     # across all enabled plugins)
    factory_module: src.plugins.my_plugin.modifier
    factory_attr: build_modifier
    llm_purposes:                    # optional; what LLM purposes this
      - name: translator             # component declares
        description: "..."
        suggested_tag: fast_generation

  - kind: tool
    factory_module: src.plugins.my_plugin.tool
    factory_attr: build_tool

  - kind: channel
    factory_module: src.plugins.my_plugin.channel
    factory_attr: build_channel
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
    from krakey.interfaces.plugin_context import PluginContext


# Plugin folder roots. Built-in plugins ship with the Krakey repo at
# krakey/plugins/<name>/; workspace plugins live at workspace/plugins/<name>/
# and are dropped in by the user. Both locations have identical
# structure (a folder with meta.yaml + component code) — the only
# distinction is "ships with code" vs "user-installed".
#
# Exposed (no underscore) so the dashboard's catalogue scanner can
# walk them too, without re-deriving the paths.
BUILTIN_ROOT = Path(__file__).resolve().parent.parent / "plugins"
WORKSPACE_ROOT = Path("workspace") / "plugins"


@dataclass
class ComponentMetadata:
    """One entry in a plugin's ``components:`` list."""
    kind: str  # "modifier" | "tool" | "channel"
    factory_module: str
    factory_attr: str
    role: str | None = None  # for kind="modifier": role string the
                              # Modifier claims; runtime errors on dup
    llm_purposes: list[dict[str, Any]] = field(default_factory=list)
    # Anything else from the component dict is preserved as `extra` so
    # plugin-specific options can ride along without schema changes.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginMetadata:
    """Parsed contents of a single ``meta.yaml`` file (unified
    plugin format, Samuel 2026-04-26).

    The old ``requires_sandbox: bool`` flag was removed when the
    Environment Router landed. Sandbox preflighting is now driven
    by ``config.environments`` allow-lists (the runtime preflights
    every env that has at least one assigned plugin), not by a
    self-declared flag in meta.yaml.

    ``dependencies`` (added 2026-05-05) is a list of pip-installable
    spec strings (e.g. ``"playwright>=1.40"``, ``"aiohttp~=3.9"``).
    Each plugin declares its own third-party deps so plugins behave
    "like an independent project". The ``krakey install`` CLI walks
    every plugin under BUILTIN_ROOT + WORKSPACE_ROOT, collects the
    union, and pip-installs in one shot.

    ``post_install`` (added 2026-05-06) is a list of secondary
    install commands run AFTER pip — for things pip can't drive
    (e.g. Playwright's ``playwright install chromium`` to download
    browser binaries). Each entry is
    ``{args: list[str], description: str, optional: bool=False}``.
    The token ``{python}`` inside ``args`` is replaced with
    ``sys.executable`` at run-time so commands always invoke the
    runtime's own interpreter.
    """
    name: str
    description: str
    components: list[ComponentMetadata] = field(default_factory=list)
    config_schema: list[dict[str, Any]] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    post_install: list[dict[str, Any]] = field(default_factory=list)
    source_path: Path | None = None


def load_plugin_meta(name: str) -> PluginMetadata | None:
    """Read one plugin's ``meta.yaml`` by name. Workspace overrides
    built-in. ``None`` if the plugin folder doesn't exist or its
    meta.yaml fails to parse.

    Used by Runtime to load by-name without scanning the rest:
    when ``config.plugins: [a, b, c]``, Runtime only opens those three
    meta.yaml files. Catalogue scanning ("list all installed") is the
    dashboard's job — see ``krakey/plugins/dashboard/services/plugin_catalogue.py``.
    """
    for root in (WORKSPACE_ROOT, BUILTIN_ROOT):
        meta_path = root / name / "meta.yaml"
        if not meta_path.exists():
            continue
        try:
            return parse_meta(meta_path)
        except Exception as e:  # noqa: BLE001
            print(
                f"warning: failed to parse {meta_path}: {e}; skipping",
                file=sys.stderr,
            )
            return None
    return None


def load_component(component: ComponentMetadata, ctx: "PluginContext") -> Any:
    """Lazily import + invoke one component's factory.

    ``ctx`` carries the per-plugin config + LLM resolutions; the
    factory may return ``None`` to opt out (e.g. unbound LLM purpose).
    """
    module = importlib.import_module(component.factory_module)
    factory = getattr(module, component.factory_attr)
    return factory(ctx)


def parse_meta(path: Path) -> PluginMetadata:
    """Parse one ``meta.yaml`` file into a ``PluginMetadata``. Public
    (no leading underscore) so the dashboard's catalogue scanner can
    reuse the same parsing logic instead of redeclaring the schema."""
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
    deps_raw = raw.get("dependencies") or []
    if not isinstance(deps_raw, list):
        raise ValueError("meta.yaml `dependencies:` must be a list")
    dependencies: list[str] = []
    for i, d in enumerate(deps_raw):
        if not isinstance(d, str) or not d.strip():
            raise ValueError(
                f"meta.yaml `dependencies[{i}]` must be a non-empty "
                "pip-installable string (e.g. 'playwright>=1.40')"
            )
        dependencies.append(d.strip())
    post_install_raw = raw.get("post_install") or []
    if not isinstance(post_install_raw, list):
        raise ValueError(
            "meta.yaml `post_install:` must be a list of "
            "{args: [...], description: str, optional?: bool}"
        )
    post_install: list[dict[str, Any]] = []
    for i, entry in enumerate(post_install_raw):
        post_install.append(_parse_post_install(entry, i))
    if "requires_sandbox" in raw:
        # Quietly tolerated for one release window — old plugin
        # meta.yamls in the wild had this field. Drop it from yours
        # at next edit.
        import sys as _sys
        print(
            f"meta.yaml {path}: `requires_sandbox` is no longer read; "
            "sandbox usage is now opt-in via config.environments.sandbox."
            "allowed_plugins. Remove the field to silence this warning.",
            file=_sys.stderr,
        )
    return PluginMetadata(
        name=str(raw["name"]),
        description=str(raw.get("description", "")),
        components=components,
        config_schema=list(schema),
        dependencies=dependencies,
        post_install=post_install,
        source_path=path,
    )


def _parse_post_install(entry: Any, index: int) -> dict[str, Any]:
    """Validate one ``post_install`` entry. Returns a dict shape
    ``{args: list[str], description: str, optional: bool}``."""
    if not isinstance(entry, dict):
        raise ValueError(
            f"meta.yaml `post_install[{index}]` must be a mapping "
            "with `args:` (and optional `description:`, `optional:`)"
        )
    args = entry.get("args")
    if (
        not isinstance(args, list)
        or not args
        or not all(isinstance(a, str) and a for a in args)
    ):
        raise ValueError(
            f"meta.yaml `post_install[{index}].args` must be a "
            "non-empty list of non-empty strings (argv-style)"
        )
    description = entry.get("description", "")
    if not isinstance(description, str):
        raise ValueError(
            f"meta.yaml `post_install[{index}].description` must "
            "be a string if provided"
        )
    optional = entry.get("optional", False)
    if not isinstance(optional, bool):
        raise ValueError(
            f"meta.yaml `post_install[{index}].optional` must be "
            "a boolean if provided"
        )
    return {
        "args":        [str(a) for a in args],
        "description": description,
        "optional":    optional,
    }


_KNOWN_COMPONENT_KINDS = {"modifier", "tool", "channel"}


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
    # Stash the rest (role already pulled, factory_* already pulled)
    known = {"kind", "role", "factory_module", "factory_attr",
             "llm_purposes"}
    extra = {k: v for k, v in c.items() if k not in known}
    return ComponentMetadata(
        kind=kind,
        role=str(c["role"]) if c.get("role") else None,
        factory_module=str(c["factory_module"]),
        factory_attr=str(c["factory_attr"]),
        llm_purposes=list(purposes),
        extra=extra,
    )
