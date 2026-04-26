"""``PluginContext`` — what every plugin's factory sees.

Passed to ``build_<component>(ctx)`` for every plugin kind
(reflect / tentacle / sensory). Wraps the runtime deps with
**plugin-scoped helpers**:

  * ``ctx.get_llm(purpose_name)`` — resolves the user's
    ``workspace/plugins/<plugin>/config.yaml`` ``llm_purposes:``
    entry for the named purpose to a concrete ``LLMClient``, or
    returns ``None`` if the user hasn't bound that purpose to a tag.
    The plugin decides what to do with ``None`` (skip itself /
    degrade gracefully / log loud).
  * ``ctx.config`` — the parsed contents of the plugin's own
    ``config.yaml`` (per-plugin folder under ``workspace/plugins/``).
    Plugin code reads its own settings from here and **never** sees
    the central config.yaml — keeps plugin code one step removed
    from API keys + provider configs.
  * ``ctx.services`` — Runtime-built resources whitelisted for
    plugin use (gm, kb_registry, embedder, web_chat_history, ...).
  * ``ctx.plugin_cache`` — per-plugin scratch dict for sharing
    instances across multi-component plugins (e.g. telegram's
    sensory + tentacle share an HttpTelegramClient via this).
  * ``ctx.deps`` — escape hatch to ``RuntimeDeps`` for plugins that
    truly need it. Reading ``deps.config.llm.providers`` from a
    plugin is allowed but discouraged — by convention, plugins
    shouldn't poke at provider bindings.

Built by ``Runtime._register_plugins_from_config`` per plugin.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from src.llm.client import LLMClient
    from src.main import RuntimeDeps

_log = logging.getLogger(__name__)


@dataclass
class PluginContext:
    """Per-plugin context handed to ``build_<component>(ctx)``."""
    deps: "RuntimeDeps"
    plugin_name: str
    config: dict[str, Any] = field(default_factory=dict)
    # Resolved ``purpose_name → LLMClient`` map. Populated by the
    # registrar before the factory is called; an entry is absent
    # whenever the user hasn't bound that purpose to a tag (or the
    # tag references a missing provider).
    llms: dict[str, "LLMClient"] = field(default_factory=dict)
    # Whitelisted Runtime-built resources (gm, kb_registry, embedder,
    # buffer, web_chat_history, build_code_runner, ...). Populated by
    # Runtime when building the ctx so plugins don't have to grab
    # unrestricted Runtime references. Same shape as the legacy
    # plugin loader's `deps` dict — keeps existing factories' lookup
    # patterns familiar.
    services: dict[str, Any] = field(default_factory=dict)
    # Shared mutable storage scoped to a single plugin (NOT across
    # plugins). Components of the same plugin (e.g. telegram's
    # sensory + tentacle that share an HttpTelegramClient) can stash
    # an instance here in the first factory call and read it in the
    # next, so multi-component plugins don't need module-level
    # singletons. Reset per-plugin during registration.
    plugin_cache: dict[str, Any] = field(default_factory=dict)

    def get_llm(self, purpose: str) -> "LLMClient | None":
        return self.llms.get(purpose)


def load_plugin_config(plugin_name: str, root: Path | str) -> dict[str, Any]:
    """Read ``<root>/<plugin_name>/config.yaml`` if present.

    Missing file → empty dict (the plugin operates with whatever
    defaults its code defines). Malformed YAML → empty dict + log
    warning, so a typo doesn't crash startup.
    """
    path = Path(root) / plugin_name / "config.yaml"
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        _log.warning(
            "plugin config %s parse failed: %s; treating as empty",
            path, e,
        )
        return {}
    if not isinstance(raw, dict):
        _log.warning(
            "plugin config %s top-level is not a mapping; treating as empty",
            path,
        )
        return {}
    return raw
