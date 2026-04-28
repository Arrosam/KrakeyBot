"""``PluginContext`` — what every plugin's factory sees.

Passed to ``build_<component>(ctx)`` for every plugin kind
(reflect / tentacle / sensory). Carries:

  * ``ctx.config`` — the parsed contents of the plugin's own
    ``workspace/plugins/<plugin>/config.yaml``. Plugin code reads
    ALL of its own settings from here, including its
    ``llm_purposes:`` map (purpose name → tag name). Runtime never
    inspects this dict — it's purely the plugin's data.
  * ``ctx.get_llm_for_tag(tag_name)`` — resolves a tag name to a
    concrete ``LLMClient``, or returns ``None`` if the tag is
    undefined / unbound. Plugin reads its config to find the tag
    name, then asks for the client. **API key isolation**: providers
    + secrets stay in Runtime; plugin only ever holds the resolved
    client object.
  * ``ctx.services`` — Runtime-built resources whitelisted for
    plugin use (gm, kb_registry, embedder, runtime, ...).
  * ``ctx.plugin_cache`` — per-plugin scratch dict for sharing
    instances across multi-component plugins (e.g. telegram's
    sensory + tentacle share an HttpTelegramClient via this).
  * ``ctx.deps`` — escape hatch to ``RuntimeDeps`` for plugins that
    truly need it. Reading ``deps.config.llm.providers`` from a
    plugin breaks API-key isolation; don't.

Built by ``Runtime._register_plugins_from_config`` per plugin.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from krakey.llm.client import LLMClient
    from krakey.main import RuntimeDeps

_log = logging.getLogger(__name__)


@dataclass
class PluginContext:
    """Per-plugin context handed to ``build_<component>(ctx)``."""
    deps: "RuntimeDeps"
    plugin_name: str
    config: dict[str, Any] = field(default_factory=dict)
    # Whitelisted Runtime-built resources (gm, kb_registry, embedder,
    # buffer, runtime, build_code_runner, ...). Populated by
    # Runtime when building the ctx so plugins don't have to grab
    # unrestricted Runtime references.
    services: dict[str, Any] = field(default_factory=dict)
    # Shared mutable storage scoped to a single plugin (NOT across
    # plugins). Components of the same plugin (e.g. telegram's
    # sensory + tentacle that share an HttpTelegramClient) can stash
    # an instance here in the first factory call and read it in the
    # next, so multi-component plugins don't need module-level
    # singletons. Reset per-plugin during registration.
    plugin_cache: dict[str, Any] = field(default_factory=dict)

    def get_llm_for_tag(self, tag_name: str | None) -> "LLMClient | None":
        """Resolve a tag name to a concrete ``LLMClient``.

        The plugin reads its own ``config.yaml`` to find which tag the
        user bound (typically under ``llm_purposes: { my_purpose:
        my_tag }``), then calls this method with the tag name. Returns
        ``None`` for: missing tag name, undefined tag, malformed tag,
        unknown provider — all the failure modes of
        ``resolve_llm_for_tag``.

        Plugins NEVER see provider configs / API keys; the resolved
        ``LLMClient`` is the only thing that crosses the boundary.
        """
        if not tag_name:
            return None
        from krakey.llm.resolve import resolve_llm_for_tag
        return resolve_llm_for_tag(
            self.deps.config, tag_name, self.deps.llm_clients_by_tag,
        )


def load_plugin_config(plugin_name: str, root: Path | str) -> dict[str, Any]:
    """Read ``<root>/<plugin_name>/config.yaml`` if present.

    Missing file → empty dict (the plugin operates with whatever
    defaults its code defines). Malformed YAML → empty dict + log
    warning, so a typo doesn't crash startup.

    Used by both Runtime (to pre-load ``ctx.config`` as a convenience)
    AND by plugins that want to re-read their own config later.
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
