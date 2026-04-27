"""Dashboard plugin — Web UI bundle (config editor + observation +
embedded chat).

Two components:
  * sensory: hosts the FastAPI/uvicorn server AND pushes inbound chat
    messages into the runtime queue. See ``sensory.py``.
  * tentacle: the ``web_chat_reply`` outbound tentacle. See
    ``tentacle.py``.

Both share one ``WebChatHistory`` instance built in ``build_sensory``
and stashed in ``ctx.plugin_cache`` so the tentacle factory can read
it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.interfaces.plugin_context import PluginContext


_HISTORY_CACHE_KEY = "web_chat_history"


def build_sensory(ctx: "PluginContext"):
    """Create the dashboard server + chat sensory. Owns the
    WebChatHistory and stashes it for the sibling tentacle factory."""
    from pathlib import Path

    from src.plugins.dashboard.sensory import DashboardSensory
    from src.plugins.dashboard.web_chat.history import WebChatHistory

    cfg = ctx.config or {}
    history_path = cfg.get(
        "history_path", "workspace/data/web_chat.jsonl",
    )
    host = cfg.get("host", "127.0.0.1")
    port = int(cfg.get("port", 8765))

    history = WebChatHistory(history_path)
    ctx.plugin_cache[_HISTORY_CACHE_KEY] = history

    runtime = ctx.services.get("runtime")
    if runtime is None:
        raise RuntimeError(
            "dashboard plugin needs services['runtime']; the runtime "
            "must expose itself in PluginContext.services."
        )

    plugin_configs_root = (
        getattr(ctx.deps, "plugin_configs_root", None) or "workspace/plugins"
    )
    return DashboardSensory(
        runtime=runtime,
        history=history,
        host=host,
        port=port,
        plugin_configs_root=Path(plugin_configs_root),
    )


def build_tentacle(ctx: "PluginContext"):
    """Reply tentacle — shares the WebChatHistory instance the sibling
    sensory factory created."""
    from src.plugins.dashboard.tentacle import WebChatReplyTentacle

    history = ctx.plugin_cache.get(_HISTORY_CACHE_KEY)
    if history is None:
        raise RuntimeError(
            "dashboard.build_tentacle: WebChatHistory not in plugin_cache. "
            "build_sensory must run first (meta.yaml component order)."
        )
    return WebChatReplyTentacle(history=history)
