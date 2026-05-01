"""Dashboard plugin — Web UI bundle (config editor + observation +
embedded chat).

Two component factories:
  * ``build_channel``: returns the chat channel AND, as a side effect,
    starts the dashboard's uvicorn server in a daemon thread (see
    ``threaded_server.py``). The server start happens at plugin
    registration time so it doesn't pollute the Channel ABC's
    ``start()/stop()`` hooks with non-channel work.
  * ``build_tool``: returns the ``web_chat_reply`` outbound
    tool. Shares the ``WebChatHistory`` instance with the channel
    via ``ctx.plugin_cache``.

``port=0`` in the per-plugin config short-circuits the server start
(used by tests so the channel + tool register without binding
a port).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from krakey.interfaces.plugin_context import PluginContext


_HISTORY_CACHE_KEY = "web_chat_history"


def build_channel(ctx: "PluginContext"):
    """Create the chat channel; start the dashboard server in a
    background thread (skipped when port=0)."""
    from krakey.plugins.dashboard.channel import WebChatChannel
    from krakey.plugins.dashboard.web_chat.history import WebChatHistory

    cfg = ctx.config or {}
    history_path = cfg.get(
        "history_path", "workspace/data/web_chat.jsonl",
    )
    host = cfg.get("host", "127.0.0.1")
    port = int(cfg.get("port", 8765))

    history = WebChatHistory(history_path)
    ctx.plugin_cache[_HISTORY_CACHE_KEY] = history

    channel = WebChatChannel()
    if port != 0:
        _start_dashboard_server(ctx, channel, history, host, port)
    return channel


def build_tool(ctx: "PluginContext"):
    """Reply tool — shares the WebChatHistory the sibling channel
    factory built."""
    from krakey.plugins.dashboard.tool import WebChatReplyTool

    history = ctx.plugin_cache.get(_HISTORY_CACHE_KEY)
    if history is None:
        raise RuntimeError(
            "dashboard.build_tool: WebChatHistory not in plugin_cache. "
            "build_channel must run first (meta.yaml component order)."
        )
    return WebChatReplyTool(history=history)


def _start_dashboard_server(ctx, channel, history, host: str, port: int) -> None:
    """Build the dashboard FastAPI app + start uvicorn in a daemon
    thread. Failures (port in use, etc.) log + leave the server
    absent — runtime continues without dashboard."""
    from krakey.plugins.dashboard.app_factory import (
        create_app as create_dashboard_app,
    )
    from krakey.plugins.dashboard.events import EventBroadcaster
    from krakey.plugins.dashboard.threaded_server import ThreadedDashboardServer

    runtime = ctx.services.get("runtime")
    if runtime is None:
        raise RuntimeError(
            "dashboard plugin needs services['runtime']; the runtime "
            "must expose itself in PluginContext.services."
        )

    config_path = getattr(ctx.deps, "config_path", None)
    plugin_configs_root = (
        getattr(ctx.deps, "plugin_configs_root", None) or "workspace/plugins"
    )

    def on_restart() -> None:
        runtime.log.hb("restart requested via dashboard — re-execing")
        os.execv(sys.executable, [sys.executable, *sys.argv])

    try:
        broadcaster = EventBroadcaster(runtime.events)
        app = create_dashboard_app(
            runtime=runtime,
            web_chat_history=history,
            on_user_message=channel.push_user_message,
            event_broadcaster=broadcaster,
            config_path=Path(config_path) if config_path else None,
            on_restart=on_restart,
            plugin_configs_root=Path(plugin_configs_root),
        )
        server = ThreadedDashboardServer(app, host=host, port=port)
        server.start()
        # Hand the server to the channel so channel.stop() (called by
        # buffer.stop_all() in Runtime.run()'s finally) can also stop
        # the server — closes WS frames cleanly + finishes in-flight
        # HTTP before the runtime loop tears down.
        channel.attach_server(server)
        runtime.log.hb(
            f"dashboard listening on http://{host}:{server.port}"
        )
    except OSError as e:
        runtime.log.runtime_error(
            f"dashboard failed to start (port {port} in use? {e}); "
            "runtime continues without dashboard"
        )
    except Exception as e:  # noqa: BLE001
        runtime.log.runtime_error(f"dashboard startup error: {e}")
