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


def _restart_self(runtime) -> None:
    """Spawn a fresh copy of the current process and exit this one.

    Why not ``os.execv``: on Windows, Python implements ``execv`` as
    spawn-then-exit; the parent dies before the child has fully
    attached to the console / event loop, and in practice the
    dashboard-driven restart leaves the user with "Krakey shut down"
    but no replacement running. ``subprocess.Popen`` followed by
    ``os._exit`` is reliable on both Windows and POSIX, and
    ``CREATE_NEW_PROCESS_GROUP`` makes the child independent of the
    parent's group so a Ctrl+C aimed at the dying parent doesn't
    propagate to the new instance.

    Launch mode matters: when the user starts with ``python -m
    krakey`` (the documented entry point), ``sys.argv[0]`` is the
    resolved path to ``krakey/__main__.py``, but re-running that path
    directly with the interpreter doesn't restore the package-import
    machinery. Re-spawn via ``-m`` when we detect that, so things
    like ``from krakey.runtime import ...`` keep working.
    """
    import subprocess

    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None) if main is not None else None
    if spec is not None and spec.name:
        # spec.name looks like "krakey.__main__" when launched via
        # ``python -m krakey`` — strip the trailing ``.__main__``
        # so the re-spawn ``-m <pkg>`` matches the original invocation.
        mod_name = spec.name
        if mod_name.endswith(".__main__"):
            mod_name = mod_name[: -len(".__main__")]
        args = [sys.executable, "-m", mod_name, *sys.argv[1:]]
    else:
        args = [sys.executable, *sys.argv]

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        subprocess.Popen(args, creationflags=creationflags, close_fds=False)
    except Exception as e:  # noqa: BLE001
        runtime.log.runtime_error(f"dashboard: restart spawn failed: {e}")
        return  # don't kill the parent if the child couldn't start
    # Tiny pause so the child has time to inherit stdio handles before
    # the parent tears them down. Empirically 100ms is enough on
    # Windows; cheap insurance.
    import time
    time.sleep(0.1)
    os._exit(0)


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
    """Reply tool — shares the ``WebChatHistory`` the sibling channel
    factory built. If the channel didn't run (disabled in central
    config, or load order shuffled), build a fresh ``WebChatHistory``
    pointed at the same JSONL: the file is the single source of
    truth, so a second instance reading it stays consistent. Cache
    it so a later channel-factory call sees the same instance.

    Per CLAUDE.md the runtime must keep working with any plugin /
    component disabled — this branch is the additive-fallback for
    "channel disabled but tool enabled".
    """
    from krakey.plugins.dashboard.tool import WebChatReplyTool
    from krakey.plugins.dashboard.web_chat.history import WebChatHistory

    history = ctx.plugin_cache.get(_HISTORY_CACHE_KEY)
    if history is None:
        cfg = ctx.config or {}
        history_path = cfg.get(
            "history_path", "workspace/data/web_chat.jsonl",
        )
        history = WebChatHistory(history_path)
        ctx.plugin_cache[_HISTORY_CACHE_KEY] = history
    return WebChatReplyTool(history=history)


def _start_dashboard_server(ctx, channel, history, host: str, port: int) -> None:
    """Build the dashboard FastAPI app + start uvicorn in a daemon
    thread. Failures (port in use, etc.) log + leave the server
    absent — runtime continues without dashboard."""
    from krakey.plugins.dashboard.app_factory import (
        create_app as create_dashboard_app,
    )
    from krakey.plugins.dashboard.auth import load_or_create_token
    from krakey.plugins.dashboard.events import EventBroadcaster
    from krakey.plugins.dashboard.log_capture import LogCapture
    from krakey.plugins.dashboard.threaded_server import ThreadedDashboardServer

    runtime = ctx.services.get("runtime")
    if runtime is None:
        # Per CLAUDE.md the runtime must keep working with any plugin
        # missing or partially loaded. We can't bring up the dashboard
        # without a runtime to introspect, but we shouldn't take the
        # rest of the runtime down with us — the channel + tool stay
        # registered, we just skip the HTTP UI.
        import logging
        logging.getLogger(__name__).warning(
            "dashboard plugin: services['runtime'] missing; skipping "
            "dashboard server start (channel + tool register without UI)"
        )
        return

    config_path = getattr(ctx.deps, "config_path", None)
    plugin_configs_root = (
        getattr(ctx.deps, "plugin_configs_root", None) or "workspace/plugins"
    )

    def on_restart() -> None:
        runtime.log.hb("restart requested via dashboard — re-execing")
        _restart_self(runtime)

    # Token sits next to the chat history on disk so the dashboard's
    # data files share a directory and `history_path` overrides naturally
    # carry the token with them.
    cfg = ctx.config or {}
    history_path = cfg.get("history_path", "workspace/data/web_chat.jsonl")
    token_path = Path(history_path).parent / "dashboard.token"
    auth_token = load_or_create_token(token_path)

    # Capture stdout/stderr so the Log tab can show live runtime output.
    # Force-enable runtime ANSI colours so the Log tab can render
    # category-tinted lines even when the daemon's stdout is captured
    # to a file (the colors helper auto-disables on non-TTY by default).
    # Side effect: the on-disk daemon log gets ANSI escapes — fine in
    # any modern terminal (Windows Terminal, cmd with VT enabled,
    # macOS / Linux). Set NO_COLOR=1 to opt out.
    if not os.environ.get("NO_COLOR"):
        try:
            from krakey.runtime.console import colors as _colors
            _colors._ENABLED = True
        except Exception:  # noqa: BLE001
            pass
    log_capture = LogCapture()
    log_capture.install()

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
            auth_token=auth_token,
            log_capture=log_capture,
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
        # Always log the one-click URL by default — this is a personal
        # local daemon and the log file lives on the user's own disk
        # next to the token file anyway. ``krakey start`` redirects
        # stdout to that log, so without the URL in the log the user
        # has no way to see what to open. Set
        # ``KRAKEY_REDACT_TOKEN_LOG=1`` to opt into redaction (for
        # shipping logs to a remote aggregator etc.).
        url_full = f"http://{host}:{server.port}/?token={auth_token}"
        if os.environ.get("KRAKEY_REDACT_TOKEN_LOG"):
            url_redacted = (
                f"http://{host}:{server.port}/?token=<see {token_path}>"
            )
            runtime.log.hb(f"dashboard URL: {url_redacted}")
            runtime.log.hb(
                "dashboard token redacted from logs (KRAKEY_REDACT_-"
                "TOKEN_LOG=1); read the token file directly"
            )
        else:
            runtime.log.hb(f"dashboard URL (one-click): {url_full}")
    except OSError as e:
        runtime.log.runtime_error(
            f"dashboard failed to start (port {port} in use? {e}); "
            "runtime continues without dashboard"
        )
    except Exception as e:  # noqa: BLE001
        runtime.log.runtime_error(f"dashboard startup error: {e}")
