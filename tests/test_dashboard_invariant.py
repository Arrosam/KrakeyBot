"""CLAUDE.md additive-plugin invariant for the dashboard plugin.

Locks the two fallback paths in dashboard/__init__.py that the
earlier review flagged: the build_tool factory must build a fresh
WebChatHistory when the channel didn't run, and _start_dashboard_server
must log + return when services['runtime'] is missing rather than
raising.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from krakey.plugins.dashboard import build_tool, build_channel


def _ctx(*, history_path: str | None = None, services: dict | None = None,
          plugin_cache: dict | None = None):
    """Minimal stand-in for PluginContext so we can drive the
    dashboard's factories without spinning up the whole runtime."""
    cfg: dict = {}
    if history_path is not None:
        cfg["history_path"] = history_path
    return SimpleNamespace(
        plugin_name="dashboard",
        config=cfg,
        services=services or {},
        plugin_cache=plugin_cache if plugin_cache is not None else {},
        deps=SimpleNamespace(config_path=None, plugin_configs_root=None),
    )


def test_build_tool_works_without_channel(tmp_path):
    """When the channel hasn't populated the history cache (component
    disabled, load order shuffled), build_tool must NOT raise — it
    builds a fresh WebChatHistory pointed at the same JSONL."""
    history_path = str(tmp_path / "web_chat.jsonl")
    ctx = _ctx(history_path=history_path)
    tool = build_tool(ctx)
    # The tool registers fine and the freshly-built history was
    # cached for subsequent calls (e.g. a later channel build).
    assert tool is not None
    assert "web_chat_history" in ctx.plugin_cache


def test_build_channel_skips_server_when_runtime_missing(tmp_path, caplog):
    """services['runtime'] missing must NOT raise — the channel
    still registers; the HTTP server is silently skipped with a
    warning so the runtime keeps booting."""
    history_path = str(tmp_path / "web_chat.jsonl")
    ctx = _ctx(history_path=history_path, services={})  # no "runtime"
    # Use a non-zero port so _start_dashboard_server actually runs
    # (port=0 is the test short-circuit that skips it entirely).
    ctx.config["port"] = 18765
    with caplog.at_level(logging.WARNING):
        channel = build_channel(ctx)
    assert channel is not None  # registered fine
    # Channel doesn't have a server attached.
    assert getattr(channel, "_server", None) is None
    # And we left a breadcrumb so the operator knows why no UI.
    assert any(
        "services['runtime']" in r.message for r in caplog.records
    )


def test_build_channel_with_port_zero_skips_server(tmp_path):
    """The historical port=0 short-circuit still works — channel
    registers, no server, no warning needed."""
    history_path = str(tmp_path / "web_chat.jsonl")
    ctx = _ctx(history_path=history_path, services={"runtime": object()})
    ctx.config["port"] = 0
    channel = build_channel(ctx)
    assert channel is not None
    assert getattr(channel, "_server", None) is None
