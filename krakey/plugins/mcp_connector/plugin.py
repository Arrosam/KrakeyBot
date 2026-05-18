"""mcp_connector plugin factory.

``build_tool(ctx)`` is the public entry point called by the plugin loader.
It is synchronous and NEVER raises; failures are logged and contribute zero
tools.

Sync↔async bridge — single long-lived loop thread
--------------------------------------------------
All MCP I/O is async and uses stdio subprocess transports on Windows.
Stdio transports bind to the event loop they are created on and require
that loop to keep running (pumping I/O) while the session is alive.  The
old design of ``asyncio.run(...)`` (or ``run_until_complete``) stops the
loop after each call, which freezes the subprocess transport so the next
``await`` hangs forever (the IOCP completion queue is never drained again).

Solution: ``loop_thread.get_loop_thread()`` returns a singleton
``_LoopThread`` that owns ONE ``asyncio.ProactorEventLoop`` (on Windows)
running continuously via ``loop.run_forever()`` in a daemon thread.

Discovery uses ``lt.run_sync(coro, timeout=server_timeout)`` which calls
``asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=...)`` so
the calling thread blocks until discovery is done, but the loop keeps
running.  Each ``ServerConnection`` is opened on that same loop and stays
there for the runtime lifetime.

See ``loop_thread.py`` for the _LoopThread implementation.
See ``proxy_tool.py`` for the execute()-time bridge (wrap_future).
"""
from __future__ import annotations

import logging
from typing import Any

from krakey.interfaces.tool import Tool

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_tool(ctx) -> list[Tool]:
    """Factory called by the plugin loader.  Returns a flat list of
    McpProxyTool instances across all enabled configured servers.

    NEVER raises — any failure is logged and contributes zero tools.
    """
    # Guard: if mcp is not importable at all, bail gracefully.
    try:
        import mcp  # noqa: F401
    except ImportError:
        log.warning(
            "mcp_connector: 'mcp' package not importable; "
            "no MCP tools will be registered."
        )
        return []

    cfg = _parse_config(ctx.config)
    if not cfg["servers"]:
        return []

    try:
        lt = _get_loop_thread()
    except Exception as exc:  # noqa: BLE001
        log.error(
            "mcp_connector: could not start background loop thread: %s — "
            "no MCP tools will be registered.",
            exc,
        )
        return []

    tools = _discover_all_sync(cfg, lt)
    log.info(
        "mcp_connector: registered %d tool(s) across %d server(s).",
        len(tools),
        len(cfg["servers"]),
    )
    return tools


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def _parse_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise the plugin config dict into a well-typed structure."""
    servers_raw = raw.get("servers")
    if not isinstance(servers_raw, list):
        servers_raw = []

    servers: list[dict[str, Any]] = []
    for entry in servers_raw:
        if not isinstance(entry, dict):
            log.debug("mcp_connector: skipping non-dict server entry: %r", entry)
            continue
        if not entry.get("id"):
            log.debug("mcp_connector: skipping server entry with no 'id': %r", entry)
            continue
        if not entry.get("enabled", True):
            log.debug(
                "mcp_connector: server '%s' is disabled — skipping.",
                entry["id"],
            )
            continue
        servers.append(entry)

    return {
        "servers": servers,
        "reconnect": bool(raw.get("reconnect", True)),
    }


# ---------------------------------------------------------------------------
# Loop-thread accessor (thin wrapper so build_tool stays readable)
# ---------------------------------------------------------------------------

def _get_loop_thread():
    """Lazily create and return the singleton _LoopThread."""
    from krakey.plugins.mcp_connector.loop_thread import get_loop_thread
    return get_loop_thread()


# ---------------------------------------------------------------------------
# Synchronous discovery (runs on the dedicated loop via run_sync)
# ---------------------------------------------------------------------------

def _discover_all_sync(cfg: dict[str, Any], lt) -> list[Tool]:
    """Connect to every enabled server and collect their tools synchronously.

    Each server is handled independently with a per-server timeout.
    Failures are logged; other servers are unaffected.
    """
    from krakey.plugins.mcp_connector.proxy_tool import McpProxyTool
    from krakey.plugins.mcp_connector.server_connection import ServerConnection

    all_tools: list[Tool] = []
    reconnect: bool = cfg["reconnect"]

    for server_cfg in cfg["servers"]:
        server_id: str = server_cfg["id"]
        timeout_s: float = float(server_cfg.get("timeout_s", 30))
        tool_prefix: str = server_cfg.get("tool_prefix") or ""

        conn = ServerConnection(server_cfg, reconnect=reconnect)

        # Submit the entire open+list_tools operation to the dedicated loop
        # and block until it completes OR the per-server timeout fires.
        try:
            mcp_tools: list[Any] = lt.run_sync(conn.open(), timeout=timeout_s)
        except TimeoutError:
            log.warning(
                "mcp_connector: server '%s' timed out after %.0fs during "
                "discovery — skipping.",
                server_id,
                timeout_s,
            )
            # Best-effort close on the dedicated loop (fire-and-forget).
            try:
                lt.submit(conn.close())
            except Exception:  # noqa: BLE001
                pass
            continue
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "mcp_connector: server '%s' failed to connect: %s — skipping.",
                server_id,
                exc,
            )
            try:
                lt.submit(conn.close())
            except Exception:  # noqa: BLE001
                pass
            continue

        for mcp_tool in mcp_tools:
            tool_name: str = mcp_tool.name
            description: str = mcp_tool.description or ""
            schema: dict[str, Any] = (
                dict(mcp_tool.inputSchema) if mcp_tool.inputSchema else {}
            )

            proxy = McpProxyTool(
                server_id=server_id,
                tool_prefix=tool_prefix,
                mcp_tool_name=tool_name,
                description=description,
                parameters_schema=schema,
                connection=conn,
                loop_thread=lt,
            )
            all_tools.append(proxy)
            log.debug(
                "mcp_connector: registered tool '%s' from server '%s'.",
                proxy.name,
                server_id,
            )

    return all_tools
