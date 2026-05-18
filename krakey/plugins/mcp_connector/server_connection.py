"""ServerConnection — manages the lifetime of one MCP server session.

ALL coroutines here MUST run on the dedicated loop owned by ``_LoopThread``
(see ``loop_thread.py``).  They are never awaited directly by plugin.py or
proxy_tool.py; instead those callers submit them via
``asyncio.run_coroutine_threadsafe`` so they run on the correct loop and the
stdio subprocess transports (which bind to the loop they were created on)
stay alive and properly pumped.

Responsibilities
----------------
- Open the transport-specific connection (stdio/SSE/HTTP).
- Keep the ClientSession alive for the runtime lifetime.
- Expose ``list_tools()`` and ``call_tool()`` as coroutines run on the
  dedicated loop.
- Reconnect transparently on call failure when ``reconnect=True`` —
  close the dead session and re-open, then retry once.

Timeouts
--------
Timeouts are enforced by the *caller* (plugin.py for discovery, proxy_tool.py
for execute) via ``fut.result(timeout=...)`` or
``asyncio.wait_for(..., timeout=...)``.  Individual operations inside
``call_tool`` are additionally wrapped in ``asyncio.wait_for`` so a hung
MCP server cannot block the dedicated loop indefinitely.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from typing import Any

log = logging.getLogger(__name__)


class ServerConnection:
    """Async connection to a single MCP server.

    All methods are ``async def`` and MUST be awaited from coroutines
    running on the dedicated loop thread.

    Usage (from dedicated-loop coroutines):
        conn = ServerConnection(cfg, reconnect=True)
        tools = await conn.open()           # opens transport, returns tool list
        text  = await conn.call_tool(...)   # proxied by McpProxyTool.execute
        await conn.close()                  # best-effort cleanup
    """

    def __init__(self, server_cfg: dict[str, Any], *, reconnect: bool) -> None:
        # Import here so ImportError from absent 'mcp' package surfaces at
        # connection time, not at module import time.
        from mcp import ClientSession  # noqa: F401  # validate import is possible
        from mcp.types import TextContent  # noqa: F401

        self._cfg = server_cfg
        self._reconnect = reconnect
        self._session: Any | None = None   # mcp.ClientSession
        self._stack: AsyncExitStack | None = None
        self._healthy = False

    # ------------------------------------------------------------------
    # Public API (all coroutines; run on dedicated loop)
    # ------------------------------------------------------------------

    async def open(self) -> list[Any]:
        """Open transport, initialize session, return raw mcp.types.Tool list.

        Raises on unrecoverable failure so the caller can skip this server
        without affecting others.
        """
        await self._connect()
        result = await self._session.list_tools()
        return result.tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any], *, timeout_s: float = 30.0
    ) -> str:
        """Call a named tool on the server and return its text output.

        On first failure when reconnect=True: closes the dead session,
        re-opens (within timeout_s), retries once.  If the retry also fails,
        propagates the exception so the caller converts it to an error Stimulus.

        Every blocking operation is wrapped in asyncio.wait_for to prevent
        a hung server from stalling the dedicated loop indefinitely.
        """
        try:
            text = await asyncio.wait_for(
                self._do_call(name, arguments), timeout=timeout_s
            )
            return text
        except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
            if not self._reconnect:
                raise

            log.warning(
                "mcp_connector: server '%s' call failed (%s); reconnecting.",
                self._cfg.get("id", "?"),
                exc,
            )
            try:
                await asyncio.wait_for(
                    self._reconnect_session(), timeout=timeout_s
                )
            except Exception as reconnect_exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Reconnect to '{self._cfg.get('id', '?')}' failed: "
                    f"{reconnect_exc}"
                ) from exc

            # One retry after reconnect.
            return await asyncio.wait_for(
                self._do_call(name, arguments), timeout=timeout_s
            )

    async def close(self) -> None:
        """Tear down the session and transport (best-effort)."""
        self._healthy = False
        self._session = None
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._stack = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        """Open transport and initialise a fresh ClientSession."""
        from mcp import ClientSession

        stack = AsyncExitStack()
        try:
            read, write = await self._enter_transport(stack)
            session = ClientSession(read, write)
            await stack.enter_async_context(session)
            await session.initialize()
        except Exception:
            await stack.aclose()
            raise

        # Swap in atomically (close previous stack if any).
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:  # noqa: BLE001
                pass
        self._stack = stack
        self._session = session
        self._healthy = True

    async def _reconnect_session(self) -> None:
        """Close the dead session and open a fresh one."""
        await self.close()
        await self._connect()
        # Re-list tools to keep the session primed; actual tool wrappers
        # in the ToolRegistry already exist — we just restore the session.
        if self._reconnect:
            try:
                await self._session.list_tools()
            except Exception:  # noqa: BLE001
                pass  # best-effort; call_tool will surface real errors

    async def _do_call(self, name: str, arguments: dict[str, Any]) -> str:
        """Send the call_tool RPC and extract text from the result."""
        from mcp.types import TextContent

        if self._session is None:
            raise RuntimeError("Session not open.")
        result = await self._session.call_tool(name, arguments or {})
        return _extract_text(result.content, name)

    async def _enter_transport(
        self, stack: AsyncExitStack
    ) -> tuple[Any, Any]:
        """Select the right transport from cfg and enter it on *stack*."""
        transport = self._cfg.get("transport", "stdio")

        if transport == "stdio":
            return await self._enter_stdio(stack)
        elif transport == "sse":
            return await self._enter_sse(stack)
        elif transport == "http":
            return await self._enter_http(stack)
        else:
            raise ValueError(
                f"Unknown transport '{transport}' for server "
                f"'{self._cfg.get('id', '?')}'. Expected stdio/sse/http."
            )

    async def _enter_stdio(self, stack: AsyncExitStack) -> tuple[Any, Any]:
        from mcp.client.stdio import StdioServerParameters, stdio_client

        cmd_list: list[str] = self._cfg.get("command", [])
        if not cmd_list:
            raise ValueError(
                f"Server '{self._cfg.get('id', '?')}' transport=stdio "
                f"requires 'command' list."
            )

        extra_env: dict[str, str] = self._cfg.get("env") or {}
        merged_env: dict[str, str] | None = None
        if extra_env:
            merged_env = {**os.environ, **extra_env}

        params = StdioServerParameters(
            command=cmd_list[0],
            args=cmd_list[1:],
            env=merged_env,
        )
        read, write = await stack.enter_async_context(stdio_client(params))
        return read, write

    async def _enter_sse(self, stack: AsyncExitStack) -> tuple[Any, Any]:
        from mcp.client.sse import sse_client

        url: str = self._cfg.get("url", "")
        if not url:
            raise ValueError(
                f"Server '{self._cfg.get('id', '?')}' transport=sse "
                f"requires 'url'."
            )
        timeout_s: float = float(self._cfg.get("timeout_s", 30))
        read, write = await stack.enter_async_context(
            sse_client(url, timeout=timeout_s)
        )
        return read, write

    async def _enter_http(self, stack: AsyncExitStack) -> tuple[Any, Any]:
        from mcp.client.streamable_http import streamablehttp_client

        url: str = self._cfg.get("url", "")
        if not url:
            raise ValueError(
                f"Server '{self._cfg.get('id', '?')}' transport=http "
                f"requires 'url'."
            )
        timeout_s: float = float(self._cfg.get("timeout_s", 30))
        # streamablehttp_client returns a 3-tuple: (read, write, get_session_id)
        read, write, _get_session_id = await stack.enter_async_context(
            streamablehttp_client(url, timeout=timeout_s)
        )
        return read, write


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(content_blocks: list[Any], tool_name: str) -> str:
    """Pull text out of a CallToolResult content list.

    MCP content can be TextContent, ImageContent, AudioContent,
    ResourceLink, or EmbeddedResource.  We extract text from whatever
    we can and join; if nothing textual is found we say so rather than
    returning empty string, which would look like a silent failure.
    """
    from mcp.types import TextContent

    parts: list[str] = []
    for block in content_blocks:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif hasattr(block, "text"):
            parts.append(str(block.text))
        else:
            block_type = getattr(block, "type", type(block).__name__)
            parts.append(f"[{block_type} content — not representable as text]")

    if not parts:
        return f"Tool '{tool_name}' returned no content."
    return "\n".join(parts)
