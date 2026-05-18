"""McpProxyTool — wraps a single remote MCP tool as a krakey Tool.

Each instance holds a reference to its ``ServerConnection`` AND to the
``_LoopThread`` singleton so ``execute()`` can marshal the async MCP call
onto the dedicated loop correctly.

Sync↔async bridge in execute()
-------------------------------
``execute()`` is ``async def`` — it is called by the heartbeat's own event
loop.  The MCP session lives on the *dedicated* loop, a different loop.  We
cannot simply ``await conn.call_tool(...)`` from the heartbeat loop because
the ``ClientSession`` and its subprocess transport are bound to the dedicated
loop.

Instead we:

1. Submit ``conn.call_tool(...)`` to the dedicated loop via
   ``asyncio.run_coroutine_threadsafe``, which returns a
   ``concurrent.futures.Future``.
2. Wrap that Future with ``asyncio.wrap_future(fut, loop=current_loop)``
   so it becomes an ``asyncio.Future`` that the heartbeat loop can
   ``await`` without blocking its thread.

This way:
- The MCP coroutine runs on the dedicated loop (session stays alive).
- The heartbeat loop is never blocked; it just awaits the cross-loop future.
- Timeouts are enforced inside ``conn.call_tool`` (per ``asyncio.wait_for``
  with ``timeout_s``) AND additionally via a ``fut.cancel()`` fallback in
  the ``asyncio.TimeoutError`` handler so we never hang.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from krakey.interfaces.tool import Tool
from krakey.models.stimulus import Stimulus

if TYPE_CHECKING:
    from krakey.plugins.mcp_connector.loop_thread import _LoopThread
    from krakey.plugins.mcp_connector.server_connection import ServerConnection

log = logging.getLogger(__name__)


class McpProxyTool(Tool):
    """A krakey Tool that proxies every call to a single MCP tool on a
    remote server.

    Naming: ``{tool_prefix}_{mcp_name}`` when the server config has a
    non-empty ``tool_prefix``; otherwise ``{server_id}_{mcp_name}``.
    """

    def __init__(
        self,
        *,
        server_id: str,
        tool_prefix: str,
        mcp_tool_name: str,
        description: str,
        parameters_schema: dict[str, Any],
        connection: "ServerConnection",
        loop_thread: "_LoopThread",
    ) -> None:
        prefix = tool_prefix if tool_prefix else server_id
        self._name = f"{prefix}_{mcp_tool_name}"
        self._description = (
            description or f"MCP tool '{mcp_tool_name}' on server '{server_id}'."
        )
        self._parameters_schema = parameters_schema or {"type": "object"}
        self._connection = connection
        self._mcp_tool_name = mcp_tool_name
        self._loop_thread = loop_thread
        # Cache timeout from server config for use in execute().
        self._timeout_s: float = float(
            connection._cfg.get("timeout_s", 30)  # noqa: SLF001
        )

    # --- Tool ABC ----------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._parameters_schema

    async def execute(self, intent: str, params: dict[str, Any]) -> Stimulus:
        """Delegate the call to the MCP session on the dedicated loop.

        The call is cross-loop: ``conn.call_tool`` must run on the dedicated
        loop, but this method is awaited on the heartbeat loop.  We bridge
        them with ``run_coroutine_threadsafe`` + ``wrap_future``.

        Any failure (timeout, dead server, MCP error, loop error) is
        converted into an error Stimulus — never raised — so the dispatcher
        and heartbeat loop remain unaffected.
        """
        try:
            coro = self._connection.call_tool(
                self._mcp_tool_name, params, timeout_s=self._timeout_s
            )
            # Submit to the dedicated loop; wrap so the heartbeat loop can await.
            concurrent_fut = self._loop_thread.submit(coro)
            try:
                caller_loop = asyncio.get_running_loop()
                asyncio_fut = asyncio.wrap_future(concurrent_fut, loop=caller_loop)
                text = await asyncio.wait_for(asyncio_fut, timeout=self._timeout_s)
            except asyncio.TimeoutError:
                concurrent_fut.cancel()
                return self._stim(
                    f"MCP tool '{self._mcp_tool_name}' timed out after "
                    f"{self._timeout_s:.0f}s."
                )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "mcp_connector: tool '%s' execute failed: %s",
                self._name,
                exc,
            )
            return self._stim(
                f"MCP tool '{self._mcp_tool_name}' call failed: {exc}"
            )
        return self._stim(text)

    # --- helpers -----------------------------------------------------------

    def _stim(self, content: str) -> Stimulus:
        return Stimulus(
            type="tool_feedback",
            source=f"tool:{self.name}",
            content=content,
            timestamp=datetime.now(),
            adrenalin=False,
        )
