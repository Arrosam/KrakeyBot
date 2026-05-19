"""mcp_connector plugin — bridges MCP servers into krakey Tools.

The plugin loader imports this module and calls ``build_tool(ctx)``.
That factory returns a ``list[Tool]`` — one McpProxyTool per MCP tool
advertised by each enabled, reachable server at startup.

Import path (pinned by tests):
    from krakey.plugins.mcp_connector import build_tool
"""
from __future__ import annotations

from krakey.plugins.mcp_connector.plugin import build_tool

__all__ = ["build_tool"]
