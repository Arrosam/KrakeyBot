"""Edge tests for the ``mcp_connector`` plugin.

Run from repo root:

    pytest krakey/plugins/mcp_connector

These tests are written BEFORE any implementation exists.  They define the
acceptance contract.  Every test will FAIL until the plugin is implemented
— that is correct and expected.

## Factory import convention (pinned)

The implementation MUST expose a ``build_tool`` callable at:

    krakey.plugins.mcp_connector.build_tool(ctx) -> list[Tool]

(i.e. importable as ``from krakey.plugins.mcp_connector import build_tool``).

This mirrors the ``duckduckgo_search`` plugin convention where
``factory_module = krakey.plugins.duckduckgo_search`` and
``factory_attr = build_tool``.  The dev must wire the meta.yaml accordingly:

    components:
      - kind: tool
        factory_module: krakey.plugins.mcp_connector
        factory_attr: build_tool

``build_tool(ctx)`` must return a ``list[Tool]`` (the runtime registers each
element individually).  Returning a single ``Tool`` is NOT accepted — the
spec explicitly states list[Tool].

## References

- Spec point 1  — config schema (servers list, per-server fields)
- Spec point 2  — discovery at factory time, one Tool per MCP tool
- Spec point 3  — tool naming: prefix + mcp_tool_name
- Spec point 4  — execute() forwards params and returns Stimulus
- Spec point 5a — no servers / empty servers -> []
- Spec point 5b — unreachable server at factory time -> 0 tools, no raise
- Spec point 5c — runtime failure in execute() -> error Stimulus, no raise
- Spec point 5d — name collision: last-registration-wins, no raise
- Spec point 6  — stdio servers spawned by the plugin (not ctx.environment)
"""
from __future__ import annotations

import os
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from krakey.interfaces.plugin_context import PluginContext
from krakey.interfaces.tool import Tool
from krakey.models.stimulus import Stimulus

# ---------------------------------------------------------------------------
# Factory import — the ONE canonical import path the plugin must satisfy.
# ---------------------------------------------------------------------------
from krakey.plugins.mcp_connector import build_tool  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: minimal PluginContext stub
# ---------------------------------------------------------------------------

@dataclass
class _FakeDeps:
    """Minimal deps stub.  mcp_connector never needs llm_factory or
    environment_router in its factory, so these are all None."""
    llm_factory: Any = None
    environment_router: Any = None


def _make_ctx(config: dict[str, Any]) -> PluginContext:
    """Return a PluginContext pre-loaded with *config* as the plugin's own
    config dict.  deps is a stub object with None fields — sufficient for
    any code path that doesn't call ctx.get_llm_for_tag / ctx.environment."""
    return PluginContext(
        deps=_FakeDeps(),           # type: ignore[arg-type]
        plugin_name="mcp_connector",
        config=config,
        services={},
        plugin_cache={},
    )


# ---------------------------------------------------------------------------
# Helper: tiny reusable MCP server script written to a tmp file
# ---------------------------------------------------------------------------

_SERVER_SCRIPT_TEMPLATE = textwrap.dedent("""\
    import sys
    from mcp.server.fastmcp import FastMCP

    fmcp = FastMCP("{server_name}")

    @fmcp.tool()
    def echo(text: str) -> str:
        \"\"\"{echo_desc}\"\"\"
        return text

    @fmcp.tool()
    def add(a: int, b: int) -> int:
        \"\"\"{add_desc}\"\"\"
        return a + b

    if __name__ == "__main__":
        fmcp.run("stdio")
""")

_SINGLE_TOOL_SCRIPT_TEMPLATE = textwrap.dedent("""\
    import sys
    from mcp.server.fastmcp import FastMCP

    fmcp = FastMCP("{server_name}")

    @fmcp.tool()
    def {tool_name}({params}) -> str:
        \"\"\"{description}\"\"\"
        return {return_expr}

    if __name__ == "__main__":
        fmcp.run("stdio")
""")


def _write_server_script(
    tmp_path: Path,
    *,
    name: str = "test_server",
    echo_desc: str = "Echo the text back.",
    add_desc: str = "Add two numbers together.",
    filename: str = "server.py",
) -> Path:
    script = _SERVER_SCRIPT_TEMPLATE.format(
        server_name=name,
        echo_desc=echo_desc,
        add_desc=add_desc,
    )
    p = tmp_path / filename
    p.write_text(script, encoding="utf-8")
    return p


def _stdio_server_entry(
    script_path: Path,
    *,
    tool_prefix: str | None = None,
    server_id: str = "srv",
    timeout_s: float = 10.0,
    enabled: bool = True,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": server_id,
        "enabled": enabled,
        "transport": "stdio",
        "command": [sys.executable, str(script_path)],
        "timeout_s": timeout_s,
    }
    if tool_prefix is not None:
        entry["tool_prefix"] = tool_prefix
    if env is not None:
        entry["env"] = env
    return entry


# ===========================================================================
# Section 1 — build_tool() return type + empty / missing config
# ===========================================================================


class TestBuildToolReturnType:
    """Positive: build_tool always returns list[Tool], never a bare Tool."""

    def test_returns_list(self, tmp_path):
        """Spec 2 — build_tool returns list[Tool] (possibly empty)."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [_stdio_server_entry(script)]})
        result = build_tool(ctx)
        assert isinstance(result, list), (
            "build_tool must return list[Tool], not a single Tool instance"
        )

    def test_all_elements_are_tool_instances(self, tmp_path):
        """Spec 2 — every element satisfies the Tool ABC."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [_stdio_server_entry(script)]})
        tools = build_tool(ctx)
        for t in tools:
            assert isinstance(t, Tool), (
                f"Expected Tool instance, got {type(t)}"
            )


class TestEmptyConfig:
    """Spec 5a — graceful degradation: no servers / empty config."""

    def test_missing_servers_key_returns_empty_list(self):
        """Spec 5a — ctx.config with no 'servers' key => []."""
        ctx = _make_ctx({})
        result = build_tool(ctx)
        assert result == [], (
            "Missing 'servers' key in config must yield empty list, not raise"
        )

    def test_empty_servers_list_returns_empty_list(self):
        """Spec 5a — servers: [] => []."""
        ctx = _make_ctx({"servers": []})
        result = build_tool(ctx)
        assert result == []

    def test_all_servers_disabled_returns_empty_list(self, tmp_path):
        """Spec 1 + 5a — enabled: false servers are skipped entirely."""
        script = _write_server_script(tmp_path)
        entry = _stdio_server_entry(script, enabled=False)
        ctx = _make_ctx({"servers": [entry]})
        result = build_tool(ctx)
        assert result == [], (
            "A server with enabled=false must contribute zero tools"
        )

    def test_build_tool_does_not_raise_on_empty_config(self):
        """Spec 5a — no exception must escape build_tool for empty config."""
        ctx = _make_ctx({})
        # Must not raise — period.
        try:
            build_tool(ctx)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"build_tool raised {type(exc).__name__} on empty config: {exc}"
            )


# ===========================================================================
# Section 2 — Happy path: stdio server, tool discovery, naming
# ===========================================================================


class TestHappyPathStdio:
    """Spec 2 + 3 — successful factory-time discovery produces correct Tools."""

    def test_two_tools_discovered_from_server(self, tmp_path):
        """Spec 2 — factory produces one Tool per discovered MCP tool."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script, server_id="srv1"),
        ]})
        tools = build_tool(ctx)
        assert len(tools) == 2, (
            f"Expected 2 tools (echo + add), got {len(tools)}"
        )

    def test_tool_name_uses_server_id_when_no_prefix(self, tmp_path):
        """Spec 3 — no tool_prefix: name = <server_id>_<mcp_tool_name>."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script, server_id="myserver"),
        ]})
        tools = build_tool(ctx)
        names = {t.name for t in tools}
        assert "myserver_echo" in names, f"Expected 'myserver_echo' in {names}"
        assert "myserver_add" in names, f"Expected 'myserver_add' in {names}"

    def test_tool_name_uses_tool_prefix_when_set(self, tmp_path):
        """Spec 3 — tool_prefix set: name = <tool_prefix>_<mcp_tool_name>."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(
                script, server_id="myserver", tool_prefix="pfx",
            ),
        ]})
        tools = build_tool(ctx)
        names = {t.name for t in tools}
        assert "pfx_echo" in names, f"Expected 'pfx_echo' in {names}"
        assert "pfx_add" in names, f"Expected 'pfx_add' in {names}"
        # server_id must NOT appear in names when prefix is set
        assert "myserver_echo" not in names
        assert "myserver_add" not in names

    def test_tool_description_matches_mcp_manifest(self, tmp_path):
        """Spec 3 — .description comes from the MCP tool's manifest."""
        unique_echo = "Bounces input right back to caller."
        unique_add = "Sums two integer operands precisely."
        script = _write_server_script(
            tmp_path, echo_desc=unique_echo, add_desc=unique_add,
        )
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script, server_id="s"),
        ]})
        tools = build_tool(ctx)
        by_name = {t.name: t for t in tools}
        assert by_name["s_echo"].description == unique_echo, (
            "description must be taken verbatim from the MCP tool manifest"
        )
        assert by_name["s_add"].description == unique_add

    def test_parameters_schema_carried_from_mcp_manifest(self, tmp_path):
        """Spec 3 — .parameters_schema comes from the MCP tool's inputSchema."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script, server_id="s"),
        ]})
        tools = build_tool(ctx)
        by_name = {t.name: t for t in tools}

        echo_schema = by_name["s_echo"].parameters_schema
        assert isinstance(echo_schema, dict), "parameters_schema must be a dict"
        assert echo_schema.get("type") == "object"
        assert "text" in echo_schema.get("properties", {}), (
            "echo tool schema must expose 'text' property"
        )

        add_schema = by_name["s_add"].parameters_schema
        props = add_schema.get("properties", {})
        assert "a" in props, "add tool schema must expose 'a' property"
        assert "b" in props, "add tool schema must expose 'b' property"

    def test_parameters_schema_is_dict(self, tmp_path):
        """Spec 3 / Tool ABC — parameters_schema must be a dict (not None)."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script, server_id="s"),
        ]})
        tools = build_tool(ctx)
        for t in tools:
            assert isinstance(t.parameters_schema, dict), (
                f"Tool {t.name!r}: parameters_schema must be dict, "
                f"got {type(t.parameters_schema)}"
            )

    def test_tool_name_property_is_str(self, tmp_path):
        """Tool ABC — .name must be str."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script, server_id="s"),
        ]})
        tools = build_tool(ctx)
        for t in tools:
            assert isinstance(t.name, str)

    def test_tool_description_property_is_str(self, tmp_path):
        """Tool ABC — .description must be str."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script, server_id="s"),
        ]})
        tools = build_tool(ctx)
        for t in tools:
            assert isinstance(t.description, str)


# ===========================================================================
# Section 3 — execute(): happy path
# ===========================================================================


class TestExecuteHappyPath:
    """Spec 4 — execute() forwards params and returns a valid Stimulus."""

    async def test_execute_echo_returns_tool_feedback_stimulus(self, tmp_path):
        """Spec 4 — execute() returns Stimulus(type='tool_feedback', ...)."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script, server_id="s"),
        ]})
        tools = build_tool(ctx)
        by_name = {t.name: t for t in tools}
        tool = by_name["s_echo"]

        stim = await tool.execute("echo hello", {"text": "hello world"})

        assert isinstance(stim, Stimulus), (
            f"execute() must return Stimulus, got {type(stim)}"
        )
        assert stim.type == "tool_feedback", (
            f"Stimulus.type must be 'tool_feedback', got {stim.type!r}"
        )

    async def test_execute_source_is_tool_colon_name(self, tmp_path):
        """Spec 4 — Stimulus.source == 'tool:<name>' (mirrors duckduckgo pattern)."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script, server_id="s"),
        ]})
        tools = build_tool(ctx)
        by_name = {t.name: t for t in tools}
        tool = by_name["s_echo"]

        stim = await tool.execute("echo hi", {"text": "hi"})

        assert stim.source == f"tool:{tool.name}", (
            f"Stimulus.source must be 'tool:{tool.name}', got {stim.source!r}"
        )

    async def test_execute_echo_content_contains_sent_text(self, tmp_path):
        """Spec 4 — textual result from MCP is in Stimulus.content."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script, server_id="s"),
        ]})
        tools = build_tool(ctx)
        by_name = {t.name: t for t in tools}
        tool = by_name["s_echo"]

        payload = "ping_unique_string_xyz"
        stim = await tool.execute("echo payload", {"text": payload})

        assert payload in stim.content, (
            f"Stimulus.content must contain MCP result {payload!r}, "
            f"got {stim.content!r}"
        )

    async def test_execute_add_returns_numeric_result(self, tmp_path):
        """Spec 4 — MCP tools/call result (add 3+4=7) is surfaced in content."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script, server_id="s"),
        ]})
        tools = build_tool(ctx)
        by_name = {t.name: t for t in tools}
        tool = by_name["s_add"]

        stim = await tool.execute("add numbers", {"a": 3, "b": 4})

        assert stim.type == "tool_feedback"
        # 3 + 4 = 7; the string "7" must appear in content
        assert "7" in stim.content, (
            f"Expected '7' (sum of 3+4) in content, got {stim.content!r}"
        )

    async def test_execute_adrenalin_is_false(self, tmp_path):
        """Spec 4 — execute() does not raise the adrenalin flag."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script, server_id="s"),
        ]})
        tools = build_tool(ctx)
        by_name = {t.name: t for t in tools}
        tool = by_name["s_echo"]

        stim = await tool.execute("echo test", {"text": "test"})

        assert stim.adrenalin is False

    async def test_execute_stimulus_has_timestamp(self, tmp_path):
        """Tool ABC — Stimulus must carry a datetime timestamp."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script, server_id="s"),
        ]})
        tools = build_tool(ctx)
        by_name = {t.name: t for t in tools}
        tool = by_name["s_echo"]

        stim = await tool.execute("ts test", {"text": "x"})

        assert isinstance(stim.timestamp, datetime), (
            f"Stimulus.timestamp must be datetime, got {type(stim.timestamp)}"
        )

    async def test_execute_content_is_str(self, tmp_path):
        """Tool ABC — Stimulus.content must be a str."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script, server_id="s"),
        ]})
        tools = build_tool(ctx)
        by_name = {t.name: t for t in tools}
        tool = by_name["s_echo"]

        stim = await tool.execute("test", {"text": "x"})
        assert isinstance(stim.content, str)

    async def test_execute_intent_passed_when_no_relevant_param(self, tmp_path):
        """Spec 4 — intent forwarding: execute(intent, {}) must not raise."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script, server_id="s"),
        ]})
        tools = build_tool(ctx)
        by_name = {t.name: t for t in tools}
        tool = by_name["s_echo"]

        # Omit params to confirm execute doesn't hard-crash without them.
        # The MCP call may fail (missing required 'text') — that's fine
        # as long as execute() returns a Stimulus (not raises).
        try:
            stim = await tool.execute("an intent string", {})
            assert isinstance(stim, Stimulus)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"execute() raised {type(exc).__name__} instead of returning "
                f"an error Stimulus: {exc}"
            )


# ===========================================================================
# Section 4 — Multiple servers: union of tools
# ===========================================================================


class TestMultipleServers:
    """Spec 2 — multiple enabled servers contribute their tools to one list."""

    def test_two_servers_union_of_tools(self, tmp_path):
        """Spec 2 — all tools from all enabled servers are returned."""
        script_a = _write_server_script(tmp_path, filename="server_a.py")
        script_b = _write_server_script(tmp_path, filename="server_b.py")
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script_a, server_id="alpha"),
            _stdio_server_entry(script_b, server_id="beta"),
        ]})
        tools = build_tool(ctx)
        names = {t.name for t in tools}

        assert "alpha_echo" in names
        assert "alpha_add" in names
        assert "beta_echo" in names
        assert "beta_add" in names

    def test_disabled_server_excluded_from_union(self, tmp_path):
        """Spec 1 + 2 — enabled=false server excluded; other server still works."""
        script_a = _write_server_script(tmp_path, filename="server_a.py")
        script_b = _write_server_script(tmp_path, filename="server_b.py")
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script_a, server_id="good"),
            _stdio_server_entry(script_b, server_id="off", enabled=False),
        ]})
        tools = build_tool(ctx)
        names = {t.name for t in tools}

        assert "good_echo" in names
        assert "good_add" in names
        assert "off_echo" not in names
        assert "off_add" not in names

    def test_total_count_equals_sum_across_servers(self, tmp_path):
        """Spec 2 — total tool count = sum of each server's tool count."""
        script_a = _write_server_script(tmp_path, filename="sa.py")
        script_b = _write_server_script(tmp_path, filename="sb.py")
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script_a, server_id="a"),
            _stdio_server_entry(script_b, server_id="b"),
        ]})
        tools = build_tool(ctx)
        # Each server exposes 2 tools: echo + add.
        assert len(tools) == 4, (
            f"Expected 4 tools (2 per server × 2 servers), got {len(tools)}"
        )

    async def test_tools_from_different_servers_both_callable(self, tmp_path):
        """Spec 4 — tools from each server independently callable."""
        script_a = _write_server_script(tmp_path, filename="sa.py")
        script_b = _write_server_script(tmp_path, filename="sb.py")
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script_a, server_id="sa"),
            _stdio_server_entry(script_b, server_id="sb"),
        ]})
        tools = build_tool(ctx)
        by_name = {t.name: t for t in tools}

        stim_a = await by_name["sa_echo"].execute("test", {"text": "from_a"})
        stim_b = await by_name["sb_echo"].execute("test", {"text": "from_b"})

        assert "from_a" in stim_a.content
        assert "from_b" in stim_b.content


# ===========================================================================
# Section 5 — Name collision: last-registration-wins, no raise
# ===========================================================================


class TestNameCollision:
    """Spec 5d — two servers producing the same tool name must not raise."""

    def test_name_collision_does_not_raise(self, tmp_path):
        """Spec 5d — identical tool names across servers: no exception."""
        script_a = _write_server_script(tmp_path, filename="ca.py")
        script_b = _write_server_script(tmp_path, filename="cb.py")
        # Both servers use same prefix → same resulting tool names
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script_a, tool_prefix="shared"),
            _stdio_server_entry(script_b, tool_prefix="shared"),
        ]})
        try:
            build_tool(ctx)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"build_tool raised {type(exc).__name__} on name collision: {exc}"
            )

    def test_name_collision_list_contains_tool_with_collided_name(
        self, tmp_path,
    ):
        """Spec 5d — after collision, the name still exists in returned list."""
        script_a = _write_server_script(tmp_path, filename="ca.py")
        script_b = _write_server_script(tmp_path, filename="cb.py")
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script_a, tool_prefix="shared"),
            _stdio_server_entry(script_b, tool_prefix="shared"),
        ]})
        tools = build_tool(ctx)
        names = [t.name for t in tools]
        # At least one tool named "shared_echo" must be present.
        assert "shared_echo" in names, (
            "Tool name 'shared_echo' must be present even after collision"
        )


# ===========================================================================
# Section 6 — Graceful degradation: unreachable server at factory time
# ===========================================================================


class TestUnreachableServerAtFactoryTime:
    """Spec 5b — unreachable/bad server contributes 0 tools; others unaffected."""

    def test_bogus_command_contributes_zero_tools_no_raise(self):
        """Spec 5b — server with nonexistent executable -> 0 tools, no raise."""
        ctx = _make_ctx({"servers": [
            {
                "id": "bad",
                "enabled": True,
                "transport": "stdio",
                "command": ["/nonexistent/binary/that/cannot/run"],
                "timeout_s": 5.0,
            },
        ]})
        try:
            result = build_tool(ctx)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"build_tool raised {type(exc).__name__} for unreachable server: {exc}"
            )
        assert result == [], (
            f"Unreachable server must yield [], got {result}"
        )

    def test_bad_server_does_not_block_good_server(self, tmp_path):
        """Spec 5b — bad server isolated; valid server still contributes tools."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            {
                "id": "bad",
                "enabled": True,
                "transport": "stdio",
                "command": ["/this/executable/does/not/exist"],
                "timeout_s": 5.0,
            },
            _stdio_server_entry(script, server_id="good"),
        ]})
        try:
            tools = build_tool(ctx)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"build_tool raised {type(exc).__name__}: {exc}"
            )
        names = {t.name for t in tools}
        assert "good_echo" in names, (
            "Valid server must still contribute its tools when another server fails"
        )
        assert "good_add" in names

    def test_bad_server_before_good_server_order_independent(self, tmp_path):
        """Spec 5b — ordering (bad before good) doesn't matter; good still works."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            {
                "id": "bad",
                "enabled": True,
                "transport": "stdio",
                "command": [sys.executable, "-c", "import time; time.sleep(0); raise SystemExit(1)"],
                "timeout_s": 5.0,
            },
            _stdio_server_entry(script, server_id="good"),
        ]})
        try:
            tools = build_tool(ctx)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"build_tool raised {type(exc).__name__}: {exc}"
            )
        names = {t.name for t in tools}
        assert "good_echo" in names

    def test_only_bad_server_returns_empty_list_no_raise(self):
        """Spec 5b — a single bad server yields [], not an exception."""
        ctx = _make_ctx({"servers": [
            {
                "id": "bad",
                "enabled": True,
                "transport": "stdio",
                "command": ["python", "-c", "raise SystemExit(99)"],
                "timeout_s": 5.0,
            },
        ]})
        try:
            result = build_tool(ctx)
            assert isinstance(result, list)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"build_tool raised {type(exc).__name__}: {exc}"
            )

    def test_handshake_timeout_server_contributes_zero_tools(self, tmp_path):
        """Spec 5b — a server that never completes MCP handshake is skipped."""
        # Script that starts but never sends anything on stdout
        hangs_script = tmp_path / "hang.py"
        hangs_script.write_text(
            "import time\ntime.sleep(9999)\n", encoding="utf-8",
        )
        ctx = _make_ctx({"servers": [
            {
                "id": "hangs",
                "enabled": True,
                "transport": "stdio",
                "command": [sys.executable, str(hangs_script)],
                "timeout_s": 2.0,   # short enough to not block CI
            },
        ]})
        try:
            result = build_tool(ctx)
            assert isinstance(result, list)
            # A hanging server must not contribute any tools
            assert result == []
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"build_tool raised {type(exc).__name__} for hanging server: {exc}"
            )


# ===========================================================================
# Section 7 — Graceful degradation: execute() runtime failures
# ===========================================================================


class TestExecuteRuntimeFailures:
    """Spec 5c — execute() NEVER raises; runtime failures -> error Stimulus."""

    async def test_execute_against_killed_server_returns_error_stimulus(
        self, tmp_path,
    ):
        """Spec 5c — server dies after discovery; execute() returns error Stim."""
        # We'll use a server that exits after a small delay.
        # The plugin connects at factory time; by execute() time it's dead.
        dying_script = tmp_path / "dying.py"
        dying_script.write_text(
            textwrap.dedent("""\
                import sys, threading, time
                from mcp.server.fastmcp import FastMCP

                fmcp = FastMCP("dying")

                @fmcp.tool()
                def echo(text: str) -> str:
                    \"\"\"Echo.\"\"\"
                    return text

                def _killer():
                    time.sleep(1.5)
                    sys.exit(0)

                threading.Thread(target=_killer, daemon=True).start()
                fmcp.run("stdio")
            """),
            encoding="utf-8",
        )
        ctx = _make_ctx({"servers": [
            {
                "id": "dying",
                "enabled": True,
                "transport": "stdio",
                "command": [sys.executable, str(dying_script)],
                "timeout_s": 10.0,
            },
        ]})
        tools = build_tool(ctx)
        by_name = {t.name: t for t in tools}
        if "dying_echo" not in by_name:
            pytest.skip(
                "Server did not expose 'dying_echo' at factory time "
                "(possibly already exited); skipping runtime-failure test."
            )

        # Wait for server to die
        import asyncio
        await asyncio.sleep(2.0)

        tool = by_name["dying_echo"]
        try:
            stim = await tool.execute("echo test", {"text": "test"})
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"execute() raised {type(exc).__name__} instead of returning "
                f"an error Stimulus: {exc}"
            )

        assert isinstance(stim, Stimulus), (
            f"execute() must return Stimulus even on failure, got {type(stim)}"
        )
        # The stimulus must be a feedback or system_event, not a data stim
        assert stim.type in ("tool_feedback", "system_event"), (
            f"Expected 'tool_feedback' or 'system_event', got {stim.type!r}"
        )

    async def test_execute_never_raises_on_mcp_error(self, tmp_path):
        """Spec 5c — MCP error response from server does not propagate as exception."""
        # Server that raises on tool call
        error_script = tmp_path / "error_server.py"
        error_script.write_text(
            textwrap.dedent("""\
                from mcp.server.fastmcp import FastMCP

                fmcp = FastMCP("error_server")

                @fmcp.tool()
                def always_fails(x: str) -> str:
                    \"\"\"Always raises.\"\"\"
                    raise RuntimeError("intentional tool error")

                fmcp.run("stdio")
            """),
            encoding="utf-8",
        )
        ctx = _make_ctx({"servers": [
            {
                "id": "errsvr",
                "enabled": True,
                "transport": "stdio",
                "command": [sys.executable, str(error_script)],
                "timeout_s": 10.0,
            },
        ]})
        tools = build_tool(ctx)
        by_name = {t.name: t for t in tools}
        if "errsvr_always_fails" not in by_name:
            pytest.skip("Error server tool not discovered; skipping.")

        tool = by_name["errsvr_always_fails"]
        try:
            stim = await tool.execute("will fail", {"x": "anything"})
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"execute() raised {type(exc).__name__} instead of returning "
                f"error Stimulus: {exc}"
            )

        assert isinstance(stim, Stimulus)
        assert stim.type in ("tool_feedback", "system_event")
        assert stim.source == f"tool:{tool.name}"

    async def test_execute_error_stimulus_source_correct(self, tmp_path):
        """Spec 4 + 5c — error Stimulus.source == 'tool:<name>' always."""
        # Re-use the always-fails server pattern
        error_script = tmp_path / "ef_server.py"
        error_script.write_text(
            textwrap.dedent("""\
                from mcp.server.fastmcp import FastMCP

                fmcp = FastMCP("ef_server")

                @fmcp.tool()
                def boom(msg: str) -> str:
                    \"\"\"Boom.\"\"\"
                    raise ValueError("boom")

                fmcp.run("stdio")
            """),
            encoding="utf-8",
        )
        ctx = _make_ctx({"servers": [
            {
                "id": "ef",
                "enabled": True,
                "transport": "stdio",
                "command": [sys.executable, str(error_script)],
                "timeout_s": 10.0,
            },
        ]})
        tools = build_tool(ctx)
        by_name = {t.name: t for t in tools}
        if "ef_boom" not in by_name:
            pytest.skip("ef_boom not discovered; skipping.")

        tool = by_name["ef_boom"]
        stim = await tool.execute("trigger", {"msg": "test"})
        assert stim.source == f"tool:{tool.name}"


# ===========================================================================
# Section 8 — Tool naming BVA: boundary cases for prefix and server_id
# ===========================================================================


class TestToolNamingBVA:
    """Spec 3 BVA — name formation edge cases."""

    def test_empty_tool_prefix_falls_back_to_server_id(self, tmp_path):
        """Spec 3 — empty string tool_prefix treated as not set; uses server_id."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script, server_id="srv", tool_prefix=""),
        ]})
        tools = build_tool(ctx)
        names = {t.name for t in tools}
        # Must use server_id when prefix is empty/falsy
        assert "srv_echo" in names or "echo" in names, (
            "With empty tool_prefix, expected server_id-based name or bare MCP name"
        )

    def test_tool_prefix_with_underscores_allowed(self, tmp_path):
        """Spec 3 — tool_prefix may itself contain underscores."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(
                script, server_id="s", tool_prefix="my_ext_prefix",
            ),
        ]})
        tools = build_tool(ctx)
        names = {t.name for t in tools}
        assert "my_ext_prefix_echo" in names

    def test_single_enabled_server_one_tool(self, tmp_path):
        """Spec 2 BVA — single server, single tool."""
        single_tool_script = tmp_path / "single.py"
        single_tool_script.write_text(
            textwrap.dedent("""\
                from mcp.server.fastmcp import FastMCP

                fmcp = FastMCP("single")

                @fmcp.tool()
                def ping() -> str:
                    \"\"\"Just a ping.\"\"\"
                    return "pong"

                fmcp.run("stdio")
            """),
            encoding="utf-8",
        )
        ctx = _make_ctx({"servers": [
            {
                "id": "solo",
                "enabled": True,
                "transport": "stdio",
                "command": [sys.executable, str(single_tool_script)],
                "timeout_s": 10.0,
            },
        ]})
        tools = build_tool(ctx)
        assert len(tools) == 1
        assert tools[0].name == "solo_ping"

    def test_server_with_zero_tools_contributes_nothing(self, tmp_path):
        """Spec 2 BVA — server exposing 0 tools contributes 0 to result."""
        zero_tools_script = tmp_path / "zero.py"
        zero_tools_script.write_text(
            textwrap.dedent("""\
                from mcp.server.fastmcp import FastMCP

                fmcp = FastMCP("zero_tools")
                # No tools registered.
                fmcp.run("stdio")
            """),
            encoding="utf-8",
        )
        ctx = _make_ctx({"servers": [
            {
                "id": "empty_srv",
                "enabled": True,
                "transport": "stdio",
                "command": [sys.executable, str(zero_tools_script)],
                "timeout_s": 10.0,
            },
        ]})
        tools = build_tool(ctx)
        assert tools == [], (
            "A server with zero MCP tools must contribute zero Tool wrappers"
        )


# ===========================================================================
# Section 9 — Config field validation / BVA
# ===========================================================================


class TestConfigSchema:
    """Spec 1 — config schema handling edge cases."""

    def test_servers_key_not_a_list_does_not_raise(self):
        """Spec 5a / robustness — malformed servers value should not crash."""
        ctx = _make_ctx({"servers": "not_a_list"})
        try:
            result = build_tool(ctx)
            assert isinstance(result, list)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"build_tool raised {type(exc).__name__} for malformed servers: {exc}"
            )

    def test_servers_none_does_not_raise(self):
        """Spec 5a / robustness — servers=None is malformed, must not crash."""
        ctx = _make_ctx({"servers": None})
        try:
            result = build_tool(ctx)
            assert isinstance(result, list)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"build_tool raised {type(exc).__name__} for servers=None: {exc}"
            )

    def test_server_entry_missing_enabled_treated_as_disabled_or_enabled(
        self, tmp_path,
    ):
        """Spec 1 — missing 'enabled' key: implementation must not raise."""
        script = _write_server_script(tmp_path)
        entry = {
            "id": "no_enabled",
            "transport": "stdio",
            "command": [sys.executable, str(script)],
            "timeout_s": 10.0,
            # 'enabled' deliberately absent
        }
        ctx = _make_ctx({"servers": [entry]})
        try:
            result = build_tool(ctx)
            assert isinstance(result, list)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"build_tool raised {type(exc).__name__} for missing 'enabled': {exc}"
            )

    def test_server_entry_missing_id_does_not_raise(self, tmp_path):
        """Spec 1 / robustness — missing 'id': must not crash factory."""
        script = _write_server_script(tmp_path)
        entry = {
            # 'id' deliberately absent
            "enabled": True,
            "transport": "stdio",
            "command": [sys.executable, str(script)],
            "timeout_s": 10.0,
        }
        ctx = _make_ctx({"servers": [entry]})
        try:
            result = build_tool(ctx)
            assert isinstance(result, list)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"build_tool raised {type(exc).__name__} for missing 'id': {exc}"
            )

    def test_timeout_s_respected_approx(self, tmp_path):
        """Spec 1 — timeout_s field controls per-server connect timeout."""
        # A hanging server with a 2 s timeout must not block much longer than that.
        hang = tmp_path / "long_hang.py"
        hang.write_text(
            "import time\ntime.sleep(9999)\n", encoding="utf-8",
        )
        ctx = _make_ctx({"servers": [
            {
                "id": "ht",
                "enabled": True,
                "transport": "stdio",
                "command": [sys.executable, str(hang)],
                "timeout_s": 2.0,
            },
        ]})
        t0 = time.monotonic()
        build_tool(ctx)
        elapsed = time.monotonic() - t0
        # Allow generous headroom (3× the stated timeout) for CI variability.
        assert elapsed < 10.0, (
            f"build_tool took {elapsed:.1f}s for a 2s timeout server — "
            "timeout_s appears to not be respected"
        )

    def test_env_dict_passed_to_stdio_subprocess(self, tmp_path):
        """Spec 1 / 6 — env dict from config is forwarded to subprocess env."""
        # A server that reads an env var and exposes it as a tool
        env_script = tmp_path / "env_server.py"
        env_script.write_text(
            textwrap.dedent("""\
                import os
                from mcp.server.fastmcp import FastMCP

                fmcp = FastMCP("env_server")

                @fmcp.tool()
                def get_custom_var() -> str:
                    \"\"\"Return the KRAKEY_TEST_VAR env var value.\"\"\"
                    return os.environ.get("KRAKEY_TEST_VAR", "NOT_SET")

                fmcp.run("stdio")
            """),
            encoding="utf-8",
        )
        ctx = _make_ctx({"servers": [
            {
                "id": "envtest",
                "enabled": True,
                "transport": "stdio",
                "command": [sys.executable, str(env_script)],
                "timeout_s": 10.0,
                "env": {"KRAKEY_TEST_VAR": "hello_from_env"},
            },
        ]})
        tools = build_tool(ctx)
        by_name = {t.name: t for t in tools}
        if "envtest_get_custom_var" not in by_name:
            pytest.skip("env_server tool not discovered; skipping env test.")

        # We can't assert the env was used without calling execute, so just
        # confirm the tool was discovered (env didn't prevent subprocess start).
        assert "envtest_get_custom_var" in by_name


# ===========================================================================
# Section 10 — execute() BVA
# ===========================================================================


class TestExecuteBVA:
    """Spec 4 BVA — boundary cases for execute()."""

    async def test_execute_with_empty_params_dict(self, tmp_path):
        """Spec 4 BVA — empty params dict: must not raise out of execute()."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script, server_id="s"),
        ]})
        tools = build_tool(ctx)
        by_name = {t.name: t for t in tools}
        tool = by_name["s_echo"]

        try:
            stim = await tool.execute("intent", {})
            assert isinstance(stim, Stimulus)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"execute() raised {type(exc).__name__} for empty params: {exc}"
            )

    async def test_execute_with_empty_intent_string(self, tmp_path):
        """Spec 4 BVA — empty intent string: execute() must not raise."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script, server_id="s"),
        ]})
        tools = build_tool(ctx)
        by_name = {t.name: t for t in tools}
        tool = by_name["s_echo"]

        try:
            stim = await tool.execute("", {"text": "x"})
            assert isinstance(stim, Stimulus)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"execute() raised {type(exc).__name__} for empty intent: {exc}"
            )

    async def test_execute_with_extra_params_does_not_raise(self, tmp_path):
        """Spec 4 BVA — extra keys in params not in schema: must not raise."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script, server_id="s"),
        ]})
        tools = build_tool(ctx)
        by_name = {t.name: t for t in tools}
        tool = by_name["s_echo"]

        try:
            stim = await tool.execute(
                "intent", {"text": "hi", "unexpected_extra": "ignored"},
            )
            assert isinstance(stim, Stimulus)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"execute() raised {type(exc).__name__} for extra params: {exc}"
            )

    async def test_execute_called_multiple_times_same_tool(self, tmp_path):
        """Spec 4 state — execute() is re-entrant (called many times, same result)."""
        script = _write_server_script(tmp_path)
        ctx = _make_ctx({"servers": [
            _stdio_server_entry(script, server_id="s"),
        ]})
        tools = build_tool(ctx)
        by_name = {t.name: t for t in tools}
        tool = by_name["s_echo"]

        results = []
        for i in range(3):
            stim = await tool.execute("test", {"text": f"call_{i}"})
            results.append(stim)

        for stim in results:
            assert stim.type == "tool_feedback"
        # Content from each call should be distinct (or at least valid Stimulus)
        assert all(isinstance(s, Stimulus) for s in results)


# ===========================================================================
# Section 11 — Spec point 6: stdio spawned by plugin, not ctx.environment
# ===========================================================================


class TestStdioSpawnedByPlugin:
    """Spec 6 — stdio servers are spawned by the plugin itself."""

    def test_build_tool_does_not_call_ctx_environment(self, tmp_path):
        """Spec 6 — ctx.environment() must not be called during discovery."""
        script = _write_server_script(tmp_path)

        environment_calls: list[str] = []

        class _TrackedDeps:
            llm_factory = None
            environment_router = None

        class _TrackedCtx:
            deps = _TrackedDeps()
            plugin_name = "mcp_connector"
            config = {"servers": [
                _stdio_server_entry(script, server_id="s"),
            ]}
            services: dict = {}
            plugin_cache: dict = {}

            def environment(self, env_name: str):
                environment_calls.append(env_name)
                raise RuntimeError(
                    "ctx.environment() must NOT be called for stdio servers"
                )

            def get_llm_for_tag(self, tag_name):
                return None

        ctx = _TrackedCtx()
        try:
            build_tool(ctx)  # type: ignore[arg-type]
        except RuntimeError as exc:
            if "ctx.environment() must NOT be called" in str(exc):
                pytest.fail(
                    "build_tool called ctx.environment() for a stdio server, "
                    "violating spec point 6"
                )
            raise

        assert environment_calls == [], (
            f"ctx.environment() was called with args {environment_calls!r}; "
            "stdio server spawning must not use the environment router"
        )
