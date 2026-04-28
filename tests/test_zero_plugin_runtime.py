"""Zero-plugin invariant — runtime must survive with all plugins
disabled.

Set 2026-04-25 as a load-bearing design rule: disabling or removing
any plugin (Reflects, tools, sensories) must NOT break the
runtime's core loop.

Specific paths exercised:
  * No recall_anchor role registered → ``new_recall`` returns
    ``NoopRecall``, heartbeat completes with empty ``[GRAPH MEMORY]``.
  * No hypothalamus role registered → tool-call fallback parses
    ``<tool_call>...</tool_call>`` blocks out of the raw Self response.
  * Zero registered tools → Self's tool calls produce
    ``Unknown tool: X`` system events, runtime keeps heartbeating.
  * All three at once → cold runtime + empty stimulus → still
    completes a heartbeat without raising.
"""
import pytest

from krakey.memory.recall import NoopRecall, RecallResult
from krakey.interfaces.reflect import ReflectRegistry
from tests._runtime_helpers import (
    NullEmbedder, ScriptedLLM, build_runtime_with_fakes,
)


def _strip_all_reflects(runtime) -> None:
    """Drop every registered Reflect — what config-driven disable will
    end up doing once the toggles land."""
    runtime.reflects._by_role.clear()
    runtime.reflects._order.clear()


# ---- noop recall when no anchor is registered -----------------------


async def test_noop_recall_satisfies_runtime_lifecycle(tmp_path):
    """The NoopRecall must respond to the same calls IncrementalRecall
    does (add_stimuli, finalize) without errors and without producing
    side effects Self would notice."""
    rec = NoopRecall()
    assert rec.processed_stimuli == []
    await rec.add_stimuli(["fake stimulus 1", "fake stimulus 2"])  # type: ignore[list-item]
    assert len(rec.processed_stimuli) == 2
    result = await rec.finalize()
    assert isinstance(result, RecallResult)
    assert result.nodes == []
    assert result.edges == []


# ---- runtime end-to-end with all Reflects stripped -------------------


async def test_runtime_heartbeat_survives_all_reflects_unregistered(tmp_path):
    """A heartbeat with zero Reflects, zero stimuli, and the tool-call
    parser finding no <tool_call> blocks must complete without raising.

    This is the core invariant in its strongest form: the runtime can
    breathe in vacuum.
    """
    self_llm = ScriptedLLM([
        "[THINKING]\nQuiet beat.\n[DECISION]\nNo action.\n[HIBERNATE]\n1",
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    _strip_all_reflects(runtime)
    assert runtime.reflects.has_role("hypothalamus") is False
    assert runtime.reflects.by_role("recall_anchor") is None

    await runtime.run(iterations=1)


async def test_runtime_heartbeat_with_no_tools_emits_unknown_tool(tmp_path):
    """Self emits a <tool_call> referencing a tool name that
    doesn't exist in the registry. Runtime must NOT crash; it must
    push an `Unknown tool: ...` system event so Self can correct
    on the next beat."""
    self_llm = ScriptedLLM([
        '[THINKING]\nlet me reply.\n'
        '[DECISION]\nGreet.\n'
        '<tool_call>\n{"name": "nonexistent_tool", "arguments": {}}\n</tool_call>\n'
        '[HIBERNATE]\n1'
    ])
    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    # Strip the hypothalamus Reflect so dispatch goes via the action
    # executor (the path we want to exercise here).
    runtime.reflects._by_role.pop("hypothalamus", None)
    runtime.reflects._order.remove("hypothalamus")

    from krakey.interfaces.tool import ToolRegistry
    runtime.tools = ToolRegistry()

    await runtime.run(iterations=1)
    drained = runtime.buffer.drain()
    unknown_events = [
        s for s in drained
        if s.type == "system_event"
        and "Unknown tool" in s.content
        and "nonexistent_tool" in s.content
    ]
    assert unknown_events, (
        "expected an 'Unknown tool: nonexistent_tool' system "
        "event so Self can self-correct, but none was pushed"
    )


async def test_runtime_construction_works_with_no_reflects(tmp_path):
    """Even Runtime.__init__ should tolerate a state where no Reflects
    end up registered."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    _strip_all_reflects(runtime)
    assert runtime.reflects.has_role("hypothalamus") is False
    # new_recall falls back to NoopRecall via the orchestrator.
    recall = runtime._new_recall()
    assert isinstance(recall, NoopRecall)
