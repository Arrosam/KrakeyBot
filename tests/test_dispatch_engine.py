"""LocalDispatchEngine — Protocol conformance + lazy DecisionDispatcher
construction + 4-side-effect orchestration.

The Engine wraps DecisionDispatcher; tests verify the wrapper drives
the same 4 side-effects (log_summary, dispatch_tool_calls,
apply_memory_writes, apply_memory_updates) the orchestrator used to
call directly. Mock the dispatcher to record calls."""
from __future__ import annotations

from typing import Any

import pytest

from krakey.engines.dispatch.default import LocalDispatchEngine
from krakey.interfaces.engines import (
    DecisionResult,
    DispatchEngine,
    ToolCall,
)


class _RecordingDispatcher:
    """Stand-in DecisionDispatcher — records each method call."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.summary_calls: list = []
        self.tool_call_batches: list = []
        self.memory_write_batches: list = []
        self.memory_update_batches: list = []

    def log_summary(self, heartbeat_id, decision_result):
        self.summary_calls.append((heartbeat_id, decision_result))

    async def dispatch_tool_calls(self, heartbeat_id, calls):
        self.tool_call_batches.append((heartbeat_id, calls))

    async def apply_memory_writes(self, writes, recall_nodes, heartbeat_id):
        self.memory_write_batches.append(
            (writes, recall_nodes, heartbeat_id),
        )

    async def apply_memory_updates(self, updates):
        self.memory_update_batches.append(updates)


class _FakeRuntime:
    """Bare runtime stand-in — only the attributes the dispatcher
    constructor reads."""

    def __init__(self):
        self.tools = "TOOLS"
        self.batch_tracker = "BATCH"
        self.buffer = "BUFFER"
        self.memory = "MEMORY"
        self.log = "LOG"
        self.events = "EVENTS"


def test_satisfies_dispatch_engine_protocol():
    eng = LocalDispatchEngine()
    assert isinstance(eng, DispatchEngine)


@pytest.mark.asyncio
async def test_dispatch_runs_all_four_side_effects(monkeypatch):
    """One dispatch() call → log_summary + dispatch_tool_calls +
    apply_memory_writes + apply_memory_updates, in order."""
    captured: dict[str, Any] = {}

    def _make(**kwargs):
        d = _RecordingDispatcher(**kwargs)
        captured["dispatcher"] = d
        return d

    monkeypatch.setattr(
        "krakey.engines.dispatch.dispatcher.DecisionDispatcher", _make,
    )
    eng = LocalDispatchEngine()
    rt = _FakeRuntime()
    result = DecisionResult(
        tool_calls=[ToolCall(tool="t", intent="x")],
        memory_writes=[{"content": "w"}],
        memory_updates=[{"node_name": "n", "new_category": "FACT"}],
        sleep=False,
    )
    await eng.dispatch(
        7, result, rt, recall_context=[{"name": "ctx"}],
    )
    d = captured["dispatcher"]
    assert d.summary_calls == [(7, result)]
    assert d.tool_call_batches == [(7, result.tool_calls)]
    assert d.memory_write_batches == [
        (result.memory_writes, [{"name": "ctx"}], 7),
    ]
    assert d.memory_update_batches == [result.memory_updates]


@pytest.mark.asyncio
async def test_dispatcher_constructed_lazily_and_reused(monkeypatch):
    """Construction happens on the first dispatch() call, not at
    Engine __init__. Subsequent calls reuse the same dispatcher."""
    construct_count = {"n": 0}

    def _make(**kwargs):
        construct_count["n"] += 1
        return _RecordingDispatcher(**kwargs)

    monkeypatch.setattr(
        "krakey.engines.dispatch.dispatcher.DecisionDispatcher", _make,
    )
    eng = LocalDispatchEngine()
    assert construct_count["n"] == 0  # not built yet

    rt = _FakeRuntime()
    result = DecisionResult()
    await eng.dispatch(1, result, rt)
    assert construct_count["n"] == 1
    await eng.dispatch(2, result, rt)
    assert construct_count["n"] == 1  # reused


@pytest.mark.asyncio
async def test_dispatch_handles_none_recall_context(monkeypatch):
    """recall_context=None should pass [] to apply_memory_writes
    (not None) so the dispatcher doesn't crash on iteration."""
    captured: dict[str, Any] = {}

    def _make(**kwargs):
        d = _RecordingDispatcher(**kwargs)
        captured["d"] = d
        return d

    monkeypatch.setattr(
        "krakey.engines.dispatch.dispatcher.DecisionDispatcher", _make,
    )
    eng = LocalDispatchEngine()
    rt = _FakeRuntime()
    await eng.dispatch(1, DecisionResult(), rt, recall_context=None)
    write_call = captured["d"].memory_write_batches[0]
    assert write_call[1] == []
