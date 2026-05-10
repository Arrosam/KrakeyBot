"""``DispatchEngine`` — execute a ``DecisionResult``'s side-effects.

After the ``DecisionEngine`` produces a ``DecisionResult``, four side-
effects need to fire:

  1. Log + publish a ``DecisionExecutedEvent`` summary.
  2. Schedule each ``ToolCall`` as an async task and register the
     batch with the ``BatchTracker`` so completion can wake Self.
  3. Apply ``memory_writes`` (LLM-extracted nodes/edges via
     ``MemoryEngine.explicit_write``).
  4. Apply ``memory_updates`` (category flips like TARGET → FACT via
     ``MemoryEngine.update_node_category``).

The default impl ``LocalDispatchEngine`` runs everything in-process —
``tool.execute()`` is a Python coroutine call, memory writes go
through the in-process ``MemoryEngine`` reference. A user replacing
this Engine can ship tool execution to a remote worker (HTTP/RPC),
queue calls for batch processing, add per-tool retry/rate-limit
policies, etc.

The ``DispatchEngine`` is also the natural place to virtualize
execution location at the Engine level — distinct from the per-tool
``Environment`` mechanism (which is for plugins choosing where to
push CLI commands). Dispatch-level virtualization replaces the entire
Python call to ``tool.execute()``; Environment-level virtualization
only redirects CLI commands the tool itself produces.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from krakey.interfaces.engines.decision import DecisionResult
    from krakey.runtime.runtime import Runtime


@runtime_checkable
class DispatchEngine(Protocol):
    """Run one ``DecisionResult``'s side-effects.

    The ``runtime`` reference is the Engine's gateway to the resources
    it needs: ``runtime.tools`` (ToolRegistry), ``runtime.batch_tracker``,
    ``runtime.buffer`` (StimulusBuffer), ``runtime.memory``,
    ``runtime.log`` + ``runtime.events``. Custom impls follow the same
    access pattern; they don't need new wiring.

    The method is async + non-returning: side-effects are observed via
    the runtime's existing event/log streams. Errors during dispatch
    of a single ToolCall are wrapped into a tool_feedback Stimulus
    (per the existing dispatcher contract); errors during memory
    writes log + skip per-write so one bad write never blocks the rest.
    """

    async def dispatch(
        self,
        heartbeat_id: int,
        decision_result: "DecisionResult",
        runtime: "Runtime",
        *,
        recall_context: list[dict] | None = None,
    ) -> None:
        """Run the 4 side-effects of a DecisionResult.

        ``recall_context`` is the heartbeat's RecallResult.nodes
        list; ``apply_memory_writes`` forwards it to
        ``MemoryEngine.explicit_write`` so the extractor LLM can
        skip duplicating content already surfaced in recall.
        Optional — None / [] is acceptable when the caller has no
        recall context to share."""
        ...
