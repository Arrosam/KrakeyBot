"""``dispatch`` Engine — run a DecisionResult's side-effects.

Default impl ``LocalDispatchEngine`` (in ``default.py``) wraps the
existing ``DecisionDispatcher`` class — same per-call task scheduling,
same BatchTracker registration, same memory-write/update plumbing. The
wrapper exposes a single ``dispatch()`` entry point that orchestrates
the four side-effects in order, replacing the orchestrator's previous
4-call sequence (``log_summary`` → ``dispatch_tool_calls`` →
``apply_memory_writes`` → ``apply_memory_updates``) with one method.

A user replacing this Engine via ``cfg.core_implementations.dispatch``
controls the entire tool-execution path. Common reasons to swap:

  * Push tool execution to a remote worker (HTTP / RPC) — dispatch
    sends the call, the worker runs ``tool.execute()``, the engine
    returns the resulting Stimulus.
  * Add per-tool retry / rate-limit / approval-gate policies.
  * Audit-log every dispatched call.

Distinct from the per-tool ``Environment`` mechanism (which routes
CLI commands the tool itself produces): Dispatch-level virtualization
replaces the entire ``tool.execute()`` invocation, not just commands
the tool decides to run.

The ``DispatchEngine`` Protocol the runtime depends on lives at
``krakey.interfaces.engines.dispatch``.
"""
from krakey.engines.dispatch.default import LocalDispatchEngine

__all__ = ["LocalDispatchEngine"]
