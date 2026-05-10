"""``dispatch`` Engine — run a DecisionResult's side-effects.

Default impl ``LocalDispatchEngine`` wraps the long-standing
``DecisionDispatcher`` class — same per-call task scheduling, same
BatchTracker registration, same memory-write/update plumbing. The
wrapper exposes a single ``dispatch()`` entry point that runs the
four side-effects (log + dispatch tool calls + apply memory writes +
apply memory updates) in order.

A user replacing this Engine controls the entire tool-execution path.
Common reasons to swap:

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
from krakey.engines.catalog import EngineImpl
from krakey.engines.dispatch.default import LocalDispatchEngine

BUILTIN_ENGINES = {
    "default": EngineImpl(
        cls=LocalDispatchEngine,
        description=(
            "In-process dispatch — runs each ToolCall as an asyncio "
            "task, applies memory writes/updates inline."
        ),
    ),
}

DEFAULT_ENGINE = "default"

__all__ = ["BUILTIN_ENGINES", "DEFAULT_ENGINE", "LocalDispatchEngine"]
