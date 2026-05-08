"""``EngineBundle`` — carrier for the 10 resolved Engine instances.

The composition root (``krakey.main.build_runtime_from_config``) calls
``EngineRegistry.resolve_all(deps)`` to get this bundle, then hands
it to ``Runtime(deps, engines=bundle)``. Runtime + all Engines reach
each other through the bundle's Protocol-typed fields — no concrete
impl class is referenced anywhere outside the bundle's construction
site (which is itself just dotted-path string lookup, so the DIP
holds end-to-end).

Every field is required (no defaults, no ``None``). The user's
""装配完成前程序都不会运行"" requirement: Runtime cannot construct
without a fully-populated EngineBundle, and EngineBundle cannot be
populated unless every slot resolves. Failure of any one slot blocks
the whole startup — exactly the fail-fast behaviour the Engine
discipline asks for.
"""
from __future__ import annotations

from dataclasses import dataclass

from krakey.interfaces.engines import (
    ContextEngine,
    DecisionEngine,
    DispatchEngine,
    EmbedderEngine,
    ExplicitHistoryEngine,
    HeartbeatEngine,
    LLMClientFactoryEngine,
    MemoryEngine,
    RecallEngine,
    RerankerEngine,
)


@dataclass
class EngineBundle:
    """Resolved Engine instances. Populated by ``EngineRegistry.resolve_all``;
    consumed by ``Runtime.__init__``. Field types are Protocols, so
    Runtime + downstream Engines can only call surface methods — never
    reach into impl-specific internals.
    """

    memory: MemoryEngine
    context: ContextEngine
    embedder: EmbedderEngine
    reranker: RerankerEngine
    llm_factory: LLMClientFactoryEngine
    explicit_history: ExplicitHistoryEngine
    decision: DecisionEngine
    recall: RecallEngine
    heartbeat: HeartbeatEngine
    dispatch: DispatchEngine
