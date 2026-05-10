"""Engine Protocols — the swappable-core abstraction layer.

Engine = a core-flow component that the runtime cannot do without. Each
Engine slot has exactly one impl wired at startup; the user picks an
override via ``cfg.core_implementations.<slot>`` (a dotted import path),
otherwise the slot's built-in default is used.

The 10 slots:

  * ``memory``            — MemoryEngine (GM CRUD + KB management + sleep cycle)
  * ``context``           — ContextEngine (prompt assembly)
  * ``embedder``          — EmbedderEngine (text → vector)
  * ``reranker``          — RerankerEngine (with scripted-scoring fallback)
  * ``llm_client_factory``— LLMClientFactoryEngine (tag → LLM client)
  * ``explicit_history``  — ExplicitHistoryEngine (working-memory window)
  * ``decision``          — DecisionEngine (decision text → ToolCall list)
  * ``recall``            — RecallEngine (per-beat memory recall)
  * ``heartbeat``         — HeartbeatEngine (per-beat orchestration)
  * ``dispatch``          — DispatchEngine (executes DecisionResult)

Engine vs Plugin (CLAUDE.md invariant):

  * Plugins (Tool / Channel / Modifier) are **strictly additive** — the
    runtime must complete a heartbeat with all plugins disabled.
  * Engines are **strictly required** — the runtime cannot start until
    every Engine slot is resolved. EngineRegistry fails fast on any
    missing / malformed / Protocol-violating impl.

Engines never import plugin code. Plugins access Engines through the
plugin context's ``services`` dict (Protocol-typed).
"""
from krakey.interfaces.engines.context import ContextEngine
from krakey.interfaces.engines.decision import (
    DecisionEngine,
    DecisionResult,
    ParseFailure,
    ToolCall,
)
from krakey.interfaces.engines.dispatch import DispatchEngine
from krakey.interfaces.engines.embedder import EmbedderEngine
from krakey.interfaces.engines.explicit_history import (
    ExplicitHistoryEngine,
    ExplicitHistoryRound,
)
from krakey.interfaces.engines.heartbeat import HeartbeatEngine
from krakey.interfaces.engines.llm_factory import LLMClientFactoryEngine
from krakey.interfaces.engines.memory import KnowledgeBaseLike, MemoryEngine
from krakey.interfaces.engines.recall import (
    RecallEngine,
    RecallResult,
    RecallSession,
)
from krakey.interfaces.engines.reranker import RerankerEngine

__all__ = [
    "ContextEngine",
    "DecisionEngine",
    "DecisionResult",
    "DispatchEngine",
    "EmbedderEngine",
    "ExplicitHistoryEngine",
    "ExplicitHistoryRound",
    "HeartbeatEngine",
    "KnowledgeBaseLike",
    "LLMClientFactoryEngine",
    "MemoryEngine",
    "ParseFailure",
    "RecallEngine",
    "RecallResult",
    "RecallSession",
    "RerankerEngine",
    "ToolCall",
]
