"""Emergency dotted-path fallback for engine slots whose ``meta.yaml``
is missing / corrupt / points at an unimportable class.

Pure strings — this module is **deliberately decoupled** from any
actual engine code. Importing ``engine_system.defaults`` does not pull
in a single engine module; the dotted paths are resolved by
``importlib`` only when ``EngineRegistry`` actually needs to fall
back.

The intent: a damaged ``engines/`` tree (someone deleted ``meta.yaml``
or it has a typo) shouldn't bring down the runtime — it should
degrade to a known-good default and log loudly so the user can fix
the meta. The fallback always points at the in-tree default impl
(``GraphMemoryEngine``, ``DefaultRerankerEngine``, etc.); a user
deploying their own custom engine can override the slot via
``cfg.core_implementations.<slot>`` like always.

If you add a new slot, add an entry here too — otherwise a missing
``meta.yaml`` for that slot means startup fails with no recovery
path.
"""
from __future__ import annotations


# slot → "module.path:ClassName" (entry-point syntax).
FALLBACK_ENGINES: dict[str, str] = {
    "memory":
        "krakey.engines.memory.default:GraphMemoryEngine",
    "context":
        "krakey.engines.context.default:PromptBuilderContextEngine",
    "embedder":
        "krakey.engines.embedder.default:DefaultEmbedderEngine",
    "reranker":
        "krakey.engines.reranker.default:DefaultRerankerEngine",
    "recall":
        "krakey.engines.recall.default:IncrementalRecallEngine",
    "explicit_history":
        "krakey.engines.explicit_history.default:SlidingWindowEngine",
    "heartbeat":
        "krakey.engines.heartbeat.default:DefaultHeartbeatEngine",
    "dispatch":
        "krakey.engines.dispatch.default:DefaultDispatchEngine",
    "decision":
        "krakey.engines.decision._internal.tool_call_parser:ToolCallParserDecisionEngine",
    "llm_factory":
        "krakey.engines.llm_factory.default:DefaultLLMClientFactoryEngine",
    "llm_client_factory":
        "krakey.engines.llm_client_factory._client:LLMClient",
}
