"""``IncrementalRecallEngine`` — default RecallEngine impl.

Stateless factory. Holds the per-beat-recall configuration captured
from cfg + the runtime's memory / embedder / reranker references at
construction time, and yields a fresh ``IncrementalRecall`` session
on every ``new_session()`` call.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from krakey.engines.recall._internal.incremental import IncrementalRecall
from krakey.engines.recall._internal.enrich import SemanticAssociationEnricher
from krakey.models.config import LLMParams

if TYPE_CHECKING:
    from krakey.interfaces.engines.memory import MemoryEngine
    from krakey.interfaces.engines.recall import RecallSession
    from krakey.interfaces.engines.reranker import RerankerEngine
    from krakey.interfaces.engines.llm_factory import LLMClientFactoryEngine
    from krakey.interfaces.duck import AsyncEmbedder
    from krakey.models.config import Config


class IncrementalRecallEngine:
    """Default RecallEngine — vec_search-then-rerank-then-budget driver.

    Constructor pulls every dependency the per-beat session needs:
    cfg (for tunables), the MemoryEngine (for vec/fts queries +
    neighbor walks), the embedder (for query vectorization), and the
    reranker.
    """

    def __init__(
        self,
        *,
        cfg: "Config",
        memory: "MemoryEngine",
        embedder: "AsyncEmbedder",
        reranker: "RerankerEngine | None",
        factory: "LLMClientFactoryEngine | None" = None,
        config: dict | None = None,
    ):
        self._cfg = cfg
        self._memory = memory
        self._embedder = embedder
        self._reranker = reranker
        self._factory = factory
        self._engine_cfg = config or {}

    def new_session(self) -> "RecallSession":
        """Build a fresh per-beat IncrementalRecall session.

        The session captures references to memory + embedder +
        reranker and accumulates stimulus-by-stimulus state across
        ``add_stimuli`` calls until ``finalize()`` produces the
        RecallResult. Heartbeat orchestrator calls this once per
        beat (and once during idle preheat for the next beat).
        """
        self_params = (
            self._cfg.llm.core_params("self_thinking") or LLMParams()
        )

        enabled = bool(
            self._engine_cfg.get("semantic_association_enabled", False)
        )
        purpose = (self._engine_cfg.get("semantic_association_purpose") or "").strip() or "compact"
        enricher = None
        if enabled and self._factory is not None:
            client = self._factory.client_for_core_purpose(purpose)
            if client is not None:
                enricher = SemanticAssociationEnricher(client)

        return IncrementalRecall(
            self._memory,
            embedder=self._embedder,
            per_stimulus_k=self._cfg.graph_memory.recall_per_stimulus_k,
            recall_token_budget=self_params.recall_token_budget,
            screening_token_multiplier=(
                self._cfg.graph_memory.recall_screening_token_multiplier
            ),
            reranker=self._reranker,
            neighbor_depth=self._cfg.graph_memory.neighbor_expand_depth,
            enricher=enricher,
        )
