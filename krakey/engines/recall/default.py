"""``IncrementalRecallEngine`` — default RecallEngine impl.

Stateless factory. Holds the per-beat-recall configuration captured
from cfg + the runtime's memory / embedder / reranker references at
construction time, and yields a fresh ``IncrementalRecall`` session
on every ``new_session()`` call.

The actual recall driver class (``IncrementalRecall``) still lives
under ``krakey.plugins.recall.incremental`` during the migration
window. Step 12 + plugin retirement moves it into
``krakey.engines.recall.incremental`` as the impl's natural home;
until then we just import it from the plugin location.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from krakey.models.config import LLMParams

if TYPE_CHECKING:
    from krakey.interfaces.engines.memory import MemoryEngine
    from krakey.interfaces.engines.recall import RecallSession
    from krakey.memory.recall import AsyncEmbedder, Reranker
    from krakey.models.config import Config


class IncrementalRecallEngine:
    """Default RecallEngine — vec_search-then-rerank-then-budget driver.

    Constructor pulls every dependency the per-beat session needs:
    cfg (for tunables), the MemoryEngine (for vec/fts queries +
    neighbor walks), the embedder (for query vectorization), and the
    optional reranker (with the engine-level fallback always
    present, this is now also always non-None — but the IncrementalRecall
    constructor takes ``Reranker | None`` for back-compat with the
    plugin path).
    """

    def __init__(
        self,
        *,
        cfg: "Config",
        memory: "MemoryEngine",
        embedder: "AsyncEmbedder",
        reranker: "Reranker | None",
    ):
        self._cfg = cfg
        self._memory = memory
        self._embedder = embedder
        self._reranker = reranker

    def new_session(self) -> "RecallSession":
        """Build a fresh per-beat IncrementalRecall session.

        The session captures references to memory + embedder +
        reranker and accumulates stimulus-by-stimulus state across
        ``add_stimuli`` calls until ``finalize()`` produces the
        RecallResult. Heartbeat orchestrator calls this once per
        beat (and once during idle preheat for the next beat).
        """
        from krakey.plugins.recall.incremental import IncrementalRecall

        self_params = (
            self._cfg.llm.core_params("self_thinking") or LLMParams()
        )
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
        )
