"""Prompt assembly + budget enforcement — extracted from the orchestrator.

The phase orchestration in ``HeartbeatOrchestrator.beat()`` and the
mechanics of building Self's input prompt are two responsibilities.
Splitting them lets each file read top-down without context-switching
between "what phase comes next?" and "how is the prompt laid out?".

Owns:

  * ``build_self_prompt`` — assemble the layered prompt
    (DNA / self_model / capabilities / status / recall / history /
    stimulus / in_mind), prepended with BOOTSTRAP_PROMPT during
    bootstrap.
  * ``enforce_input_budget`` — second-line budget defense. If the
    full prompt exceeds the Self role's ``max_input_tokens`` the
    oldest history round is pruned into GM (via compact) and recall
    is re-run; loop until it fits or the window is empty.
  * ``record_prompt`` — append the assembled prompt into the per-run
    ring buffer the dashboard's Prompts tab reads.
  * ``get_genesis_text`` — lazy + cached GENESIS.md load. Bootstrap
    is the only consumer; in steady state the file is never read.

State lives on Runtime; this class holds a back-reference to read
``rt.builder``, ``rt.window``, ``rt.gm``, ``rt.compact_llm``,
``rt.reflects``, ``rt.bootstrap``, ``rt.tentacles``, ``rt.self_model``,
``rt.config``, ``rt.heartbeat_count``, ``rt._prompt_log``,
``rt._genesis_path/_text``, and to call ``orchestrator.new_recall``
during budget enforcement.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from src.bootstrap import BOOTSTRAP_PROMPT, load_genesis
from src.models.config import LLMParams

if TYPE_CHECKING:
    from src.prompt.views import CapabilityView, StatusSnapshot
    from src.runtime.heartbeat.heartbeat_orchestrator import _GMCounts
    from src.runtime.runtime import Runtime


class PromptAssembler:
    """Builds Self's prompt + enforces the input-token budget."""

    def __init__(self, runtime: "Runtime"):
        self._rt = runtime

    def build_self_prompt(self, stimuli, recall_result,
                          counts: "_GMCounts") -> str:
        rt = self._rt
        # Suppress the [ACTION FORMAT] layer when a hypothalamus
        # Reflect is registered: the translator owns the dispatch
        # path, and teaching Self structured tags would conflict with
        # its job. See docs/design/reflects-and-self-model.md
        # Reflect #1 design.
        in_mind_state = rt.reflects.in_mind_state()
        in_mind_instructions: str | None = None
        if in_mind_state is not None:
            from src.plugins.default_in_mind.prompt import (
                IN_MIND_INSTRUCTIONS_LAYER,
            )
            in_mind_instructions = IN_MIND_INSTRUCTIONS_LAYER
        prompt = rt.builder.build(
            self_model=rt.self_model,
            capabilities=self._capabilities(),
            status=self._status(counts.node_count, counts.edge_count,
                                counts.fatigue_pct, counts.fatigue_hint),
            recall=recall_result,
            window=rt.window.get_rounds(),
            stimuli=stimuli,
            current_time=datetime.now(),
            suppress_action_format=rt.reflects.has_hypothalamus(),
            in_mind=in_mind_state,
            in_mind_instructions=in_mind_instructions,
        )
        if rt.bootstrap.should_inject_intro_prompt():
            prompt = (BOOTSTRAP_PROMPT.format(
                          genesis_text=self.get_genesis_text())
                      + "\n\n" + prompt)
        return prompt

    async def enforce_input_budget(self, stimuli, recall_result,
                                   counts: "_GMCounts"):
        """Overall prompt-budget enforcement (DevSpec §10.2).

        After recall is finalized and we have a candidate prompt, if
        the full prompt exceeds the Self role's ``max_input_tokens``,
        prune the oldest history round into GM (normal compact path)
        and re-run recall (GM changed → new nodes may be more
        relevant). Repeat until the prompt fits or the window is empty.

        This is the second line of defense: ``_phase_compact`` already
        caps history at ``max_input_tokens * history_token_fraction``,
        but the rest of the prompt (DNA + self-model + capabilities +
        stimulus + recall + status) can push the total over budget
        even when history is within its own share. When that happens
        we borrow from history (oldest rounds are least valuable) and
        promote them to GM so nothing is lost.

        Returns the final (prompt, recall_result) pair. Hard cap on
        iterations so a pathological configuration can't spin forever.
        """
        from src.runtime.heartbeat.compact import compact_round
        from src.utils.tokens import estimate_tokens

        rt = self._rt
        self_params = rt.config.llm.core_params("self_thinking") or LLMParams()
        budget = int(self_params.max_input_tokens or 128_000)

        async def _recall_fn(text: str):
            return await rt.gm.fts_search(text, top_k=10)

        prompt = self.build_self_prompt(stimuli, recall_result, counts)
        max_iters = 10  # safety bound — should never need more than 2-3
        for _ in range(max_iters):
            total = estimate_tokens(prompt)
            if total <= budget:
                return prompt, recall_result
            if not rt.window.rounds:
                rt.log.hb_warn(
                    f"prompt {total} > max_input_tokens {budget} and "
                    "window is empty; sending anyway"
                )
                return prompt, recall_result
            oldest = rt.window.pop_oldest()
            assert oldest is not None
            rt.log.hb(
                f"input budget: prompt {total} > {budget}; pruning oldest "
                f"round (heartbeat #{oldest.heartbeat_id}) into GM"
            )
            try:
                await compact_round(oldest, rt.gm, rt.compact_llm,
                                    _recall_fn)
            except Exception as e:  # noqa: BLE001 — never crash the beat
                rt.log.hb_warn(
                    f"budget-driven compact failed: {e} — round "
                    f"#{oldest.heartbeat_id} dropped without GM write"
                )
            fresh = rt._orchestrator.new_recall()
            await fresh.add_stimuli(stimuli)
            recall_result = await fresh.finalize()
            rt._recall = fresh
            prompt = self.build_self_prompt(stimuli, recall_result, counts)
        rt.log.hb_warn(
            f"input budget not satisfied after {max_iters} prune "
            f"iterations; sending oversized prompt "
            f"({estimate_tokens(prompt)} > {budget})"
        )
        return prompt, recall_result

    def get_genesis_text(self) -> str:
        """Lazy-load GENESIS.md on first use.

        Bootstrap is the ONLY consumer of this text — after
        bootstrap_complete flips to True, the agent should never see
        GENESIS again. Reading the file unconditionally at startup
        was both wasteful I/O (80% of runs are steady-state) and a
        correctness trap.

        Cached on first call so repeat heartbeats during a long
        Bootstrap don't re-read the file 50 times.
        """
        rt = self._rt
        if rt._genesis_text is None:
            rt._genesis_text = load_genesis(rt._genesis_path)
        return rt._genesis_text

    def record_prompt(self, heartbeat_id: int, prompt: str) -> None:
        self._rt._prompt_log.append({
            "heartbeat_id": heartbeat_id,
            "ts": datetime.now().isoformat(),
            "full_prompt": prompt,
        })

    # ---- private prompt-section helpers ---------------------------------

    def _capabilities(self) -> list["CapabilityView"]:
        """Tentacle list for the [CAPABILITIES] layer. Only changes on
        plugin reload, so this gets rendered high in the prompt above
        the cache-breaking volatile layers."""
        from src.prompt.views import CapabilityView
        return [
            CapabilityView(name=t["name"], description=t["description"])
            for t in self._rt.tentacles.list_descriptions()
        ]

    def _status(self, node_count: int, edge_count: int,
                pct: int, hint: str) -> "StatusSnapshot":
        """Runtime status numbers — changes every beat (heartbeat
        counter, fatigue), so this section is deliberately placed near
        the end of the prompt to preserve the cacheable prefix above it."""
        from src.prompt.views import StatusSnapshot
        return StatusSnapshot(
            gm_node_count=node_count,
            gm_edge_count=edge_count,
            fatigue_pct=pct,
            fatigue_hint=hint,
            last_sleep_time="never",
            heartbeats_since_sleep=self._rt.heartbeat_count,
        )
