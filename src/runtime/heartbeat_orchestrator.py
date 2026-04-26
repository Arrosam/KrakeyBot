"""Per-beat heartbeat algorithm — extracted from Runtime.

Owns the orchestration of one heartbeat: phase ordering, sleep
short-circuits, override consumption, the prompt-build / Self-call /
hypothalamus-dispatch / hibernate flow.

Composition over inheritance: takes a ``Runtime`` reference and reads
state through it (``rt.gm``, ``rt.window``, ``rt.heartbeat_count``,
…). Almost every Runtime field gets touched here, so wiring the
orchestrator with 13+ narrow protocols would just reproduce Runtime's
shape — pass the runtime ref and accept the verbosity. The split
buys two things anyway:

  1. **Read clarity** — Runtime is now ~600 lines of resources +
     lifecycle; the heartbeat algorithm is in one file.
  2. **Stable seam for future replacement** — a different orchestrator
     (e.g. test fake, or Phase 3 multi-stage scheduler) can be
     swapped in without touching Runtime construction.

The orchestrator owns NO state of its own. All beat-loop counters
(``heartbeat_count``, ``_recall``, ``_classify_tasks``, …) live on
Runtime so the existing test surface keeps reading them directly.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.bootstrap import BOOTSTRAP_PROMPT, load_genesis
from src.models.config import LLMParams
from src.models.stimulus import Stimulus
from src.prompt.views import SlidingWindowRound
from src.runtime.compact import compact_if_needed
from src.runtime.event_bus import (
    DecisionEvent, GMStatsEvent, HeartbeatStartEvent, HibernateEvent,
    NoteEvent, PromptBuiltEvent, SleepDoneEvent, SleepStartEvent,
    StimuliQueuedEvent, ThinkingEvent,
)
from src.runtime.fatigue import calculate_fatigue
from src.runtime.hibernate import hibernate_with_recall
from src.memory.sleep.sleep_manager import enter_sleep_mode
from src.runtime.override_commands import (
    OverrideAction, handle_override, parse_override,
)
from src.self_agent import parse_self_output

if TYPE_CHECKING:
    from src.memory.recall import IncrementalRecall
    from src.prompt.views import CapabilityView, StatusSnapshot
    from src.runtime.runtime import Runtime


MAX_RECALL_RETRIES = 1
"""Cap on uncovered stimulus re-tries to prevent infinite pushback loops
when GM has no related nodes yet (e.g. first-ever user message)."""


@dataclass
class _GMCounts:
    """Snapshot from one heartbeat's fatigue phase, threaded into later
    phases so they don't re-query GM redundantly."""
    node_count: int
    edge_count: int
    fatigue_pct: int
    fatigue_hint: str


def _delta_str(delta: int) -> str:
    if delta > 0:
        return f" (+{delta})"
    if delta < 0:
        return f" ({delta})"
    return ""


def _summarize_stimuli(stimuli: list[Stimulus]) -> str:
    """Render the stimulus list for persistence in a ``SlidingWindowRound``.

    This text is what Self sees in the ``[HISTORY]`` layer every
    subsequent beat — truncation here is destructive: downstream
    mechanisms (recall-anchor extraction, compact summarization,
    bootstrap-signal detection, user-message echo checks) all rely on
    the full content. The window's token budget handles overflow via
    compact_if_needed, so we don't need a blunt character cap here.
    """
    if not stimuli:
        return "(none)"
    return " | ".join(f"{s.source}: {s.content}" for s in stimuli)


class HeartbeatOrchestrator:
    """Runs one heartbeat per ``beat()`` call. Pure logic over Runtime
    state — owns no fields of its own."""

    def __init__(self, runtime: "Runtime"):
        self._rt = runtime

    # ---- one full beat -------------------------------------------------

    async def beat(self) -> None:
        """Orchestrates one heartbeat. Sleep can short-circuit at two
        points (force-sleep at fatigue threshold, voluntary sleep from
        Self's [DECISION]). Override commands (/kill, /sleep) can also
        short-circuit."""
        rt = self._rt
        rt.heartbeat_count += 1
        rt.log.set_heartbeat(rt.heartbeat_count)
        stimuli = await self._phase_drain_and_seed_recall()

        # Override commands run out-of-band (Self never sees /<cmd>).
        stimuli, override_action = await self._phase_handle_overrides(stimuli)
        if override_action is OverrideAction.KILL:
            return
        if override_action is OverrideAction.SLEEP:
            await self._perform_sleep(
                "manual /sleep override",
                wake_msg="完成了一次手动触发的睡眠。",
            )
            return

        counts = await self._phase_compute_fatigue()

        if counts.fatigue_pct >= rt.config.fatigue.force_sleep_threshold:
            await self._perform_sleep(
                f"force-sleep at fatigue {counts.fatigue_pct}%",
                wake_msg="之前因过于疲劳昏睡过去了。",
            )
            return

        await self._phase_compact()
        recall_result = await self._phase_finalize_recall_and_pushback()
        parsed = await self._phase_run_self(stimuli, recall_result, counts)
        if parsed is None:
            return  # Self LLM error already logged + slept
        self._phase_save_round(parsed, stimuli)
        self._phase_log_self_output(parsed)
        if rt.bootstrap.is_active:
            self._phase_apply_bootstrap_signals(parsed)
        await self._phase_auto_ingest_feedback(stimuli)
        sleep_requested = await self._phase_apply_hypothalamus(
            parsed, recall_result,
        )
        if sleep_requested:
            await self._perform_sleep(
                "voluntary sleep requested by Self",
                wake_msg=("完成了一次完整睡眠 (聚类 + KB 迁移 + Index 重建)。"
                          "醒来感觉清爽一些。"),
            )
            return
        self._phase_schedule_classify()
        await self._phase_hibernate(parsed, recall_result)

    # ---- phases --------------------------------------------------------

    async def _phase_handle_overrides(
        self, stimuli: list[Stimulus],
    ) -> tuple[list[Stimulus], "OverrideAction | None"]:
        """Scan drained stimuli for /<cmd>. Each match is consumed (Self
        never sees it) and executed out-of-band. Returns the filtered
        stimulus list + the highest-priority action (KILL > SLEEP)."""
        rt = self._rt
        filtered: list[Stimulus] = []
        triggered: OverrideAction | None = None
        for s in stimuli:
            if s.type != "user_message":
                filtered.append(s)
                continue
            cmd = parse_override(s.content)
            if cmd is None:
                filtered.append(s)
                continue
            result = await handle_override(cmd, rt)
            rt.log.hb(f"override /{cmd}: {result.output}")
            if result.action is OverrideAction.KILL:
                triggered = OverrideAction.KILL
                rt._stop = True
                break
            if (result.action is OverrideAction.SLEEP
                    and triggered is not OverrideAction.KILL):
                triggered = OverrideAction.SLEEP
            # Self can still see informational overrides as system events
            if result.action is OverrideAction.NONE:
                await rt.buffer.push(Stimulus(
                    type="system_event", source="system:override",
                    content=f"/{cmd}: {result.output}",
                    timestamp=datetime.now(), adrenalin=False,
                ))
        return filtered, triggered

    async def _phase_drain_and_seed_recall(self) -> list[Stimulus]:
        rt = self._rt
        stimuli = rt.buffer.drain()
        rt.log.hb(f"stimuli={len(stimuli)} (thinking...)")
        rt.events.publish(HeartbeatStartEvent(
            heartbeat_id=rt.heartbeat_count, stimulus_count=len(stimuli),
        ))
        rt.events.publish(StimuliQueuedEvent(stimuli=[
            {"type": s.type, "source": s.source, "content": s.content,
             "adrenalin": s.adrenalin, "ts": s.timestamp.isoformat()}
            for s in stimuli
        ]))
        assert rt._recall is not None
        already = {id(s) for s in rt._recall.processed_stimuli}
        new_for_recall = [s for s in stimuli if id(s) not in already]
        if new_for_recall:
            await rt._recall.add_stimuli(new_for_recall)
        return stimuli

    async def _phase_compute_fatigue(self) -> _GMCounts:
        rt = self._rt
        node_count = await rt.gm.count_nodes()
        edge_count = await rt.gm.count_edges()
        pct, hint = calculate_fatigue(
            node_count=node_count,
            soft_limit=rt.config.fatigue.gm_node_soft_limit,
            thresholds=rt.config.fatigue.thresholds,
        )
        node_delta = node_count - rt._last_node_count
        edge_delta = edge_count - rt._last_edge_count
        rt._last_node_count = node_count
        rt._last_edge_count = edge_count
        rt.log.hb(
            f"gm: nodes={node_count}{_delta_str(node_delta)}, "
            f"edges={edge_count}{_delta_str(edge_delta)}, fatigue={pct}%"
        )
        if pct >= rt.config.fatigue.force_sleep_threshold:
            rt.log.hb_warn(f"force-sleep threshold reached (fatigue={pct}%);"
                              " Sleep mode lands in Phase 2.")
        rt.events.publish(GMStatsEvent(
            heartbeat_id=rt.heartbeat_count,
            node_count=node_count, edge_count=edge_count, fatigue_pct=pct,
        ))
        return _GMCounts(node_count=node_count, edge_count=edge_count,
                          fatigue_pct=pct, fatigue_hint=hint)

    async def _phase_compact(self) -> None:
        rt = self._rt
        async def _recall_fn(text: str):
            return await rt.gm.fts_search(text, top_k=10)
        await compact_if_needed(rt.window, rt.gm, rt.compact_llm,
                                 recall_fn=_recall_fn)

    async def _phase_finalize_recall_and_pushback(self):
        """Finalize recall + cap-1 retry of uncovered stimuli."""
        rt = self._rt
        assert rt._recall is not None
        recall_result = await rt._recall.finalize()
        for s in recall_result.uncovered_stimuli:
            retries = s.metadata.get("recall_retries", 0)
            if retries >= MAX_RECALL_RETRIES:
                continue
            s.metadata["recall_retries"] = retries + 1
            await rt.buffer.push(s)
        return recall_result

    async def _phase_run_self(self, stimuli, recall_result,
                                counts: "_GMCounts"):
        """Build prompt + call Self LLM + parse. Returns None on LLM error
        (sleeps min_interval and short-circuits the heartbeat)."""
        rt = self._rt
        prompt, recall_result = await self.enforce_input_budget(
            stimuli, recall_result, counts,
        )
        self.record_prompt(rt.heartbeat_count, prompt)
        rt.events.publish(PromptBuiltEvent(
            heartbeat_id=rt.heartbeat_count,
            layers={"full_prompt": prompt},
        ))
        try:
            raw = await rt.self_llm.chat(
                [{"role": "user", "content": prompt}]
            )
        except Exception as e:  # noqa: BLE001
            rt.log.hb(f"Self LLM error: {e}")
            await asyncio.sleep(rt._min)
            return None
        return parse_self_output(raw)

    def _phase_save_round(self, parsed, stimuli) -> None:
        rt = self._rt
        rt.window.append(SlidingWindowRound(
            heartbeat_id=rt.heartbeat_count,
            stimulus_summary=_summarize_stimuli(stimuli),
            decision_text=parsed.decision,
            note_text=parsed.note,
        ))

    def _phase_apply_bootstrap_signals(self, parsed) -> None:
        """During Bootstrap, hand Self's NOTE to the coordinator for
        self-model patch + completion-marker detection."""
        rt = self._rt
        result = rt.bootstrap.apply_note_signals(parsed.note)
        if result.update or result.completed:
            rt.self_model = rt._self_model_store.load()
        if result.update:
            rt.log.hb(f"bootstrap: self-model updated "
                          f"({list(result.update.keys())})")
        if result.completed:
            rt.log.hb("bootstrap: complete — entering normal operation")

    def _phase_log_self_output(self, parsed) -> None:
        rt = self._rt
        decision_text = parsed.decision.strip() or "(none)"
        rt.log.hb_thought("decision", decision_text)
        rt.events.publish(DecisionEvent(
            heartbeat_id=rt.heartbeat_count, text=decision_text,
        ))
        if parsed.thinking:
            rt.log.hb_thought("thinking", parsed.thinking)
            rt.events.publish(ThinkingEvent(
                heartbeat_id=rt.heartbeat_count,
                text=parsed.thinking.strip(),
            ))
        if parsed.note:
            rt.log.hb_thought("note", parsed.note)
            rt.events.publish(NoteEvent(
                heartbeat_id=rt.heartbeat_count, text=parsed.note.strip(),
            ))

    async def _phase_auto_ingest_feedback(self, stimuli) -> None:
        rt = self._rt
        for s in stimuli:
            if s.type != "tentacle_feedback":
                continue
            try:
                await rt.gm.auto_ingest(
                    s.content, source_heartbeat=rt.heartbeat_count,
                )
            except Exception as e:  # noqa: BLE001
                rt.log.runtime_error(f"auto_ingest error: {e}")

    async def _phase_apply_hypothalamus(self, parsed, recall_result) -> bool:
        """Convert Self's response into tentacle calls + dispatch.

        Routes through ``rt.reflects.dispatch_decision`` which picks
        the path:
          * Hypothalamus Reflect registered → LLM translation of
            ``parsed.decision`` (existing behavior).
          * No Hypothalamus Reflect → script-only action executor
            scans ``parsed.raw`` for ``[ACTION]...[/ACTION]`` JSONL.

        Returns True iff Self requested sleep.
        """
        rt = self._rt
        decision = parsed.decision.strip().lower()
        if not decision or decision in ("no action", "无行动"):
            return False
        try:
            result = await rt.reflects.dispatch_decision(
                parsed.raw, parsed.decision,
                rt.tentacles.list_descriptions(),
            )
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e!r}"
            rt.log.hb(f"Hypothalamus error: {err}")
            await rt.buffer.push(Stimulus(
                type="system_event", source="system:hypothalamus",
                content=(
                    "Your last [DECISION] could not be translated by the "
                    f"Hypothalamus ({err}). Nothing was dispatched. "
                    "Try re-stating the intent more explicitly next beat."
                ),
                timestamp=datetime.now(),
                adrenalin=True,
            ))
            return False
        rt._dispatcher.log_summary(rt.heartbeat_count, result)
        await rt._dispatcher.dispatch_tentacle_calls(
            rt.heartbeat_count, result.tentacle_calls,
        )
        await rt._dispatcher.apply_memory_writes(
            result.memory_writes, recall_result.nodes,
            rt.heartbeat_count,
        )
        await rt._dispatcher.apply_memory_updates(result.memory_updates)
        return bool(result.sleep)

    def _phase_schedule_classify(self) -> None:
        """Background classify+link doesn't block the heartbeat."""
        rt = self._rt
        rt._classify_tasks.append(
            asyncio.create_task(rt.gm.classify_and_link_pending()),
        )

    async def _phase_hibernate(self, parsed, recall_result) -> None:
        rt = self._rt
        if recall_result.uncovered_stimuli:
            base = rt.config.hibernate.min_interval
        else:
            base = (parsed.hibernate_seconds
                    or rt.config.hibernate.default_interval)
        # Bootstrap-mode cadence (DevSpec §12.2) — coordinator returns
        # the bootstrap-fixed value when active, else passes ``base``
        # through unchanged.
        interval = rt.bootstrap.hibernate_interval(default=base)
        rt.log.hb(f"hibernate {interval}s")
        rt.events.publish(HibernateEvent(
            heartbeat_id=rt.heartbeat_count, interval_seconds=interval,
        ))
        rt._recall = self.new_recall()
        await hibernate_with_recall(
            interval, rt.buffer, rt._recall,
            min_interval=rt._min, max_interval=rt._max,
        )

    # ---- prompt assembly (used by _phase_run_self) ----------------------

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
        from src.runtime.compact import compact_round
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
            fresh = self.new_recall()
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

    def new_recall(self) -> "IncrementalRecall":
        # Routed through the Reflect registry (kind="recall_anchor").
        # Default built-in mirrors the previous in-line factory; future
        # Reflects (#2 LLM-anchor) replace it from config without
        # Runtime needing to know.
        return self._rt.reflects.make_recall(self._rt)

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

    # ---- sleep ---------------------------------------------------------

    async def _perform_sleep(self, reason: str, *, wake_msg: str) -> None:
        """Run 7-phase Sleep, persist self-model bookkeeping, push wake-up
        stimulus, reset incremental recall (GM state changed)."""
        rt = self._rt
        rt.log.hb(f"sleep started — {reason}")
        rt.events.publish(SleepStartEvent(reason=reason))
        try:
            sl = rt.config.sleep
            stats = await enter_sleep_mode(
                rt.gm, rt.kb_registry, rt.buffer,
                llm=rt.compact_llm, embedder=rt.embedder,
                log_dir=rt.sleep_log_dir,
                min_community_size=sl.min_community_size,
                kb_consolidation_threshold=sl.kb_consolidation_threshold,
                kb_index_max=sl.kb_index_max,
                kb_archive_pct=sl.kb_archive_pct,
                kb_revive_threshold=sl.kb_revive_threshold,
            )
        except Exception as e:  # noqa: BLE001
            rt.log.hb(f"sleep failed: {e}")
            return
        rt.events.publish(SleepDoneEvent(stats=stats))
        rt.log.hb(
            f"sleep done: facts_migrated={stats['facts_migrated']}, "
            f"focus_cleared={stats['focus_cleared']}, "
            f"kbs={stats['kbs_created']}, index_nodes={stats['index_nodes']}"
        )
        # Sleep bookkeeping is a per-process runtime concern, not
        # something Self needs to remember across restarts.
        rt._sleep_cycles += 1
        # Wake-up stimulus
        await rt.buffer.push(Stimulus(
            type="system_event", source="system:sleep",
            content=wake_msg, timestamp=datetime.now(),
            adrenalin=False,
        ))
        # GM changed underneath us — start a fresh recall
        rt._recall = self.new_recall()
