"""CogniBot entrypoint + main heartbeat loop (DevSpec §6.4).

Phase 1 wiring: GraphMemory + IncrementalRecall + SlidingWindow + compact +
fatigue calc + BatchTracker + async classify. Phase 2 will add Bootstrap,
KnowledgeBase, and full Sleep.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Protocol

from src.hypothalamus import Hypothalamus, TentacleCall
from src.interfaces.sensory import SensoryRegistry
from src.interfaces.tentacle import TentacleRegistry
from src.llm.client import LLMClient
from src.memory.graph_memory import GraphMemory
from src.memory.recall import IncrementalRecall, Reranker
from src.models.config import Config, load_config
from src.models.stimulus import Stimulus
from src.prompt.builder import PromptBuilder, SlidingWindowRound
from src.runtime.batch_tracker import BatchTrackerSensory
from src.runtime.colors import cyan, green, yellow
from src.runtime.compact import compact_if_needed
from src.runtime.fatigue import calculate_fatigue
from src.runtime.hibernate import hibernate_with_recall
from src.runtime.sliding_window import SlidingWindow
from src.runtime.stimulus_buffer import StimulusBuffer
from src.self_agent import parse_self_output
from src.sensories.cli_input import CliInputSensory
from src.tentacles.action import ActionTentacle
from src.tentacles.memory_recall import MemoryRecallTentacle


class ChatLike(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...


class AsyncEmbedder(Protocol):
    async def __call__(self, text: str) -> list[float]: ...


@dataclass
class RuntimeDeps:
    config: Config
    self_llm: ChatLike
    hypo_llm: ChatLike
    action_llm: ChatLike
    compact_llm: ChatLike
    classify_llm: ChatLike
    embedder: AsyncEmbedder
    reranker: Reranker | None = None
    reader: Callable[[], Awaitable[str | None]] | None = None


MAX_RECALL_RETRIES = 1
"""Cap on uncovered stimulus re-tries to prevent infinite pushback loops
when GM has no related nodes yet (e.g. first-ever user message)."""


@dataclass
class _GMCounts:
    """Snapshot from one heartbeat's fatigue phase, threaded into later phases
    so they don't re-query GM redundantly."""
    node_count: int
    edge_count: int
    fatigue_pct: int
    fatigue_hint: str


HARDCODED_SELF_MODEL: dict[str, Any] = {
    "identity": {"name": "Krakey", "persona": "nascent cognitive entity"},
    "state": {"mood_baseline": "neutral", "energy_level": 1.0,
              "focus_topic": "", "is_sleeping": False,
              "bootstrap_complete": True},
    "goals": {"active": [], "completed": []},
    "relationships": {"users": []},
    "statistics": {"total_heartbeats": 0, "total_sleep_cycles": 0,
                   "uptime_hours": 0.0, "first_boot": "",
                   "last_heartbeat": "", "last_sleep": ""},
}


class Runtime:
    def __init__(self, deps: RuntimeDeps, *, hibernate_min: float | None = None,
                 hibernate_max: float | None = None):
        self.config = deps.config
        self.self_llm = deps.self_llm
        self.compact_llm = deps.compact_llm
        self.embedder = deps.embedder
        self.reranker = deps.reranker
        self.hypothalamus = Hypothalamus(deps.hypo_llm)
        self.buffer = StimulusBuffer()
        self.window = SlidingWindow(
            max_tokens=self.config.sliding_window.max_tokens,
        )
        self.builder = PromptBuilder()

        gm_path = self.config.graph_memory.db_path or ":memory:"
        self.gm = GraphMemory(
            gm_path,
            embedder=deps.embedder,
            auto_ingest_threshold=self.config.graph_memory.auto_ingest_similarity_threshold,
            extractor_llm=deps.classify_llm,
            classifier_llm=deps.classify_llm,
        )

        self.tentacles = TentacleRegistry()
        self.tentacles.register(ActionTentacle(
            llm=deps.action_llm,
            max_context_tokens=self.config.tentacle.get("action", {})
                .get("max_context_tokens", 4096),
        ))
        self.tentacles.register(MemoryRecallTentacle(
            gm=self.gm, embedder=self.embedder,
        ))

        self.sensories = SensoryRegistry()
        if self.config.sensory.get("cli_input", {}).get("enabled", False):
            self.sensories.register(CliInputSensory(
                default_adrenalin=self.config.sensory["cli_input"]
                    .get("default_adrenalin", True),
                reader=deps.reader,
            ))
        self.batch_tracker = BatchTrackerSensory()
        self.sensories.register(self.batch_tracker)

        self.self_model = dict(HARDCODED_SELF_MODEL)
        self.heartbeat_count = 0
        self._stop = False
        self._min = hibernate_min if hibernate_min is not None else self.config.hibernate.min_interval
        self._max = hibernate_max if hibernate_max is not None else self.config.hibernate.max_interval

        self._recall: IncrementalRecall | None = None
        self._classify_tasks: list[asyncio.Task] = []
        self._last_node_count = 0
        self._last_edge_count = 0

    def _new_recall(self) -> IncrementalRecall:
        return IncrementalRecall(
            self.gm,
            embedder=self.embedder,
            per_stimulus_k=self.config.graph_memory.recall_per_stimulus_k,
            max_recall_nodes=self.config.graph_memory.max_recall_nodes,
            reranker=self.reranker,
            neighbor_depth=self.config.graph_memory.neighbor_expand_depth,
        )

    async def run(self, iterations: int | None = None) -> None:
        await self.gm.initialize()
        await self.sensories.start_all(self.buffer)
        self._recall = self._new_recall()
        try:
            count = 0
            while not self._stop:
                await self._heartbeat()
                count += 1
                if iterations is not None and count >= iterations:
                    return
        finally:
            await self.sensories.stop_all()
            # Cancel in-flight background classify tasks so asyncio doesn't warn.
            pending = [t for t in self._classify_tasks if not t.done()]
            for t in pending:
                t.cancel()

    async def close(self) -> None:
        """Shut down persistent resources (GM connection). Idempotent."""
        await self.gm.close()

    async def _heartbeat(self) -> None:
        """Orchestration only. Each phase is its own method."""
        self.heartbeat_count += 1
        stimuli = await self._phase_drain_and_seed_recall()
        counts = await self._phase_compute_fatigue()
        await self._phase_compact()
        recall_result = await self._phase_finalize_recall_and_pushback()
        parsed = await self._phase_run_self(stimuli, recall_result, counts)
        if parsed is None:
            return  # Self LLM error already logged + slept
        self._phase_save_round(parsed, stimuli)
        self._phase_log_self_output(parsed)
        await self._phase_auto_ingest_feedback(stimuli)
        await self._phase_apply_hypothalamus(parsed, recall_result)
        self._phase_schedule_classify()
        await self._phase_hibernate(parsed, recall_result)

    # ---------- heartbeat phases ----------

    async def _phase_drain_and_seed_recall(self) -> list[Stimulus]:
        stimuli = self.buffer.drain()
        print(f"[HB #{self.heartbeat_count}] stimuli={len(stimuli)} "
              "(thinking...)", flush=True)
        assert self._recall is not None
        already = {id(s) for s in self._recall.processed_stimuli}
        new_for_recall = [s for s in stimuli if id(s) not in already]
        if new_for_recall:
            await self._recall.add_stimuli(new_for_recall)
        return stimuli

    async def _phase_compute_fatigue(self) -> _GMCounts:
        node_count = await self.gm.count_nodes()
        edge_count = await self.gm.count_edges()
        pct, hint = calculate_fatigue(
            node_count=node_count,
            soft_limit=self.config.fatigue.gm_node_soft_limit,
            thresholds=self.config.fatigue.thresholds,
        )
        node_delta = node_count - self._last_node_count
        edge_delta = edge_count - self._last_edge_count
        self._last_node_count = node_count
        self._last_edge_count = edge_count
        print(
            f"[HB #{self.heartbeat_count}] gm: nodes={node_count}"
            f"{_delta_str(node_delta)}, edges={edge_count}"
            f"{_delta_str(edge_delta)}, fatigue={pct}%",
            flush=True,
        )
        if pct >= self.config.fatigue.force_sleep_threshold:
            print(f"[runtime] force-sleep threshold reached "
                  f"(fatigue={pct}%); Sleep mode lands in Phase 2.",
                  file=sys.stderr, flush=True)
        return _GMCounts(node_count=node_count, edge_count=edge_count,
                          fatigue_pct=pct, fatigue_hint=hint)

    async def _phase_compact(self) -> None:
        async def _recall_fn(text: str):
            return await self.gm.fts_search(text, top_k=10)
        await compact_if_needed(self.window, self.gm, self.compact_llm,
                                 recall_fn=_recall_fn)

    async def _phase_finalize_recall_and_pushback(self):
        """Finalize recall + cap-1 retry of uncovered stimuli."""
        assert self._recall is not None
        recall_result = await self._recall.finalize()
        for s in recall_result.uncovered_stimuli:
            retries = s.metadata.get("recall_retries", 0)
            if retries >= MAX_RECALL_RETRIES:
                continue
            s.metadata["recall_retries"] = retries + 1
            await self.buffer.push(s)
        return recall_result

    async def _phase_run_self(self, stimuli, recall_result,
                                counts: "_GMCounts"):
        """Build prompt + call Self LLM + parse. Returns None on LLM error
        (sleeps min_interval and short-circuits the heartbeat)."""
        prompt = self.builder.build(
            self_model=self.self_model,
            status=self._status(counts.node_count, counts.edge_count,
                                  counts.fatigue_pct, counts.fatigue_hint),
            recall={"nodes": recall_result.nodes,
                    "edges": recall_result.edges},
            window=self.window.get_rounds(),
            stimuli=stimuli,
        )
        try:
            raw = await self.self_llm.chat(
                [{"role": "user", "content": prompt}]
            )
        except Exception as e:  # noqa: BLE001
            print(f"[HB #{self.heartbeat_count}] Self LLM error: {e}",
                  flush=True)
            await asyncio.sleep(self._min)
            return None
        return parse_self_output(raw)

    def _phase_save_round(self, parsed, stimuli) -> None:
        self.window.append(SlidingWindowRound(
            heartbeat_id=self.heartbeat_count,
            stimulus_summary=_summarize_stimuli(stimuli),
            decision_text=parsed.decision,
            note_text=parsed.note,
        ))

    def _phase_log_self_output(self, parsed) -> None:
        decision_text = parsed.decision.strip() or "(none)"
        print(cyan(f"[HB #{self.heartbeat_count}] decision: {decision_text}"),
              flush=True)
        if parsed.thinking:
            print(cyan(f"[HB #{self.heartbeat_count}] thinking: "
                        f"{parsed.thinking.strip()}"), flush=True)
        if parsed.note:
            print(cyan(f"[HB #{self.heartbeat_count}] note: "
                        f"{parsed.note.strip()}"), flush=True)

    async def _phase_auto_ingest_feedback(self, stimuli) -> None:
        for s in stimuli:
            if s.type != "tentacle_feedback":
                continue
            try:
                await self.gm.auto_ingest(
                    s.content, source_heartbeat=self.heartbeat_count,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[runtime] auto_ingest error: {e}", flush=True)

    async def _phase_apply_hypothalamus(self, parsed, recall_result) -> None:
        decision = parsed.decision.strip().lower()
        if not decision or decision in ("no action", "无行动"):
            return
        try:
            result = await self.hypothalamus.translate(
                parsed.decision, self.tentacles.list_descriptions(),
            )
        except Exception as e:  # noqa: BLE001
            print(f"[HB #{self.heartbeat_count}] Hypothalamus error: {e}",
                  flush=True)
            return
        self._log_hypo_summary(result)
        await self._dispatch_tentacle_calls(result.tentacle_calls)
        await self._apply_memory_writes(result.memory_writes, recall_result)
        await self._apply_memory_updates(result.memory_updates)

    def _phase_schedule_classify(self) -> None:
        """Background classify+link doesn't block the heartbeat."""
        self._classify_tasks.append(
            asyncio.create_task(self.gm.classify_and_link_pending()),
        )

    async def _phase_hibernate(self, parsed, recall_result) -> None:
        if recall_result.uncovered_stimuli:
            interval = self.config.hibernate.min_interval
        else:
            interval = (parsed.hibernate_seconds
                        or self.config.hibernate.default_interval)
        print(f"[HB #{self.heartbeat_count}] hibernate {interval}s",
              flush=True)
        self._recall = self._new_recall()
        await hibernate_with_recall(
            interval, self.buffer, self._recall,
            min_interval=self._min, max_interval=self._max,
        )

    # ---------- hypothalamus side-effects ----------

    def _log_hypo_summary(self, result) -> None:
        print(yellow(
            f"[hypo] tentacle_calls={len(result.tentacle_calls)} "
            f"memory_writes={len(result.memory_writes)} "
            f"memory_updates={len(result.memory_updates)} "
            f"sleep={result.sleep}"
        ), flush=True)
        if result.sleep:
            print(yellow("[hypo] sleep requested "
                          "(7-phase sleep lands in Phase 2)"),
                  file=sys.stderr, flush=True)

    async def _dispatch_tentacle_calls(self, calls) -> None:
        call_ids: list[str] = []
        for idx, call in enumerate(calls):
            cid = f"hb{self.heartbeat_count}_c{idx}"
            call_ids.append(cid)
            asyncio.create_task(self._dispatch(call, cid))
        if call_ids:
            self.batch_tracker.register_batch(call_ids)

    async def _apply_memory_writes(self, writes, recall_result) -> None:
        for w in writes:
            print(yellow(f"[hypo] memory_write: "
                           f"{w.get('content', '')[:80]}"), flush=True)
            try:
                await self.gm.explicit_write(
                    w["content"],
                    importance=w.get("importance", "normal"),
                    recall_context=recall_result.nodes,
                    source_heartbeat=self.heartbeat_count,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[runtime] explicit_write error: {e}", flush=True)

    async def _apply_memory_updates(self, updates) -> None:
        for u in updates:
            print(yellow(f"[hypo] memory_update: "
                           f"{u.get('node_name')} → "
                           f"{u.get('new_category')}"), flush=True)
            try:
                await self.gm.update_node_category(
                    u["node_name"], u["new_category"],
                )
            except Exception as e:  # noqa: BLE001
                print(f"[runtime] update_category error: {e}", flush=True)

    async def _dispatch(self, call: TentacleCall, call_id: str) -> None:
        try:
            tentacle = self.tentacles.get(call.tentacle)
        except KeyError:
            await self.buffer.push(Stimulus(
                type="system_event", source="runtime",
                content=f"Unknown tentacle: {call.tentacle}",
                timestamp=datetime.now(), adrenalin=False,
            ))
            print(f"[dispatch] unknown tentacle: {call.tentacle}", flush=True)
            await self.batch_tracker.mark_completed(call_id)
            return

        print(yellow(f"[dispatch] {call.tentacle} ← {call.intent!r}"
                       f"{' (adrenalin)' if call.adrenalin else ''}"),
              flush=True)
        try:
            stim = await tentacle.execute(call.intent, call.params)
        except Exception as e:  # noqa: BLE001
            print(f"[dispatch] {call.tentacle} error: {e}", flush=True)
            await self.buffer.push(Stimulus(
                type="system_event", source=f"tentacle:{call.tentacle}",
                content=f"error: {e}", timestamp=datetime.now(),
                adrenalin=call.adrenalin,
            ))
            await self.batch_tracker.mark_completed(call_id)
            return

        if call.adrenalin and not stim.adrenalin:
            stim.adrenalin = True
        # Bot's actual outward reply (chat) — green like a chat app message.
        print(green(f"[{call.tentacle}] {stim.content}"), flush=True)
        await self.buffer.push(stim)
        await self.batch_tracker.mark_completed(call_id)

    def _status(self, node_count: int, edge_count: int,
                  pct: int, hint: str) -> dict[str, Any]:
        return {
            "gm_node_count": node_count,
            "gm_edge_count": edge_count,
            "fatigue_pct": pct,
            "fatigue_hint": hint,
            "last_sleep_time": "never",
            "heartbeats_since_sleep": self.heartbeat_count,
            "tentacles": self.tentacles.list_descriptions(),
        }


def _delta_str(delta: int) -> str:
    if delta > 0:
        return f" (+{delta})"
    if delta < 0:
        return f" ({delta})"
    return ""


def _summarize_stimuli(stimuli: list[Stimulus]) -> str:
    if not stimuli:
        return "(none)"
    return " | ".join(f"{s.source}: {s.content[:60]}" for s in stimuli)


# ---------------- production builder ----------------


def build_runtime_from_config(config_path: str = "config.yaml") -> Runtime:
    cfg = load_config(config_path)
    self_role = cfg.llm.roles["self"]
    hypo_role = cfg.llm.roles["hypothalamus"]
    tentacle_role = cfg.llm.roles["tentacle_default"]
    compact_role = cfg.llm.roles.get("compact", self_role)
    embedding_role = cfg.llm.roles["embedding"]

    self_llm = LLMClient(cfg.llm.providers[self_role.provider],
                           self_role.model)
    hypo_llm = LLMClient(cfg.llm.providers[hypo_role.provider],
                           hypo_role.model)
    action_llm = LLMClient(cfg.llm.providers[tentacle_role.provider],
                            tentacle_role.model)
    compact_llm = LLMClient(cfg.llm.providers[compact_role.provider],
                              compact_role.model)
    classify_llm = compact_llm  # reuse
    embed_client = LLMClient(cfg.llm.providers[embedding_role.provider],
                               embedding_role.model)

    async def embedder(text: str) -> list[float]:
        return await embed_client.embed(text)

    reranker = None
    rerank_role = cfg.llm.roles.get("reranker")
    if rerank_role is not None:
        reranker_client = LLMClient(
            cfg.llm.providers[rerank_role.provider], rerank_role.model,
        )

        class _RerankerAdapter:
            async def rerank(self, query, docs):
                return await reranker_client.rerank(query, docs)

        reranker = _RerankerAdapter()

    deps = RuntimeDeps(
        config=cfg, self_llm=self_llm, hypo_llm=hypo_llm,
        action_llm=action_llm, compact_llm=compact_llm,
        classify_llm=classify_llm, embedder=embedder, reranker=reranker,
    )
    return Runtime(deps)


async def _amain() -> None:
    runtime = build_runtime_from_config()
    try:
        await runtime.run()
    finally:
        await runtime.close()


if __name__ == "__main__":
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass
