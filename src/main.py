"""CogniBot entrypoint + main heartbeat loop (DevSpec §6.4).

Phase 1 wiring: GraphMemory + IncrementalRecall + SlidingWindow + compact +
fatigue calc + BatchTracker + async classify. Phase 2 will add Bootstrap,
KnowledgeBase, and full Sleep.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from src.bootstrap import (
    BOOTSTRAP_PROMPT, detect_bootstrap_complete, load_genesis,
    load_self_model_or_default, parse_self_model_update,
)
from src.dashboard.events_ws import EventBroadcaster
from src.dashboard.server import DashboardServer, create_app as create_dashboard_app
from src.dashboard.web_chat import WebChatHistory
from src.hypothalamus import Hypothalamus, TentacleCall
from src.models.self_model import SelfModelStore
from src.interfaces.sensory import SensoryRegistry
from src.interfaces.tentacle import TentacleRegistry
from src.llm.client import LLMClient
from src.memory.graph_memory import GraphMemory
from src.memory.knowledge_base import KBRegistry
from src.memory.recall import IncrementalRecall, Reranker
from src.models.config import Config, load_config
from src.models.config_backup import backup_config
from src.models.stimulus import Stimulus
from src.prompt.builder import PromptBuilder, SlidingWindowRound
from src.runtime.batch_tracker import BatchTrackerSensory
from src.runtime.event_bus import (
    DecisionEvent, DispatchEvent, EventBus, GMStatsEvent, HeartbeatStartEvent,
    HibernateEvent, HypothalamusEvent, NoteEvent, PromptBuiltEvent,
    SleepDoneEvent, SleepStartEvent, StimuliQueuedEvent, TentacleResultEvent,
    ThinkingEvent,
)
from src.runtime.heartbeat_logger import HeartbeatLogger
from src.runtime.override_commands import (
    OverrideAction, handle_override, parse_override,
)
from src.sleep.sleep_manager import enter_sleep_mode
from src.runtime.compact import compact_if_needed
from src.runtime.fatigue import calculate_fatigue
from src.runtime.hibernate import hibernate_with_recall
from src.runtime.sliding_window import SlidingWindow
from src.runtime.stimulus_buffer import StimulusBuffer
from src.self_agent import parse_self_output
from src.sensories.telegram import HttpTelegramClient, TelegramSensory
from src.tentacles.coding import CodingTentacle, SubprocessRunner
from src.tentacles.gui_control import GuiControlTentacle, PyAutoGUIBackend
from src.tentacles.memory_recall import MemoryRecallTentacle
from src.tentacles.search import DDGSBackend, SearchTentacle
from src.tentacles.telegram_reply import TelegramReplyTentacle
from src.tentacles.web_chat_reply import WebChatTentacle


class ChatLike(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...


class AsyncEmbedder(Protocol):
    async def __call__(self, text: str) -> list[float]: ...


@dataclass
class RuntimeDeps:
    config: Config
    self_llm: ChatLike
    hypo_llm: ChatLike
    compact_llm: ChatLike
    classify_llm: ChatLike
    embedder: AsyncEmbedder
    reranker: Reranker | None = None
    self_model_path: str | None = None      # default: workspace/self_model.yaml
    genesis_path: str | None = None         # default: workspace/GENESIS.md
    config_path: str | None = None          # default: config.yaml — for dashboard
    backup_dir: str | None = None           # default: workspace/backups


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


class Runtime:
    def __init__(self, deps: RuntimeDeps, *, hibernate_min: float | None = None,
                 hibernate_max: float | None = None,
                 logger: HeartbeatLogger | None = None,
                 is_bootstrap_override: bool | None = None,
                 event_bus: EventBus | None = None):
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

        self.kb_registry = KBRegistry(
            self.gm,
            kb_dir=self.config.knowledge_base.dir or "workspace/data/knowledge_bases",
            embedder=deps.embedder,
        )

        self.tentacles = TentacleRegistry()
        self.tentacles.register(MemoryRecallTentacle(
            gm=self.gm, embedder=self.embedder,
            kb_registry=self.kb_registry,
        ))
        if self.config.tentacle.get("search", {}).get("enabled", True):
            self.tentacles.register(SearchTentacle(
                backend=DDGSBackend(),
                max_results=self.config.tentacle.get("search", {})
                    .get("max_results", 5),
            ))
        coding_cfg = self.config.tentacle.get("coding", {})
        if coding_cfg.get("enabled", False):
            self.tentacles.register(CodingTentacle(
                runner=SubprocessRunner(),
                sandbox_dir=coding_cfg.get("sandbox_dir",
                                              "workspace/sandbox"),
                timeout_seconds=coding_cfg.get("timeout_seconds", 30),
                max_output_chars=coding_cfg.get("max_output_chars", 4000),
            ))
        gui_cfg = self.config.tentacle.get("gui_control", {})
        if gui_cfg.get("enabled", False):
            self.tentacles.register(GuiControlTentacle(
                backend=PyAutoGUIBackend(),
                screenshot_dir=gui_cfg.get("screenshot_dir",
                                              "workspace/screenshots"),
            ))

        # Web chat (always wired — Self can write to history even if no
        # browser is connected; messages persist for next viewer)
        web_chat_cfg = self.config.tentacle.get("web_chat", {})
        chat_path = web_chat_cfg.get("history_path",
                                        "workspace/data/web_chat.jsonl")
        self.web_chat_history = WebChatHistory(chat_path)
        self.tentacles.register(WebChatTentacle(history=self.web_chat_history))

        self.sensories = SensoryRegistry()
        tg_cfg = self.config.sensory.get("telegram", {})
        if tg_cfg.get("enabled", False):
            tg_token = tg_cfg.get("bot_token") or ""
            allowed = tg_cfg.get("allowed_chat_ids") or None
            tg_client = HttpTelegramClient(token=tg_token)
            self.sensories.register(TelegramSensory(
                client=tg_client,
                allowed_chat_ids=set(allowed) if allowed else None,
            ))
            default_chat = tg_cfg.get("default_chat_id")
            self.tentacles.register(TelegramReplyTentacle(
                client=tg_client, default_chat_id=default_chat,
            ))
        self.batch_tracker = BatchTrackerSensory()
        self.sensories.register(self.batch_tracker)

        # Self-model + Bootstrap state (Phase 2.1)
        sm_path = deps.self_model_path or "workspace/self_model.yaml"
        gen_path = deps.genesis_path or "workspace/GENESIS.md"
        self._self_model_store = SelfModelStore(sm_path)
        self._genesis_text = load_genesis(gen_path)
        self.self_model, detected_bootstrap = load_self_model_or_default(sm_path)
        self.is_bootstrap = (is_bootstrap_override
                              if is_bootstrap_override is not None
                              else detected_bootstrap)

        self.heartbeat_count = 0
        self._stop = False
        self._min = hibernate_min if hibernate_min is not None else self.config.hibernate.min_interval
        self._max = hibernate_max if hibernate_max is not None else self.config.hibernate.max_interval

        self._recall: IncrementalRecall | None = None
        self._classify_tasks: list[asyncio.Task] = []
        self._last_node_count = 0
        self._last_edge_count = 0
        self.log = logger or HeartbeatLogger()
        self.sleep_log_dir = "workspace/logs"
        self.events = event_bus or EventBus()
        self._dashboard: DashboardServer | None = None
        self._config_path = deps.config_path  # for dashboard settings page
        self._backup_dir = deps.backup_dir or "workspace/backups"

        # Snapshot config.yaml on every startup so a bad save can be rolled
        # back from workspace/backups/.
        if self._config_path:
            try:
                backup_config(self._config_path, self._backup_dir)
            except Exception as e:  # noqa: BLE001
                self.log.runtime_error(f"config backup failed: {e}")

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
        await self._maybe_start_dashboard()
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
        """Shut down persistent resources (Dashboard + GM + open KBs)."""
        if self._dashboard is not None:
            try:
                await self._dashboard.stop()
            except Exception as e:  # noqa: BLE001
                self.log.runtime_error(f"dashboard stop error: {e}")
            self._dashboard = None
        await self.kb_registry.close_all()
        await self.gm.close()

    async def _maybe_start_dashboard(self) -> None:
        cfg = getattr(self.config, "dashboard", None)
        if cfg is None or not cfg.enabled:
            return

        async def _on_user_message(text: str,
                                     attachments: list[dict] | None = None) -> None:
            # Web user message → push as a normal user_message stimulus.
            # Attachments rendered as text notice so Self knows about them
            # (vision/binary handling lives behind that).
            content = text
            if attachments:
                lines = [text] if text else []
                for a in attachments:
                    name = a.get("name", "file")
                    typ = a.get("type", "")
                    size = a.get("size", 0)
                    url = a.get("url", "")
                    lines.append(f"[附件: {name} ({typ}, {size} bytes) {url}]")
                content = "\n".join(lines)
            md: dict = {"channel": "web_chat"}
            if attachments:
                md["attachments"] = attachments
            await self.buffer.push(Stimulus(
                type="user_message", source="sensory:web_chat",
                content=content, timestamp=datetime.now(),
                adrenalin=True,
                metadata=md,
            ))

        def _on_restart() -> None:
            """Re-exec process so a new instance picks up edited config."""
            import os as _os
            import sys as _sys
            self.log.hb("restart requested via dashboard — re-execing")
            _os.execv(_sys.executable, [_sys.executable, *_sys.argv])

        try:
            broadcaster = EventBroadcaster(self.events)
            self._dashboard = DashboardServer(
                create_dashboard_app(
                    runtime=self,
                    web_chat_history=self.web_chat_history,
                    on_user_message=_on_user_message,
                    event_broadcaster=broadcaster,
                    config_path=Path(self._config_path) if self._config_path else None,
                    on_restart=_on_restart,
                ),
                host=cfg.host, port=cfg.port,
            )
            await self._dashboard.start()
            self.log.hb(
                f"dashboard listening on http://{cfg.host}:{self._dashboard.port}"
            )
        except OSError as e:
            self.log.runtime_error(
                f"dashboard failed to start (port {cfg.port} in use? {e}); "
                "runtime continues without dashboard"
            )
            self._dashboard = None
        except Exception as e:  # noqa: BLE001
            self.log.runtime_error(f"dashboard startup error: {e}")
            self._dashboard = None

    async def _heartbeat(self) -> None:
        """Orchestration only. Each phase is its own method.

        Sleep can short-circuit the heartbeat at two points:
          - immediately after fatigue compute (force-sleep at threshold)
          - after Hypothalamus translation if Self requested 'enter sleep'
        Override commands (/kill, /sleep) can also short-circuit.
        """
        self.heartbeat_count += 1
        self.log.set_heartbeat(self.heartbeat_count)
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

        if counts.fatigue_pct >= self.config.fatigue.force_sleep_threshold:
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
        if self.is_bootstrap:
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

    # ---------- heartbeat phases ----------

    async def _phase_handle_overrides(
        self, stimuli: list[Stimulus],
    ) -> tuple[list[Stimulus], "OverrideAction | None"]:
        """Scan drained stimuli for /<cmd>. Each match is consumed (Self
        never sees it) and executed out-of-band. Returns the filtered
        stimulus list + the highest-priority action triggered (KILL > SLEEP)."""
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
            result = await handle_override(cmd, self)
            self.log.hb(f"override /{cmd}: {result.output}")
            if result.action is OverrideAction.KILL:
                triggered = OverrideAction.KILL
                self._stop = True
                break
            if (result.action is OverrideAction.SLEEP
                    and triggered is not OverrideAction.KILL):
                triggered = OverrideAction.SLEEP
            # Self can still see informational overrides as system events
            if result.action is OverrideAction.NONE:
                await self.buffer.push(Stimulus(
                    type="system_event", source="system:override",
                    content=f"/{cmd}: {result.output}",
                    timestamp=datetime.now(), adrenalin=False,
                ))
        return filtered, triggered

    async def _phase_drain_and_seed_recall(self) -> list[Stimulus]:
        stimuli = self.buffer.drain()
        self.log.hb(f"stimuli={len(stimuli)} (thinking...)")
        self.events.publish(HeartbeatStartEvent(
            heartbeat_id=self.heartbeat_count, stimulus_count=len(stimuli),
        ))
        self.events.publish(StimuliQueuedEvent(stimuli=[
            {"type": s.type, "source": s.source, "content": s.content,
             "adrenalin": s.adrenalin, "ts": s.timestamp.isoformat()}
            for s in stimuli
        ]))
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
        self.log.hb(
            f"gm: nodes={node_count}{_delta_str(node_delta)}, "
            f"edges={edge_count}{_delta_str(edge_delta)}, fatigue={pct}%"
        )
        if pct >= self.config.fatigue.force_sleep_threshold:
            self.log.hb_warn(f"force-sleep threshold reached (fatigue={pct}%);"
                              " Sleep mode lands in Phase 2.")
        self.events.publish(GMStatsEvent(
            heartbeat_id=self.heartbeat_count,
            node_count=node_count, edge_count=edge_count, fatigue_pct=pct,
        ))
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
        if self.is_bootstrap:
            prompt = (BOOTSTRAP_PROMPT.format(genesis_text=self._genesis_text)
                      + "\n\n" + prompt)
        self.events.publish(PromptBuiltEvent(
            heartbeat_id=self.heartbeat_count,
            layers={"full_prompt": prompt},
        ))
        try:
            raw = await self.self_llm.chat(
                [{"role": "user", "content": prompt}]
            )
        except Exception as e:  # noqa: BLE001
            self.log.hb(f"Self LLM error: {e}")
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

    def _phase_apply_bootstrap_signals(self, parsed) -> None:
        """During Bootstrap, parse Self's NOTE for self-model JSON updates
        and the 'bootstrap complete' marker."""
        note = parsed.note or ""
        update = parse_self_model_update(note)
        if update:
            self.self_model = self._self_model_store.update(update)
            self.log.hb(f"bootstrap: self-model updated "
                          f"({list(update.keys())})")
        if detect_bootstrap_complete(note):
            self.self_model = self._self_model_store.update(
                {"state": {"bootstrap_complete": True}}
            )
            self.is_bootstrap = False
            self.log.hb("bootstrap: complete — entering normal operation")

    def _phase_log_self_output(self, parsed) -> None:
        decision_text = parsed.decision.strip() or "(none)"
        self.log.hb_thought("decision", decision_text)
        self.events.publish(DecisionEvent(
            heartbeat_id=self.heartbeat_count, text=decision_text,
        ))
        if parsed.thinking:
            self.log.hb_thought("thinking", parsed.thinking)
            self.events.publish(ThinkingEvent(
                heartbeat_id=self.heartbeat_count,
                text=parsed.thinking.strip(),
            ))
        if parsed.note:
            self.log.hb_thought("note", parsed.note)
            self.events.publish(NoteEvent(
                heartbeat_id=self.heartbeat_count, text=parsed.note.strip(),
            ))

    async def _phase_auto_ingest_feedback(self, stimuli) -> None:
        for s in stimuli:
            if s.type != "tentacle_feedback":
                continue
            try:
                await self.gm.auto_ingest(
                    s.content, source_heartbeat=self.heartbeat_count,
                )
            except Exception as e:  # noqa: BLE001
                self.log.runtime_error(f"auto_ingest error: {e}")

    async def _phase_apply_hypothalamus(self, parsed, recall_result) -> bool:
        """Returns True iff Self requested sleep (caller should short-circuit
        and run _perform_sleep)."""
        decision = parsed.decision.strip().lower()
        if not decision or decision in ("no action", "无行动"):
            return False
        try:
            result = await self.hypothalamus.translate(
                parsed.decision, self.tentacles.list_descriptions(),
            )
        except Exception as e:  # noqa: BLE001
            self.log.hb(f"Hypothalamus error: {e}")
            return False
        self._log_hypo_summary(result)
        await self._dispatch_tentacle_calls(result.tentacle_calls)
        await self._apply_memory_writes(result.memory_writes, recall_result)
        await self._apply_memory_updates(result.memory_updates)
        return bool(result.sleep)

    def _phase_schedule_classify(self) -> None:
        """Background classify+link doesn't block the heartbeat."""
        self._classify_tasks.append(
            asyncio.create_task(self.gm.classify_and_link_pending()),
        )

    async def _phase_hibernate(self, parsed, recall_result) -> None:
        if self.is_bootstrap:
            # DevSpec §12.2 — bootstrap heartbeat is fixed 10s
            interval = 10
        elif recall_result.uncovered_stimuli:
            interval = self.config.hibernate.min_interval
        else:
            interval = (parsed.hibernate_seconds
                        or self.config.hibernate.default_interval)
        self.log.hb(f"hibernate {interval}s")
        self.events.publish(HibernateEvent(
            heartbeat_id=self.heartbeat_count, interval_seconds=interval,
        ))
        self._recall = self._new_recall()
        await hibernate_with_recall(
            interval, self.buffer, self._recall,
            min_interval=self._min, max_interval=self._max,
        )

    # ---------- sleep ----------

    async def _perform_sleep(self, reason: str, *, wake_msg: str) -> None:
        """Run 7-phase Sleep, persist self-model bookkeeping, push wake-up
        stimulus, reset incremental recall (GM state changed)."""
        self.log.hb(f"sleep started — {reason}")
        self.events.publish(SleepStartEvent(reason=reason))
        try:
            sl = self.config.sleep
            stats = await enter_sleep_mode(
                self.gm, self.kb_registry, self.sensories,
                llm=self.compact_llm, embedder=self.embedder,
                log_dir=self.sleep_log_dir,
                min_community_size=sl.min_community_size,
                kb_consolidation_threshold=sl.kb_consolidation_threshold,
                kb_index_max=sl.kb_index_max,
                kb_archive_pct=sl.kb_archive_pct,
                kb_revive_threshold=sl.kb_revive_threshold,
            )
        except Exception as e:  # noqa: BLE001
            self.log.hb(f"sleep failed: {e}")
            return
        self.events.publish(SleepDoneEvent(stats=stats))
        self.log.hb(
            f"sleep done: facts_migrated={stats['facts_migrated']}, "
            f"focus_cleared={stats['focus_cleared']}, "
            f"kbs={stats['kbs_created']}, index_nodes={stats['index_nodes']}"
        )

        # Self-model bookkeeping
        cycles = (self.self_model.get("statistics", {})
                   .get("total_sleep_cycles", 0)) + 1
        self.self_model = self._self_model_store.update({
            "state": {"is_sleeping": False},
            "statistics": {
                "total_sleep_cycles": cycles,
                "last_sleep": datetime.now().isoformat(),
            },
        })

        # Wake-up stimulus
        await self.buffer.push(Stimulus(
            type="system_event", source="system:sleep",
            content=wake_msg, timestamp=datetime.now(),
            adrenalin=False,
        ))

        # GM changed underneath us — start a fresh recall
        self._recall = self._new_recall()

    # ---------- hypothalamus side-effects ----------

    def _log_hypo_summary(self, result) -> None:
        self.log.hypo(
            f"tentacle_calls={len(result.tentacle_calls)} "
            f"memory_writes={len(result.memory_writes)} "
            f"memory_updates={len(result.memory_updates)} "
            f"sleep={result.sleep}"
        )
        self.events.publish(HypothalamusEvent(
            heartbeat_id=self.heartbeat_count,
            tentacle_calls_count=len(result.tentacle_calls),
            memory_writes_count=len(result.memory_writes),
            memory_updates_count=len(result.memory_updates),
            sleep_requested=result.sleep,
        ))

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
            self.log.hypo(f"memory_write: {w.get('content', '')[:80]}")
            try:
                await self.gm.explicit_write(
                    w["content"],
                    importance=w.get("importance", "normal"),
                    recall_context=recall_result.nodes,
                    source_heartbeat=self.heartbeat_count,
                )
            except Exception as e:  # noqa: BLE001
                self.log.runtime_error(f"explicit_write error: {e}")

    async def _apply_memory_updates(self, updates) -> None:
        for u in updates:
            self.log.hypo(f"memory_update: {u.get('node_name')} → "
                           f"{u.get('new_category')}")
            try:
                await self.gm.update_node_category(
                    u["node_name"], u["new_category"],
                )
            except Exception as e:  # noqa: BLE001
                self.log.runtime_error(f"update_category error: {e}")

    async def _dispatch(self, call: TentacleCall, call_id: str) -> None:
        try:
            tentacle = self.tentacles.get(call.tentacle)
        except KeyError:
            await self.buffer.push(Stimulus(
                type="system_event", source="runtime",
                content=f"Unknown tentacle: {call.tentacle}",
                timestamp=datetime.now(), adrenalin=False,
            ))
            self.log.dispatch(f"unknown tentacle: {call.tentacle}")
            await self.batch_tracker.mark_completed(call_id)
            return

        self.log.dispatch(
            f"{call.tentacle} ← {call.intent!r}"
            f"{' (adrenalin)' if call.adrenalin else ''}"
        )
        self.events.publish(DispatchEvent(
            heartbeat_id=self.heartbeat_count, tentacle=call.tentacle,
            intent=call.intent, adrenalin=call.adrenalin,
        ))
        try:
            stim = await tentacle.execute(call.intent, call.params)
        except Exception as e:  # noqa: BLE001
            self.log.dispatch(f"{call.tentacle} error: {e}")
            await self.buffer.push(Stimulus(
                type="system_event", source=f"tentacle:{call.tentacle}",
                content=f"error: {e}", timestamp=datetime.now(),
                adrenalin=call.adrenalin,
            ))
            await self.batch_tracker.mark_completed(call_id)
            return

        if call.adrenalin and not stim.adrenalin:
            stim.adrenalin = True
        if tentacle.is_internal:
            self.log.internal(call.tentacle, stim.content)
        else:
            self.log.chat(call.tentacle, stim.content)
        self.events.publish(TentacleResultEvent(
            tentacle=call.tentacle, content=stim.content,
            is_internal=tentacle.is_internal,
        ))
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
    compact_role = cfg.llm.roles.get("compact", self_role)
    embedding_role = cfg.llm.roles["embedding"]

    self_llm = LLMClient(cfg.llm.providers[self_role.provider],
                           self_role.model)
    hypo_llm = LLMClient(cfg.llm.providers[hypo_role.provider],
                           hypo_role.model)
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
        compact_llm=compact_llm,
        classify_llm=classify_llm, embedder=embedder, reranker=reranker,
        config_path=str(config_path),
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
