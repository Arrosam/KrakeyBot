"""KrakeyBot Runtime — composition root + lifecycle.

Wires up all collaborators (GraphMemory, KBRegistry, the three
plugin registries, BatchTracker, dispatcher, orchestrator, plugin
loader/observer, bootstrap coordinator) and drives the per-beat
loop via ``run()``. Process entry point + ``build_runtime_from_config``
live in ``src/main.py``; per-beat algorithm lives in
``src/runtime/heartbeat/heartbeat_orchestrator.py``.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.bootstrap import load_self_model_or_default
from src.models.self_model import SelfModelStore
from src.interfaces.tentacle import TentacleRegistry
from src.llm.resolve import AsyncEmbedder, ChatLike, resolve_llm_for_tag
from src.memory.graph_memory import GraphMemory
from src.memory.knowledge_base import KBRegistry
from src.memory.recall import RecallLike, Reranker
from src.models.config import Config, LLMParams
from src.models.config_backup import backup_config
from src.prompt.builder import PromptBuilder
from src.runtime.stimuli.batch_tracker import BatchTrackerSensory
from src.runtime.events.event_bus import EventBus
from src.runtime.console.heartbeat_logger import HeartbeatLogger
from src.runtime.heartbeat.sliding_window import SlidingWindow
from src.runtime.stimuli.stimulus_buffer import StimulusBuffer
from src.sandbox.policy import build_code_runner, preflight_if_required


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
    # Root directory holding per-plugin folders. Each plugin's user
    # config lives at ``<root>/<name>/config.yaml`` co-located with
    # whatever else the plugin needs. Default ``workspace/plugins``.
    # Tests pass a tmpdir so per-plugin config writes (history_path,
    # llm_purposes, …) don't bleed into the production workspace.
    plugin_configs_root: str | None = None  # default: workspace/plugins
    # in_mind Reflect's state file. None → workspace/data/in_mind.json
    # (the locked-in production path, see design doc Reflect #3).
    # Tests pass a tmpdir so update_in_mind dispatches don't bleed
    # into the production state.
    in_mind_state_path: str | None = None
    # Shared LLMClient cache keyed by tag name. Populated by
    # build_runtime_from_config for core purposes; Runtime adds plugin
    # purpose entries on top. Sharing the cache means two purposes that
    # map to the same tag share one client (saves connections + keeps
    # rate-limit accounting consistent).
    llm_clients_by_tag: dict[str, Any] = field(default_factory=dict)


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
        # Reflect registry — role-keyed (one Reflect per role; second
        # registration claiming the same role raises). Plugin loader
        # runs LATER (after tentacles + sensories registries exist)
        # since a single plugin can contribute components of all three
        # kinds; loading them before all three registries are built
        # would crash.
        from src.interfaces.reflect import ReflectRegistry
        self.reflects = ReflectRegistry()
        self.buffer = StimulusBuffer()
        # History token budget is derived from the Self role's input
        # context window × history_token_fraction. The Self role's
        # params.max_input_tokens is already resolved by the config
        # loader (YAML > model lookup > 128k default) so `int()` is
        # safe. Only Self consumes this window, so we key it off
        # Self's role config.
        self_params = self.config.llm.core_params("self_thinking") or LLMParams()
        _history_budget = int(
            (self_params.max_input_tokens or 128_000)
            * self_params.history_token_fraction
        )
        self.window = SlidingWindow(history_token_budget=_history_budget)
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
        # Each plugin's config lives at
        # <plugin_configs_root>/<name>/config.yaml. Same root as the
        # plugin folders themselves (workspace/plugins/) so user config
        # is co-located with plugin code, exactly like third-party
        # workspace plugins. Runtime never WRITES to these files —
        # the dashboard owns its own FilePluginConfigStore for that.
        plugin_root = Path(deps.plugin_configs_root
                            or "workspace/plugins")
        self._plugin_configs_root = plugin_root

        # ``self.buffer`` is both the live stimulus queue AND the live
        # sensory set (see src/runtime/stimuli/stimulus_buffer.py).
        # BatchTracker is core runtime infrastructure (dispatch wake-up
        # mechanism) — kept out of the plugin system so it can't be
        # disabled by accident.
        self.batch_tracker = BatchTrackerSensory()
        self.buffer.register(self.batch_tracker)

        # Plugin registration. Each plugin can contribute reflect /
        # tentacle / sensory components in any combination via its
        # meta.yaml. Has to run AFTER the three registries exist
        # (above) because one plugin may register into any of them.
        #
        # Bring up log + events + config paths BEFORE plugin registration
        # so the dashboard plugin (which pulls a runtime ref via
        # ctx.services["runtime"]) sees a runtime with the fields it
        # needs at sensory.start() time.
        self.log = logger or HeartbeatLogger()
        self.sleep_log_dir = "workspace/logs"
        self.events = event_bus or EventBus()
        self._config_path = deps.config_path  # for dashboard settings page
        self._backup_dir = deps.backup_dir or "workspace/backups"
        # Per-run ring buffer of assembled heartbeat prompts for the
        # dashboard Prompts tab. Hardcoded size — was tied to
        # config.dashboard.prompt_log_size which is gone now that the
        # dashboard owns its own per-plugin config.
        self._prompt_log: deque[dict[str, Any]] = deque(maxlen=50)

        from src.runtime.plugin_register.loader import PluginLoader
        self._plugin_loader = PluginLoader(
            config=self.config,
            reflects=self.reflects,
            tentacles=self.tentacles,
            sensories=self.buffer,
            services={
                "runtime": self,
                "gm": self.gm,
                "kb_registry": self.kb_registry,
                "embedder": self.embedder,
                "reranker": self.reranker,
                "buffer": self.buffer,
                "events": self.events,
                "config": self.config,
                "build_code_runner": self._build_code_runner,
            },
        )
        self._plugin_loader.register_from_config(deps)

        # Self-model + Bootstrap state (Phase 2.1)
        sm_path = deps.self_model_path or "workspace/self_model.yaml"
        # GENESIS is read lazily (see `_get_genesis_text`) — in steady
        # state (bootstrap complete, GM populated) the file is never
        # touched, and `self._genesis_text` stays None to avoid both
        # the I/O on every start AND the correctness trap of having
        # stale genesis text sitting in memory when it's not supposed
        # to influence the prompt.
        self._genesis_path = deps.genesis_path or "workspace/GENESIS.md"
        self._genesis_text: str | None = None
        self._self_model_store = SelfModelStore(sm_path)
        self.self_model, detected_bootstrap = load_self_model_or_default(sm_path)
        # Bootstrap-mode state lives in a dedicated coordinator so the
        # three behaviors it gates (intro-prompt injection, hibernate
        # cadence, NOTE-signal parsing) don't have to be expressed as
        # scattered ``if self.is_bootstrap`` checks. The provisional
        # value here gets re-derived once gm.initialize() lets us
        # probe actual data via ``bootstrap.refine_from_data``.
        from src.runtime.bootstrap.bootstrap_coordinator import BootstrapCoordinator
        # When no test override is supplied, the coordinator initializes
        # from the self-model's bootstrap_complete marker (equivalent to
        # the legacy `detected_bootstrap` heuristic) and lets
        # refine_from_data re-derive from actual GM/KB data later. A
        # supplied override pins the value and skips the data probe.
        self.bootstrap = BootstrapCoordinator(
            self_model=self.self_model,
            self_model_store=self._self_model_store,
            override=is_bootstrap_override,
        )

        self.heartbeat_count = 0
        # Sleep cycle counter — runtime-only. Used to be persisted in
        # self_model.statistics.total_sleep_cycles, but stats was the
        # bulk of self_model's noise (most fields never written) so the
        # 2026-04-25 slim refactor pulled them all out. Per-process
        # counter is enough for /status and dashboard display; cross-run
        # totals weren't actually used by any product feature.
        self._sleep_cycles = 0
        self._stop = False
        self._min = hibernate_min if hibernate_min is not None else self.config.hibernate.min_interval
        self._max = hibernate_max if hibernate_max is not None else self.config.hibernate.max_interval

        self._recall: RecallLike | None = None
        self._classify_tasks: list[asyncio.Task] = []
        self._last_node_count = 0
        self._last_edge_count = 0

        # DecisionDispatcher — executes the 4 side-effects of a
        # DecisionResult (log+publish summary, dispatch tentacle calls,
        # apply memory writes, apply memory updates). Pure composition
        # over the same 5 collaborators (tentacles, batch_tracker,
        # buffer, gm, log+events). Built once; heartbeat passes its
        # current heartbeat_id on each call.
        from src.runtime.heartbeat.dispatcher import DecisionDispatcher
        self._dispatcher = DecisionDispatcher(
            tentacles=self.tentacles,
            batch_tracker=self.batch_tracker,
            buffer=self.buffer,
            gm=self.gm,
            log=self.log,
            events=self.events,
        )

        # Heartbeat algorithm — owns the per-beat orchestration but
        # holds no state. Reads + mutates Runtime fields through the
        # `runtime` ref. Built last (after all collaborators exist) so
        # nothing in its phase methods sees a half-constructed Runtime.
        from src.runtime.heartbeat.heartbeat_orchestrator import HeartbeatOrchestrator
        self._orchestrator = HeartbeatOrchestrator(self)

        # Snapshot config.yaml on every startup so a bad save can be rolled
        # back from workspace/backups/.
        if self._config_path:
            try:
                backup_config(self._config_path, self._backup_dir)
            except Exception as e:  # noqa: BLE001
                self.log.runtime_error(f"config backup failed: {e}")

        # Reflect attach() lifecycle hook — fires after every other
        # subsystem (TentacleRegistry, plugin loader, etc.) is up so
        # Reflects with extra runtime-coupled wiring beyond what the
        # meta.yaml components: list expresses can do it without
        # ordering surprises. No in-tree Reflect uses this today —
        # multi-component plugins ship sibling tentacles via meta.yaml
        # — but the hook stays available. attach_all is exception-
        # tolerant by contract; one bad Reflect won't block the others.
        self.reflects.attach_all(self)

        # Plugin observer is built AFTER attach_all so the snapshot it
        # walks includes anything an attach() hook added. Components
        # registered by the loader appear with source="builtin";
        # everything else (BatchTracker, attach() extras) appears as
        # source="core".
        from src.runtime.plugin_register.observer import PluginObserver
        self._plugin_observer = PluginObserver(
            reflects=self.reflects,
            tentacles=self.tentacles,
            sensories=self.buffer,
            loader=self._plugin_loader,
        )

    # Test-only facade — two reflect-config tests rebuild registries by
    # clearing them and re-running registration. New code should reach
    # for ``self._plugin_loader.register_from_config(deps)`` directly.
    def _register_plugins_from_config(self, deps: "RuntimeDeps") -> None:
        self._plugin_loader.register_from_config(deps)

    def _new_recall(self) -> RecallLike:
        # Facade — heartbeat algorithm lives in HeartbeatOrchestrator.
        return self._orchestrator.new_recall()

    @property
    def is_setup_mode(self) -> bool:
        """True when no Self LLM is bound (config incomplete).

        In this state Krakey still starts the dashboard so the user
        can finish configuration via Web UI, but the heartbeat loop
        is skipped. After the user fills in providers + tags + the
        ``self_thinking`` core purpose binding and clicks "Restart"
        in the dashboard, the next process boot will see a complete
        config and run the real heartbeat.
        """
        return self.self_llm is None

    @property
    def is_bootstrap(self) -> bool:
        """Bootstrap mode flag — delegates to the coordinator. Kept as
        a property so tests that read ``runtime.is_bootstrap`` keep
        working without knowing about the coordinator."""
        return self.bootstrap.is_active

    @is_bootstrap.setter
    def is_bootstrap(self, value: bool) -> None:
        """Test compat: ``runtime.is_bootstrap = True`` forwards to the
        coordinator's force_active escape hatch."""
        self.bootstrap.force_active(bool(value))

    async def run(self, iterations: int | None = None) -> None:
        await self.gm.initialize()
        await self._preflight_sandbox()
        await self._refine_bootstrap_from_data()
        # buffer.start_all() walks every registered sensory and calls its
        # start(); the dashboard plugin's sensory uses that hook to spin
        # up the Web UI server. No special-case "start dashboard" path —
        # dashboard is just a sensory like any other.
        await self.buffer.start_all()

        if self.is_setup_mode:
            await self._run_setup_mode()
            await self.buffer.stop_all()
            return

        self._recall = self._new_recall()
        try:
            count = 0
            while not self._stop:
                await self._heartbeat()
                count += 1
                if iterations is not None and count >= iterations:
                    return
        finally:
            await self.buffer.stop_all()
            # Cancel in-flight background classify tasks so asyncio doesn't warn.
            pending = [t for t in self._classify_tasks if not t.done()]
            for t in pending:
                t.cancel()

    async def _run_setup_mode(self) -> None:
        # Facade — banner + idle loop live in src.runtime.setup_mode.
        from src.runtime.setup_mode import run_setup_mode
        await run_setup_mode(self)

    @property
    def _plugin_infos(self) -> list:
        """PluginInfo list — exposed as a property so call sites that
        historically read ``self._plugin_infos`` (mostly tests) keep
        working. Walks the live registries each call."""
        return self._plugin_observer.collect_infos()

    def loaded_plugin_report(self) -> dict[str, Any]:
        """Pure runtime observation of which tentacles + sensories are
        live (no plugin-config file reads). The dashboard adapter
        combines this with its own ``FilePluginConfigStore`` reads to
        build the ``/api/plugins`` payload."""
        return self._plugin_observer.loaded_report()

    def _build_code_runner(self, coding_cfg: dict):
        # Facade — sandbox vs subprocess policy lives in
        # src.sandbox.policy. Kept as a method so the services dict
        # can hand a bound callable to the coding plugin.
        return build_code_runner(coding_cfg, self.config.sandbox)

    def _record_prompt(self, heartbeat_id: int, prompt: str) -> None:
        # Facade — heartbeat algorithm lives in HeartbeatOrchestrator.
        self._orchestrator.record_prompt(heartbeat_id, prompt)

    def recent_prompts(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Newest-first list of recorded prompts. Used by the dashboard
        /api/prompts endpoint. Returns shallow copies."""
        items = list(self._prompt_log)
        items.reverse()
        if limit is not None:
            items = items[:limit]
        return [dict(p) for p in items]

    async def _preflight_sandbox(self) -> None:
        # Facade — preflight scan logic lives in src.sandbox.policy.
        # Wrapper keeps the success log line attached to the heartbeat
        # log instead of stderr from a free function.
        info = await preflight_if_required(
            self.config, plugin_configs_root=self._plugin_configs_root,
        )
        if info is not None:
            self.log.hb(
                f"sandbox preflight ok: guest_os={info.get('guest_os')} "
                f"agent_version={info.get('agent_version')}"
            )

    async def _refine_bootstrap_from_data(self) -> None:
        """Thin wrapper around ``BootstrapCoordinator.refine_from_data``
        — kept on Runtime so the call site in ``run()`` stays a single
        line and the log message stays attached to the heartbeat log."""
        await self.bootstrap.refine_from_data(self.gm, self.kb_registry)
        if self.bootstrap.is_active:
            self.log.hb("bootstrap mode: empty GM + KBs → injecting GENESIS")

    async def close(self) -> None:
        """Shut down persistent resources (GM + open KBs). Sensories
        — including the dashboard plugin's server — are stopped via
        ``buffer.stop_all()`` in ``run()``'s finally block."""
        await self.kb_registry.close_all()
        await self.gm.close()

    # ---------- heartbeat algorithm ----------

    async def _heartbeat(self) -> None:
        # Facade — algorithm lives in HeartbeatOrchestrator.
        await self._orchestrator.beat()

    # The next four facades exist because tests reach in directly to
    # exercise prompt assembly + budget enforcement + GENESIS lazy-load
    # without running a whole heartbeat. Production callers go through
    # the orchestrator.

    def _build_self_prompt(self, stimuli, recall_result, counts):
        return self._orchestrator.build_self_prompt(
            stimuli, recall_result, counts,
        )

    async def _enforce_input_budget(self, stimuli, recall_result, counts):
        return await self._orchestrator.enforce_input_budget(
            stimuli, recall_result, counts,
        )

    def _get_genesis_text(self) -> str:
        return self._orchestrator.get_genesis_text()




