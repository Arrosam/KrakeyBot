"""KrakeyBot Runtime — composition root + lifecycle.

Holds the resolved Engine bundle + the three plugin registries
(Tool / Channel / Modifier) + lifecycle plumbing (event bus,
stimulus buffer, plugin loader/observer, environment router) and
drives the per-beat loop via ``run()``. Process entry point +
``build_runtime_from_config`` live in ``krakey/main.py``; per-beat
algorithm lives in
``krakey/runtime/heartbeat/heartbeat_orchestrator.py``.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from krakey.models.self_model import (
    SelfModelStore, load_self_model_or_default,
)
from krakey.interfaces.engines.recall import RecallSession
from krakey.interfaces.tool import ToolRegistry
from krakey.llm.resolve import AsyncEmbedder, ChatLike
from krakey.models.config import Config, LLMParams
from krakey.models.config_backup import backup_config
from krakey.runtime.stimuli.batch_tracker import BatchTrackerChannel
from krakey.runtime.events.event_bus import EventBus
from krakey.environment.local import LocalEnvironment
from krakey.environment.router import EnvironmentRouter
from krakey.environment.sandbox import SandboxConfig, SandboxEnvironment
from krakey.interfaces.environment import Environment
from krakey.runtime.console.heartbeat_logger import HeartbeatLogger
from krakey.runtime.stimuli.stimulus_buffer import StimulusBuffer


@dataclass
class RuntimeDeps:
    config: Config
    # ``self_llm`` is the only LLM whose presence is load-bearing
    # for the heartbeat. ``None`` is allowed (pause mode) so the
    # runtime can still come up — channels/sensories/dashboard
    # all start, the heartbeat loop just doesn't tick. Useful when
    # the user installs Krakey and skips chat config in onboarding,
    # planning to fill it in via the dashboard.
    self_llm: ChatLike | None
    compact_llm: ChatLike
    classify_llm: ChatLike
    embedder: AsyncEmbedder
    self_model_path: str | None = None      # default: workspace/self_model.yaml
    config_path: str | None = None          # default: config.yaml — for dashboard
    backup_dir: str | None = None           # default: workspace/backups
    # Root directory holding per-plugin folders. Each plugin's user
    # config lives at ``<root>/<name>/config.yaml`` co-located with
    # whatever else the plugin needs. Default ``workspace/plugins``.
    # Tests pass a tmpdir so per-plugin config writes (history_path,
    # llm_purposes, …) don't bleed into the production workspace.
    plugin_configs_root: str | None = None  # default: workspace/plugins
    # in_mind Modifier's state file. None → workspace/data/in_mind.json
    # (the locked-in production path, see design doc Modifier #3).
    # Tests pass a tmpdir so update_in_mind dispatches don't bleed
    # into the production state.
    in_mind_state_path: str | None = None
    # Sliding window persistence file. None → workspace/data/
    # sliding_window.json. Tests pass a tmpdir so per-test
    # heartbeat history doesn't bleed across runs. Set to ""
    # (empty string) to opt out of persistence entirely — the
    # window stays in-memory only and a process restart loses
    # working memory (the pre-2026-05-07 behavior).
    sliding_window_state_path: str | None = None
    # Shared LLMClient cache keyed by tag name. Populated by
    # build_runtime_from_config for core purposes; Runtime adds plugin
    # purpose entries on top. Sharing the cache means two purposes that
    # map to the same tag share one client (saves connections + keeps
    # rate-limit accounting consistent).
    llm_clients_by_tag: dict[str, Any] = field(default_factory=dict)
    # Shared LLMClientFactoryEngine instance. composition root builds
    # one via EngineRegistry and threads it here so Engine resolution
    # in Runtime can pass it into other engines that need it
    # (Embedder / Reranker / HypothalamusDecisionEngine). Sharing the
    # factory means every engine sees the same per-tag client cache —
    # no duplicate clients for the same tag across engines.
    llm_factory: Any = None
    # EnvironmentRouter — central dispatch for plugin → environment
    # requests. ``None`` means Runtime will build one from
    # ``config.sandbox`` (and, after commit 6, ``config.environments``)
    # at construction time. Tests pass a pre-built Router (or leave
    # this None for the default empty-allow-list build).
    environment_router: EnvironmentRouter | None = None


class Runtime:
    def __init__(self, deps: RuntimeDeps, *, idle_min: float | None = None,
                 idle_max: float | None = None,
                 logger: HeartbeatLogger | None = None,
                 event_bus: EventBus | None = None):
        self.config = deps.config
        self.self_llm = deps.self_llm
        self.compact_llm = deps.compact_llm
        self.embedder = deps.embedder
        # Modifier registry — role-keyed (one Modifier per role; second
        # registration claiming the same role raises). Plugin loader
        # runs LATER (after tools + channels registries exist)
        # since a single plugin can contribute components of all three
        # kinds; loading them before all three registries are built
        # would crash.
        from krakey.interfaces.modifier import ModifierRegistry
        self.modifiers = ModifierRegistry()
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
        # ExplicitHistory Engine — working-memory window. Renamed from
        # the inline-constructed ``SlidingWindow`` because sliding-of-
        # rounds is just one possible strategy; future Engines could
        # implement summary trees, relevance-scored caches, etc.
        # ``state_path`` semantics carried over: ``""`` opts out of
        # persistence (tests / pure-memory mode), ``None`` falls
        # back to the default workspace path.
        if deps.sliding_window_state_path == "":
            sw_state_path: Path | None = None
        else:
            sw_state_path = Path(
                deps.sliding_window_state_path
                or "workspace/data/sliding_window.json"
            )
        from krakey.engines.registry import EngineRegistry
        from krakey.interfaces.engines import (
            ContextEngine,
            DecisionEngine,
            DispatchEngine,
            ExplicitHistoryEngine,
            HeartbeatEngine,
            MemoryEngine,
            RecallEngine,
            RerankerEngine,
        )
        self._engine_registry = EngineRegistry(self.config)
        self.llm_factory = deps.llm_factory
        # Reranker Engine — composition root resolves it BEFORE the
        # recall + dispatch slots that take it as a constructor kwarg.
        # The default impl wraps the configured reranker tag with a
        # preserve-order fallback, so the slot is always populated
        # even when no tag is bound.
        self.reranker = self._engine_registry.resolve(
            "reranker",
            default_path=(
                "krakey.engines.reranker.default:DefaultRerankerEngine"
            ),
            expected_protocol=RerankerEngine,
            factory=self.llm_factory,
        )
        self.explicit_history = self._engine_registry.resolve(
            "explicit_history",
            default_path=(
                "krakey.engines.explicit_history.default:"
                "SlidingWindowExplicitHistoryEngine"
            ),
            expected_protocol=ExplicitHistoryEngine,
            history_token_budget=_history_budget,
            state_path=sw_state_path,
        )

        # Context Engine — prompt assembly.
        self.context = self._engine_registry.resolve(
            "context",
            default_path=(
                "krakey.engines.context.default:DefaultContextEngine"
            ),
            expected_protocol=ContextEngine,
        )

        # Decision Engine — translates Self's [DECISION] text into a
        # structured DecisionResult (tool calls + memory writes +
        # sleep flag + parse failures). Default impl is the scripted
        # ``<tool_call>`` parser; users swap in
        # HypothalamusDecisionEngine (or their own LLM-based
        # translator) by setting ``cfg.core_implementations.decision``.
        # Receives the shared factory so an alternative impl can
        # reach LLM clients via the same per-tag cache the rest of
        # the runtime uses. Default ToolCallParserDecisionEngine
        # accepts and ignores the kwarg.
        self.decision = self._engine_registry.resolve(
            "decision",
            default_path=(
                "krakey.engines.decision.tool_call_parser:"
                "ToolCallParserDecisionEngine"
            ),
            expected_protocol=DecisionEngine,
            cfg=self.config,
            factory=self.llm_factory,
        )

        # Memory Engine — unified GM CRUD + KB fleet management +
        # sleep cycle behind one MemoryEngine Protocol. Custom impls
        # override ``cfg.core_implementations.memory`` with a class
        # satisfying ``MemoryEngine``.
        gm_path = self.config.graph_memory.db_path or ":memory:"
        self.memory = self._engine_registry.resolve(
            "memory",
            default_path=(
                "krakey.engines.memory.default:GraphMemoryEngine"
            ),
            expected_protocol=MemoryEngine,
            db_path=gm_path,
            embedder=deps.embedder,
            kb_dir=(
                self.config.knowledge_base.dir
                or "workspace/data/knowledge_bases"
            ),
            auto_ingest_threshold=(
                self.config.graph_memory.auto_ingest_similarity_threshold
            ),
            extractor_llm=deps.classify_llm,
            classifier_llm=deps.classify_llm,
        )

        # Recall Engine resolve — placed AFTER memory because the
        # default IncrementalRecallEngine takes the resolved memory
        # instance as a constructor input.
        self.recall = self._engine_registry.resolve(
            "recall",
            default_path=(
                "krakey.engines.recall.default:IncrementalRecallEngine"
            ),
            expected_protocol=RecallEngine,
            cfg=self.config,
            memory=self.memory,
            embedder=deps.embedder,
            reranker=self.reranker,
        )

        # Dispatch Engine — runs the 4 side-effects of a
        # DecisionResult (log + publish, dispatch tool calls,
        # apply memory writes, apply memory updates). Default
        # impl wraps the long-standing DecisionDispatcher class
        # so behavior is unchanged. Users override via
        # cfg.core_implementations.dispatch (e.g. RemoteDispatchEngine
        # ships tool execution to a worker over HTTP).
        self.dispatch = self._engine_registry.resolve(
            "dispatch",
            default_path=(
                "krakey.engines.dispatch.default:LocalDispatchEngine"
            ),
            expected_protocol=DispatchEngine,
            cfg=self.config,
        )

        self.tools = ToolRegistry()
        # Built-in tools registered BEFORE the plugin loader runs so
        # plugins can't shadow them by registering a same-named Tool
        # (the registry's register() raises on duplicate name).
        from krakey.runtime.builtin_tools import SleepTool
        self.tools.register(SleepTool())
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
        # channel set (see krakey/runtime/stimuli/stimulus_buffer.py).
        # BatchTracker is core runtime infrastructure (dispatch wake-up
        # mechanism) — kept out of the plugin system so it can't be
        # disabled by accident.
        self.batch_tracker = BatchTrackerChannel()
        self.buffer.register(self.batch_tracker)

        # Plugin registration. Each plugin can contribute modifier /
        # tool / channel components in any combination via its
        # meta.yaml. Has to run AFTER the three registries exist
        # (above) because one plugin may register into any of them.
        #
        # Bring up log + events + config paths BEFORE plugin registration
        # so the dashboard plugin (which pulls a runtime ref via
        # ctx.services["runtime"]) sees a runtime with the fields it
        # needs at channel.start() time.
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

        # Environment Router. Accept a caller-provided Router (tests
        # pre-build a custom allow-list) or build one from
        # ``config.environments`` here. Always-on Local + optional
        # Sandbox-when-configured; allow-list comes straight from
        # the ``allowed_plugins`` lists in the same config block.
        if deps.environment_router is not None:
            self.environment_router = deps.environment_router
        else:
            self.environment_router = self._build_environment_router()
        # Re-bind onto deps so PluginContext can reach the Router via
        # ``ctx.deps.environment_router`` (ctx.environment(...) wrapper).
        deps.environment_router = self.environment_router

        # Self-model setup runs BEFORE the plugin loader so the
        # bootstrap modifier (and any future plugin) can reach the
        # SelfModelStore via ``ctx.services["self_model_store"]``.
        sm_path = deps.self_model_path or "workspace/self_model.yaml"
        self._self_model_store = SelfModelStore(sm_path)
        self.self_model, _detected_bootstrap = load_self_model_or_default(
            sm_path,
        )

        from krakey.runtime.plugin_register.loader import PluginLoader
        self._plugin_loader = PluginLoader(
            config=self.config,
            modifiers=self.modifiers,
            tools=self.tools,
            channels=self.buffer,
            services={
                "runtime": self,
                "memory": self.memory,
                "embedder": self.embedder,
                "reranker": self.reranker,
                "buffer": self.buffer,
                "events": self.events,
                "config": self.config,
                # Bootstrap plugin reads SelfModelStore from services
                # so it can write self_model patches on NoteEvent
                # without reaching back into the runtime.
                "self_model_store": self._self_model_store,
            },
        )
        self._plugin_loader.register_from_config(deps)
        # Stashed so hot_reload_plugins (called later from the
        # dashboard) can re-invoke ``register_one`` with the same
        # deps. Plugins that need a fresh ctx per call build it
        # inside register_one — deps is the constant boundary.
        self._deps = deps

        self.heartbeat_count = 0
        # Per-process sleep-cycle counter, surfaced via /status + the
        # dashboard. Not persisted across restarts.
        self._sleep_cycles = 0
        self._stop = False
        self._min = idle_min if idle_min is not None else self.config.idle.min_interval
        self._max = idle_max if idle_max is not None else self.config.idle.max_interval

        self._recall: RecallSession | None = None
        self._classify_tasks: list[asyncio.Task] = []
        self._last_node_count = 0
        self._last_edge_count = 0

        # Heartbeat algorithm — owns the per-beat orchestration but
        # holds no state. Reads + mutates Runtime fields through the
        # `runtime` ref. Built last (after all collaborators exist) so
        # nothing in its phase methods sees a half-constructed Runtime.
        from krakey.runtime.heartbeat.heartbeat_orchestrator import HeartbeatOrchestrator
        self._orchestrator = HeartbeatOrchestrator(self)

        # Heartbeat Engine — owns the per-beat run loop. The default
        # impl wraps the orchestrator just constructed; user impls
        # via cfg.core_implementations.heartbeat replace the entire
        # cognitive cadence (phase order, multi-stage thinking,
        # event-driven scheduling, etc.). Setup + teardown around
        # the loop stay on Runtime.run().
        self.heartbeat = self._engine_registry.resolve(
            "heartbeat",
            default_path=(
                "krakey.engines.heartbeat.default:DefaultHeartbeatEngine"
            ),
            expected_protocol=HeartbeatEngine,
            cfg=self.config,
        )

        # Snapshot config.yaml on every startup so a bad save can be rolled
        # back from workspace/backups/.
        if self._config_path:
            try:
                backup_config(self._config_path, self._backup_dir)
            except Exception as e:  # noqa: BLE001
                self.log.runtime_error(f"config backup failed: {e}")

        # Modifier attach() lifecycle hook — fires after every other
        # subsystem (ToolRegistry, plugin loader, etc.) is up so
        # Modifiers with extra runtime-coupled wiring beyond what the
        # meta.yaml components: list expresses can do it without
        # ordering surprises. No in-tree Modifier uses this today —
        # multi-component plugins ship sibling tools via meta.yaml
        # — but the hook stays available. attach_all is exception-
        # tolerant by contract; one bad Modifier won't block the others.
        self.modifiers.attach_all(self)

        # Plugin observer is built AFTER attach_all so the snapshot it
        # walks includes anything an attach() hook added. Components
        # registered by the loader appear with source="builtin";
        # everything else (BatchTracker, attach() extras) appears as
        # source="core".
        from krakey.runtime.plugin_register.observer import PluginObserver
        self._plugin_observer = PluginObserver(
            modifiers=self.modifiers,
            tools=self.tools,
            channels=self.buffer,
            loader=self._plugin_loader,
        )

    async def hot_reload_plugins(
        self,
        target_plugin_names: list[str],
        *,
        force_reload: bool = True,
    ) -> dict[str, Any]:
        """Reconcile loaded plugins with ``target_plugin_names``.

        Three actions per plugin:

          - target ∈ loaded, force_reload=True
                → unregister + register (full reload, picks up
                  config edits + LLM-binding changes)
          - target ∉ loaded
                → register_one (hot-add)
          - loaded ∉ target
                → unregister_one (hot-disable)

        Newly-registered channels are started immediately via
        ``buffer.start_one(name)``. Removed channels are stopped
        by ``buffer.deregister(name)`` inside ``unregister_one``.

        Set ``force_reload=False`` to make this method hot-ADD
        only — same shape as the original v1 behaviour. The
        dashboard's "Apply changes" button uses force_reload=True
        because the operator just edited config and expects every
        relevant change (LLM bindings, plugin config) to take
        effect.

        Returns::

            {
              "reloaded": [{plugin, components: [...]}],
              "added":    [{plugin, components: [...]}],
              "removed":  [{plugin, components: [...]}],
              "skipped":  [{plugin, reason}],   # only when
                                                 # force_reload=False
              "errors":   [{plugin, error}],
            }
        """
        report: dict[str, Any] = {
            "reloaded": [], "added": [], "removed": [],
            "skipped":  [], "errors":  [],
        }
        target_set = set(target_plugin_names)
        # Snapshot before we start mutating; unregister + register
        # both rewrite this set.
        loaded_snapshot = set(
            self._plugin_loader.loaded_plugin_names,
        )

        # 1. REMOVE plugins that dropped out of target.
        for plugin_name in sorted(loaded_snapshot - target_set):
            sub = await self._plugin_loader.unregister_one(plugin_name)
            if sub["errors"]:
                report["errors"].extend([
                    {"plugin": plugin_name,
                     "error":
                         f"during unregister: "
                         f"{e['kind']}/{e['name']}: {e['error']}"}
                    for e in sub["errors"]
                ])
            if sub["removed"]:
                report["removed"].append({
                    "plugin":     plugin_name,
                    "components": sub["removed"],
                })

        # 2. RELOAD plugins still in target that were already
        # loaded (force_reload=True path).
        for plugin_name in target_plugin_names:
            if plugin_name not in loaded_snapshot:
                continue  # handled in step 3
            if not force_reload:
                report["skipped"].append({
                    "plugin": plugin_name,
                    "reason":
                        "already loaded; force_reload=False "
                        "(call again with force_reload=True to "
                        "pick up config / LLM binding changes)",
                })
                continue
            sub_unreg = await self._plugin_loader.unregister_one(
                plugin_name,
            )
            if sub_unreg["errors"]:
                report["errors"].extend([
                    {"plugin": plugin_name,
                     "error":
                         f"during unregister: "
                         f"{e['kind']}/{e['name']}: {e['error']}"}
                    for e in sub_unreg["errors"]
                ])
            sub_reg = self._plugin_loader.register_one(
                plugin_name, self._deps,
            )
            if sub_reg["ok"]:
                for comp in sub_reg["components"]:
                    if comp["kind"] == "channel":
                        try:
                            await self.buffer.start_one(comp["name"])
                        except Exception as e:  # noqa: BLE001
                            report["errors"].append({
                                "plugin": plugin_name,
                                "error":
                                    f"channel {comp['name']} "
                                    f"failed to start: "
                                    f"{type(e).__name__}: {e}",
                            })
                report["reloaded"].append({
                    "plugin":     plugin_name,
                    "components": sub_reg["components"],
                })
            else:
                report["errors"].append({
                    "plugin": plugin_name,
                    "error":
                        f"during re-register: "
                        f"{sub_reg['error'] or 'unknown'}",
                })

        # 3. ADD plugins that weren't loaded before.
        for plugin_name in target_plugin_names:
            if plugin_name in loaded_snapshot:
                continue  # handled in step 2
            sub = self._plugin_loader.register_one(
                plugin_name, self._deps,
            )
            if sub["ok"]:
                for comp in sub["components"]:
                    if comp["kind"] == "channel":
                        try:
                            await self.buffer.start_one(comp["name"])
                        except Exception as e:  # noqa: BLE001
                            report["errors"].append({
                                "plugin": plugin_name,
                                "error":
                                    f"channel {comp['name']} "
                                    f"failed to start: "
                                    f"{type(e).__name__}: {e}",
                            })
                report["added"].append({
                    "plugin":     plugin_name,
                    "components": sub["components"],
                })
            else:
                report["errors"].append({
                    "plugin": plugin_name,
                    "error":  sub["error"] or "unknown",
                })

        return report

    @property
    def is_bootstrap(self) -> bool:
        """True iff the bootstrap plugin is registered AND active.

        With no plugin loaded the runtime can't drive bootstrap, so
        the answer is False — the runtime is not in bootstrap mode
        by definition. Callers that want to inspect the persisted
        ``state.bootstrap_complete`` marker directly should read
        ``self.self_model`` themselves; this property is the runtime
        view, not the on-disk one.
        """
        anchor = self.modifiers.by_role("bootstrap")
        if anchor is None:
            return False
        return bool(getattr(anchor, "is_active", False))

    @property
    def sleep_cycles(self) -> int:
        """Per-process sleep-cycle count, surfaced via /status + the
        dashboard. Resets across restarts (not persisted)."""
        return self._sleep_cycles

    async def run(self, iterations: int | None = None) -> None:
        await self.memory.initialize()
        await self._preflight_environments()

        # Signal: runtime resources (memory, environments) are live;
        # plugins that subscribed to RuntimeReadyEvent (e.g. the
        # bootstrap modifier probing GM emptiness) can now do their
        # startup work without ordering surprises.
        from krakey.runtime.events.event_types import RuntimeReadyEvent
        self.events.publish(RuntimeReadyEvent())

        # buffer.start_all() walks every registered channel and calls its
        # start(); the dashboard plugin's channel uses that hook to spin
        # up the Web UI server. No special-case "start dashboard" path —
        # dashboard is just a channel like any other.
        await self.buffer.start_all()

        self._recall = self.recall.new_session()

        # Pause mode: no Self LLM bound (user skipped chat in onboarding,
        # plans to fix in dashboard). Channels are alive — the dashboard
        # is up — but the heartbeat doesn't fire. Park here until stop
        # is signalled. Periodic-ish console reminder so an attached
        # terminal doesn't look frozen.
        if self.self_llm is None:
            self.log.hb_warn(
                "no chat LLM configured — heartbeat is paused. "
                "Configure providers in the dashboard's LLM tab "
                "(or re-run `krakey onboard`), then restart."
            )
            try:
                # Short poll interval (0.25s) so Ctrl+C is responsive.
                # On Windows + asyncio, signal delivery only happens
                # when the loop yields back to Python; a long sleep
                # here makes Ctrl+C feel like the program hung.
                while not self._stop:
                    await asyncio.sleep(0.25)
            finally:
                await self.buffer.stop_all()
            return

        try:
            # Heartbeat Engine drives the loop. Setup (gm.initialize
            # + channels.start_all + the install advisory + initial
            # recall session) stayed on Runtime above this try block;
            # teardown (buffer.stop_all + classify-task cancellation)
            # stays in the finally below. The Engine's ``run()`` is
            # purely the iteration loop body — phase ordering happens
            # inside ``beat()`` per the Engine's own contract.
            await self.heartbeat.run(self, iterations)
        finally:
            await self.buffer.stop_all()
            # Cancel in-flight background classify tasks so asyncio doesn't warn.
            pending = [t for t in self._classify_tasks if not t.done()]
            for t in pending:
                t.cancel()

    @property
    def _plugin_infos(self) -> list:
        """PluginInfo list — exposed as a property so call sites that
        historically read ``self._plugin_infos`` (mostly tests) keep
        working. Walks the live registries each call."""
        return self._plugin_observer.collect_infos()

    def loaded_plugin_report(self) -> dict[str, Any]:
        """Pure runtime observation of which tools + channels are
        live (no plugin-config file reads). The dashboard adapter
        combines this with its own ``FilePluginConfigStore`` reads to
        build the ``/api/plugins`` payload."""
        return self._plugin_observer.loaded_report()

    def _build_environment_router(self) -> EnvironmentRouter:
        """Compose Local + Sandbox-if-configured into a Router whose
        allow-list comes straight from ``config.environments``.

        Local is always registered — it's zero-config and never
        fails to start. Its allow-list is whatever the user put in
        ``environments.local.allowed_plugins`` (default empty).

        Sandbox is registered only when ``environments.sandbox`` is
        set AND fully configured. Partial config raises so a typo
        doesn't silently downgrade to "no sandbox available".
        """
        envs: dict[str, Environment] = {"local": LocalEnvironment()}
        envs_cfg = self.config.environments
        allow_list: dict[str, list[str]] = {
            "local": list(envs_cfg.local.allowed_plugins),
        }
        sb = envs_cfg.sandbox
        if sb is not None:
            missing: list[str] = []
            if not sb.guest_os:
                missing.append("environments.sandbox.guest_os")
            if not sb.agent.url:
                missing.append("environments.sandbox.agent.url")
            if not sb.agent.token:
                missing.append("environments.sandbox.agent.token")
            if missing:
                raise RuntimeError(
                    "sandbox env config is partial. Missing: "
                    + ", ".join(missing)
                    + ". Either complete the `environments.sandbox:` "
                    "block in config.yaml or remove the section "
                    "entirely to disable the sandbox env."
                )
            envs["sandbox"] = SandboxEnvironment(SandboxConfig(
                agent_url=sb.agent.url,
                agent_token=sb.agent.token,
                guest_os=sb.guest_os,
            ))
            allow_list["sandbox"] = list(sb.allowed_plugins)
        return EnvironmentRouter(envs=envs, allow_list=allow_list)

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

    async def _preflight_environments(self) -> None:
        """Walk the Router's envs that have at least one allow-listed
        plugin and confirm each is reachable. Wrapper keeps the
        success log line attached to the heartbeat log; the Router
        itself stays IO-pattern-agnostic.
        """
        infos = await self.environment_router.preflight_all()
        for info in infos:
            env_name = info.get("env", "?")
            details = " ".join(
                f"{k}={v}" for k, v in info.items() if k != "env"
            )
            self.log.hb(f"{env_name} preflight ok: {details}")

    async def close(self) -> None:
        """Shut down every Engine slot that exposes ``close()``.

        Engines are not required to define ``close()`` — only ones
        holding persistent resources (DB connections, sockets,
        threadpools) need the hook. We probe each Engine for the
        method and invoke it; missing closes are silent. Channels —
        including the dashboard plugin's server — are stopped via
        ``buffer.stop_all()`` in ``run()``'s finally block.
        """
        for engine in (
            self.memory, self.context, self.decision, self.recall,
            self.dispatch, self.heartbeat, self.explicit_history,
            self.reranker, self.llm_factory,
        ):
            close = getattr(engine, "close", None)
            if close is None:
                continue
            try:
                result = close()
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:  # noqa: BLE001
                self.log.runtime_error(
                    f"engine {type(engine).__name__} close raised "
                    f"{type(e).__name__}: {e}"
                )

    def request_stop(self) -> None:
        """Cooperative shutdown signal — sets a flag the heartbeat
        loop checks between beats. CLI signal handlers + dashboard
        kill commands + the orchestrator's /kill path call this;
        the private storage stays an implementation detail."""
        self._stop = True

    @property
    def stop_requested(self) -> bool:
        """Public read-side of ``request_stop`` — what the heartbeat
        Engine polls between beats."""
        return self._stop

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





