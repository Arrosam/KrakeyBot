"""KrakeyBot entrypoint + main heartbeat loop (DevSpec §6.4).

Phase 1 wiring: GraphMemory + IncrementalRecall + SlidingWindow + compact +
fatigue calc + BatchTracker + async classify. Phase 2 will add Bootstrap,
KnowledgeBase, and full Sleep.
"""
from __future__ import annotations

import asyncio
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from src.bootstrap import (
    BOOTSTRAP_PROMPT, detect_bootstrap_complete, load_genesis,
    load_self_model_or_default, parse_self_model_update,
)
from src.dashboard.app_factory import create_app as create_dashboard_app
from src.dashboard.events import EventBroadcaster
from src.dashboard.server import DashboardServer
from src.dashboard.web_chat import WebChatHistory
# TentacleCall is the contract dataclass returned by the hypothalamus
# Reflect's translate(); Runtime dispatches it. Lives in the Reflect
# protocol module so the runtime never imports any plugin module.
from src.reflects.protocol import TentacleCall
from src.models.self_model import SelfModelStore
from src.interfaces.sensory import SensoryRegistry
from src.interfaces.tentacle import TentacleRegistry
from src.llm.client import LLMClient
from src.memory.graph_memory import GraphMemory
from src.memory.knowledge_base import KBRegistry
from src.memory.recall import IncrementalRecall, Reranker
from src.models.config import Config, LLMParams, load_config
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
from src.memory.sleep.sleep_manager import enter_sleep_mode
from src.runtime.compact import compact_if_needed
from src.runtime.fatigue import calculate_fatigue
from src.runtime.hibernate import hibernate_with_recall
from src.runtime.sliding_window import SlidingWindow
from src.runtime.stimulus_buffer import StimulusBuffer
from src.self_agent import parse_self_output
# All tentacle / sensory classes now load via the plugin system
# (src.plugins.builtin.<project>/). SubprocessRunner stays imported
# here because Runtime._build_code_runner owns the sandbox-vs-subprocess
# policy decision and hands the runner to the coding plugin via deps.
from src.sandbox.subprocess_runner import SubprocessRunner


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
    # Per-plugin YAML root. Must be overridable so tests can point at
    # a tmpdir — otherwise the test helper's `config.plugins["web_chat"]`
    # (e.g. a tmpdir history_path) gets shadowed by the production
    # ``workspace/plugin-configs/web_chat.yaml`` that FilePluginConfigStore
    # reads first, and test messages leak into the real user chat log.
    plugin_configs_root: str | None = None  # default: workspace/plugin-configs
    # in_mind Reflect's state file. None → workspace/data/in_mind.json
    # (the locked-in production path, see design doc Reflect #3).
    # Tests pass a tmpdir so update_in_mind dispatches don't bleed
    # into the production state.
    in_mind_state_path: str | None = None
    # Root directory for per-Reflect config files. Each enabled Reflect
    # reads ``<root>/<name>/config.yaml`` (a pure-text settings file
    # the plugin can safely access). None → ``workspace/reflects``.
    # Tests pass a tmpdir so plugin config edits don't bleed into the
    # production workspace.
    reflect_configs_root: str | None = None
    # Shared LLMClient cache keyed by tag name. Populated by
    # build_runtime_from_config for core purposes; Runtime adds plugin
    # purpose entries on top. Sharing the cache means two purposes that
    # map to the same tag share one client (saves connections + keeps
    # rate-limit accounting consistent).
    llm_clients_by_tag: dict[str, Any] = field(default_factory=dict)


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
        # Reflect registry — kind-grouped, ordered storage of pluggable
        # mechanisms. Plugin loader runs LATER (after tentacles +
        # sensories registries exist) since a single plugin can
        # contribute components of all three kinds; loading them
        # before all three registries are built would crash.
        from src.reflects import ReflectRegistry
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
        # Web chat history is a data layer (JSONL persistence + broadcast
        # bus) that the dashboard WebSocket subscribes to. Must exist
        # before the plugin loader so the `web_chat` project (sensory +
        # reply tentacle) can pick it up via deps. Built regardless of
        # `web_chat.enabled` so the dashboard can still display the
        # existing transcript in monitor-only mode.
        #
        # Per-plugin config store: one YAML file per plugin project
        # under workspace/plugin-configs/. On first run the store
        # migrates values from the deprecated config.yaml `plugins:`
        # pile, so users upgrading from the old layout keep their
        # settings. peek_config() is used here because web_chat's file
        # may not exist yet (loader hasn't run) — peek falls back to
        # legacy without materializing anything.
        from src.plugins.plugin_config import FilePluginConfigStore
        # Root is overridable (see RuntimeDeps.plugin_configs_root) so
        # tests can isolate at a tmpdir and their legacy-dict overrides
        # actually take effect.
        plugin_root = Path(deps.plugin_configs_root
                            or "workspace/plugin-configs")
        self._plugin_config_store = FilePluginConfigStore(
            root=plugin_root,
            legacy_plugins=self.config.legacy_plugin_configs,
        )
        wc_cfg = self._plugin_config_store.peek_config("web_chat")
        chat_path = wc_cfg.get("history_path",
                                   "workspace/data/web_chat.jsonl")
        self.web_chat_history = WebChatHistory(chat_path)

        self.sensories = SensoryRegistry()
        # BatchTracker is core runtime infrastructure (dispatch wake-up
        # mechanism) — kept out of the plugin system so it can't be
        # disabled by accident.
        self.batch_tracker = BatchTrackerSensory()
        self.sensories.register(self.batch_tracker)

        # Unified-format plugins (meta.yaml + components list). Each
        # plugin can contribute reflect / tentacle / sensory components
        # in any combination — see src/plugins/unified_discovery.py +
        # docs/design/reflects-and-self-model.md (Samuel 2026-04-26).
        # Has to run AFTER the three registries exist (above) because
        # one plugin may register into any of them.
        self._register_plugins_from_config(deps)

        # Legacy MANIFEST-format plugin loading (tentacles + sensories
        # in the old src/plugins/builtin/<name>/__init__.py shape) is
        # deferred until after self.log is set because discover_plugins
        # logs each plugin's success or failure.
        self._plugin_infos: list = []

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
        # Provisional value used by tests that bypass run() / data probe.
        # Re-derived from GM+KB emptiness once gm.initialize() runs (see
        # _refine_bootstrap_from_data). The override flag pins the value
        # so test-only Runtimes can still force a mode.
        self._bootstrap_overridden = is_bootstrap_override is not None
        self.is_bootstrap = (is_bootstrap_override
                              if is_bootstrap_override is not None
                              else detected_bootstrap)

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

        # Per-run ring buffer of assembled heartbeat prompts for the
        # dashboard Prompts tab. Size from config; not persisted.
        dash_cfg = getattr(self.config, "dashboard", None)
        _pl_size = getattr(dash_cfg, "prompt_log_size", 20) if dash_cfg else 20
        self._prompt_log: deque[dict[str, Any]] = deque(maxlen=max(1, _pl_size))

        # Phase 2 (2026-04-26): all plugins go through
        # _register_plugins_from_config above. _load_plugins() now
        # only derives a PluginInfo list from the already-populated
        # registries so the dashboard's plugin_report() keeps working
        # — no second registration pass needed.
        self._plugin_infos = self._load_plugins()

        # Snapshot config.yaml on every startup so a bad save can be rolled
        # back from workspace/backups/.
        if self._config_path:
            try:
                backup_config(self._config_path, self._backup_dir)
            except Exception as e:  # noqa: BLE001
                self.log.runtime_error(f"config backup failed: {e}")

        # Reflect attach() lifecycle hook — fires after every other
        # subsystem (TentacleRegistry, plugin loader, etc.) is up so
        # Reflects that need to register their own tentacles
        # (in_mind's update_in_mind, and future hooks) can do so
        # without ordering surprises. attach_all is exception-tolerant
        # by contract; one bad Reflect won't block the others.
        self.reflects.attach_all(self)

    def _register_plugins_from_config(
        self, deps: "RuntimeDeps",
    ) -> None:
        """Walk ``config.plugins`` and lazily load each plugin's
        components.

        For each enabled plugin name:
          1. Look up its metadata (no plugin code imported yet).
          2. Read its per-plugin config from
             ``<reflect_configs_root>/<name>/config.yaml`` —
             pure-text settings the plugin is allowed to see (no
             API keys / providers).
          3. For each declared component:
             a. Resolve its declared ``llm_purposes`` entries to
                ``LLMClient`` instances via the shared tag cache,
                using the user's ``llm_purposes`` mapping in the
                per-plugin config.
             b. Build a ``PluginContext`` for that component.
             c. Lazy-import the factory module + invoke factory.
             d. Route the returned object to the right registry by
                component kind (reflect → self.reflects, tentacle →
                self.tentacles, sensory → self.sensories).

        Failure modes are isolated per-plugin AND per-component:
        broken metadata, missing plugin config, factory exceptions,
        and unbound purposes all log + skip without blocking
        startup of the rest. Plugin model is strictly additive
        (CLAUDE.md invariant).

        Phase 1 (Samuel 2026-04-26): only the new meta.yaml-format
        plugins go through here. Tentacle/sensory plugins on the
        legacy ``MANIFEST = {}`` shape are still loaded by
        ``src.plugins.loader.discover_plugins`` (called later in
        Runtime construction); Phase 2 will fold them in too.
        """
        from src.plugins.unified_discovery import (
            discover_plugins as _discover_unified, load_component,
        )
        from src.reflects.context import PluginContext, load_plugin_config

        names = self.config.plugins
        if names is None:
            print(
                "config: no `plugins:` section in config.yaml — "
                "starting with zero unified plugins. Add an explicit "
                "list (e.g. `plugins: [default_recall_anchor]`) to "
                "enable any. Available built-ins: "
                f"{sorted(_discover_unified().keys())}",
                file=sys.stderr,
            )
            return
        if not names:
            return  # explicit empty list: respect, no nag

        all_meta = _discover_unified()
        cfg_root = Path(deps.reflect_configs_root or "workspace/plugins")

        # Whitelisted Runtime-built resources exposed to plugin
        # factories via ``ctx.services``. Same shape as the legacy
        # plugin loader's `deps` dict so old factories' lookup
        # patterns stay familiar.
        services = {
            "gm": self.gm,
            "kb_registry": self.kb_registry,
            "embedder": self.embedder,
            "buffer": self.buffer,
            "web_chat_history": getattr(self, "web_chat_history", None),
            "config": self.config,
            "build_code_runner": self._build_code_runner,
        }

        for plugin_name in names:
            meta = all_meta.get(plugin_name)
            if meta is None:
                print(
                    f"config: unknown plugin {plugin_name!r} — skipping. "
                    f"Available: {sorted(all_meta.keys())}",
                    file=sys.stderr,
                )
                continue

            plugin_cfg = load_plugin_config(plugin_name, cfg_root)
            user_purposes = plugin_cfg.get("llm_purposes") or {}
            if not isinstance(user_purposes, dict):
                print(
                    f"config: {plugin_name}/config.yaml `llm_purposes:` "
                    "must be a mapping; ignoring all purpose bindings",
                    file=sys.stderr,
                )
                user_purposes = {}

            # Single ``plugin_cache`` dict shared across this plugin's
            # components — multi-component plugins (telegram /
            # web_chat) use it to share state between their sensory
            # + tentacle factories.
            plugin_cache: dict[str, Any] = {}

            for component in meta.components:
                ctx_llms: dict[str, Any] = {}
                for purpose_decl in component.llm_purposes:
                    purpose_name = str(purpose_decl.get("name", "")).strip()
                    if not purpose_name:
                        continue
                    tag_name = user_purposes.get(purpose_name)
                    if not isinstance(tag_name, str) or not tag_name:
                        continue
                    client = resolve_llm_for_tag(
                        self.config, tag_name, deps.llm_clients_by_tag,
                    )
                    if client is None:
                        continue
                    ctx_llms[purpose_name] = client

                ctx = PluginContext(
                    deps=deps, plugin_name=plugin_name,
                    config=plugin_cfg, llms=ctx_llms,
                    services=services, plugin_cache=plugin_cache,
                )

                try:
                    instance = load_component(component, ctx)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"config: plugin {plugin_name!r} component "
                        f"{component.kind!r} failed to load: "
                        f"{type(e).__name__}: {e}; skipping.",
                        file=sys.stderr,
                    )
                    continue
                if instance is None:
                    continue  # factory opted out (e.g. unbound LLM)

                self._register_component(plugin_name, component, instance)

    def _register_component(
        self, plugin_name: str, component: "Any", instance: Any,
    ) -> None:
        """Route a built component to the right registry by kind."""
        kind = component.kind
        try:
            if kind == "reflect":
                self.reflects.register(instance)
            elif kind == "tentacle":
                self.tentacles.register(instance)
            elif kind == "sensory":
                self.sensories.register(instance)
            else:
                print(
                    f"config: plugin {plugin_name!r} produced unknown "
                    f"component kind {kind!r}; skipping",
                    file=sys.stderr,
                )
        except Exception as e:  # noqa: BLE001
            print(
                f"config: plugin {plugin_name!r} {kind} registration "
                f"failed: {type(e).__name__}: {e}",
                file=sys.stderr,
            )

    def _new_recall(self) -> IncrementalRecall:
        # Routed through the Reflect registry (kind="recall_anchor")
        # since 2026-04-25. The default built-in mirrors the previous
        # in-line factory; future Reflects (#2 LLM-anchor) will replace
        # it from config without Runtime needing to know the difference.
        return self.reflects.make_recall(self)

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

    async def run(self, iterations: int | None = None) -> None:
        await self.gm.initialize()
        await self._preflight_sandbox()
        await self._refine_bootstrap_from_data()
        await self.sensories.start_all(self.buffer)
        await self._maybe_start_dashboard()

        if self.is_setup_mode:
            await self._run_setup_mode()
            await self.sensories.stop_all()
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
            await self.sensories.stop_all()
            # Cancel in-flight background classify tasks so asyncio doesn't warn.
            pending = [t for t in self._classify_tasks if not t.done()]
            for t in pending:
                t.cancel()

    async def _run_setup_mode(self) -> None:
        """Setup-mode loop: dashboard is up, heartbeat is skipped.

        Idle here until the user signals stop (Ctrl-C / kill / a
        dashboard /api/restart). This gives the Web UI a process to
        live in while the user fills in the missing providers + tags
        + ``core_purposes.self_thinking`` binding.
        """
        host = self.config.dashboard.host
        port = self.config.dashboard.port
        msg = (
            "\n=========================================================\n"
            "  Krakey is in SETUP MODE.\n\n"
            "  No `core_purposes.self_thinking` tag binding found in\n"
            "  config.yaml — the heartbeat is paused. Open the Web UI\n"
            "  to finish setup:\n\n"
            f"      http://{host}:{port}\n\n"
            "  Then in the LLM section: add a provider, define a tag,\n"
            "  bind core_purposes.self_thinking + embedding. Save +\n"
            "  Restart. The next boot will run the real heartbeat.\n"
            "=========================================================\n"
        )
        self.log.runtime_error(msg)
        # Idle until externally stopped.
        while not self._stop:
            await asyncio.sleep(1.0)

    def _load_plugins(self):
        """Legacy MANIFEST plugin loader was removed in Phase 2 of the
        plugin unification (2026-04-26). All plugins now go through
        ``_register_plugins_from_config`` which fires earlier in
        ``__init__``. This method survives only so the dashboard's
        ``plugin_report()`` keeps working — it derives a list of
        already-registered components from the registries instead of
        discovering them anew.
        """
        infos: list = []
        # Reflects: surfaced by name; kind = "reflect"
        for r in self.reflects.by_kind("hypothalamus") + \
                self.reflects.by_kind("recall_anchor") + \
                self.reflects.by_kind("in_mind"):
            from src.plugins.loader import PluginInfo
            infos.append(PluginInfo(
                name=r.name, kind="reflect", source="builtin",
                path="", project=r.name, instance=r,
            ))
        # Tentacles
        for name in sorted(self.tentacles._tentacles.keys()):
            from src.plugins.loader import PluginInfo
            infos.append(PluginInfo(
                name=name, kind="tentacle", source="builtin",
                path="", project=name,
                instance=self.tentacles._tentacles[name],
            ))
        # Sensories
        from src.plugins.loader import PluginInfo
        for s in self.sensories._sensories.values():
            sname = getattr(s, "name", type(s).__name__)
            infos.append(PluginInfo(
                name=sname, kind="sensory", source="builtin",
                path="", project=sname, instance=s,
            ))
        return infos

    def plugin_report(self) -> dict[str, Any]:
        """Serializable snapshot for the dashboard /api/plugins endpoint.

        Unifies three populations:
          - PluginInfos from the loader (builtin + user plugins).
          - Core items still wired hard in __init__ (batch_tracker) —
            reported with source="core" so the UI can show a locked
            badge.

        Each component carries a `values` dict: the current on-disk
        config for its project, minus the loader-owned `enabled` flag
        (which is surfaced separately). The dashboard renders forms
        directly from `config_schema` + `values` — no need to hit
        /api/settings for plugin config anymore.
        """
        def _values_for(project: str) -> dict[str, Any]:
            if not project:
                return {}
            cfg = self._plugin_config_store.peek_config(project)
            return {k: v for k, v in cfg.items() if k != "enabled"}

        def _flatten(infos, loaded_names):
            return [{
                "name": i.name,
                "kind": i.kind,
                "source": i.source,
                "path": i.path,
                "project": i.project,
                "description": i.description,
                "is_internal": i.is_internal,
                "config_schema": i.config_schema,
                # Loader-owned flag so the dashboard can render a
                # dedicated toggle (separate from the plugin's own
                # config_schema rows).
                "enabled": i.enabled,
                "values": _values_for(i.project),
                "loaded": i.name in loaded_names and i.error is None,
                "error": i.error,
            } for i in infos]

        plugin_t_names = {i.name for i in self._plugin_infos
                          if i.kind == "tentacle"}
        plugin_s_names = {i.name for i in self._plugin_infos
                          if i.kind == "sensory"}
        loaded_t = {n for n in self.tentacles._tentacles}  # noqa: SLF001
        loaded_s = {n for n in self.sensories._sensories}  # noqa: SLF001

        core_tentacles = [
            {"name": t.name, "kind": "tentacle", "source": "core",
             "path": "", "project": "", "description": t.description,
             "is_internal": t.is_internal, "config_schema": [],
             "enabled": True, "values": {},
             "loaded": True, "error": None}
            for t in self.tentacles._tentacles.values()  # noqa: SLF001
            if t.name not in plugin_t_names
        ]
        core_sensories = [
            {"name": s.name, "kind": "sensory", "source": "core",
             "path": "", "project": "", "description": "",
             "is_internal": False, "config_schema": [],
             "enabled": True, "values": {},
             "loaded": True, "error": None}
            for s in self.sensories._sensories.values()  # noqa: SLF001
            if s.name not in plugin_s_names
        ]
        t_infos = [i for i in self._plugin_infos if i.kind == "tentacle"]
        s_infos = [i for i in self._plugin_infos if i.kind == "sensory"]
        return {
            "tentacles": core_tentacles + _flatten(t_infos, loaded_t),
            "sensories": core_sensories + _flatten(s_infos, loaded_s),
        }

    def update_plugin_config(
        self, project: str, body: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist an edit from the dashboard into the per-plugin file.

        `body` has shape {"enabled": bool, "values": {...}}. The final
        on-disk file merges them: {"enabled": ..., **values}. Changes
        take effect on next restart (plugins aren't hot-reloaded).
        """
        if not project or not isinstance(project, str):
            raise ValueError("project name required")
        enabled = bool(body.get("enabled", False))
        values = dict(body.get("values") or {})
        # Guard: `enabled` is loader-owned; refuse to let a client
        # stuff it inside `values` and double-write.
        values.pop("enabled", None)
        config = {"enabled": enabled, **values}
        path = self._plugin_config_store.write(project, config)
        return {"project": project, "path": str(path),
                "config": config}

    def _build_code_runner(self, coding_cfg: dict):
        """Return Subprocess on sandbox=false, SandboxRunner otherwise.

        Sandbox defaults to TRUE. When any tentacle enables sandbox but
        the top-level `sandbox` config is incomplete, refuse to start
        with a clear error — user must configure the guest VM first.
        """
        want_sandbox = bool(coding_cfg.get("sandbox", True))
        if not want_sandbox:
            return SubprocessRunner()
        sb = self.config.sandbox
        missing = []
        if not sb.guest_os:
            missing.append("sandbox.guest_os")
        if not sb.agent.url:
            missing.append("sandbox.agent.url")
        if not sb.agent.token:
            missing.append("sandbox.agent.token")
        if missing:
            raise RuntimeError(
                "coding.sandbox=true but sandbox is not configured. "
                "Missing: " + ", ".join(missing) + ". "
                "Either complete the `sandbox:` block in config.yaml or "
                "set tentacle.coding.sandbox=false (unsafe)."
            )
        from src.sandbox.backend import SandboxConfig, SandboxRunner
        return SandboxRunner(SandboxConfig(
            agent_url=sb.agent.url,
            agent_token=sb.agent.token,
            guest_os=sb.guest_os,
        ))

    def _record_prompt(self, heartbeat_id: int, prompt: str) -> None:
        self._prompt_log.append({
            "heartbeat_id": heartbeat_id,
            "ts": datetime.now().isoformat(),
            "full_prompt": prompt,
        })

    def recent_prompts(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Newest-first list of recorded prompts. Used by the dashboard
        /api/prompts endpoint. Returns shallow copies."""
        items = list(self._prompt_log)
        items.reverse()
        if limit is not None:
            items = items[:limit]
        return [dict(p) for p in items]

    async def _preflight_sandbox(self) -> None:
        """Ping the guest agent if any sandboxed tentacle is enabled.
        Refuses to start the runtime when the agent is unreachable."""
        from src.sandbox.backend import (
            SandboxConfig, SandboxUnavailableError, preflight,
        )
        # Per-plugin store replaces config.yaml's `plugins:` pile. peek
        # is safe: by the time preflight runs the loader has already
        # materialized each plugin's file, so every per-plugin dict
        # reflects its on-disk state.
        def _plugin_flag(name: str, key: str, fallback: bool) -> bool:
            return bool(
                self._plugin_config_store.peek_config(name).get(key, fallback)
            )
        any_sandboxed = any(
            _plugin_flag(name, "enabled", False)
            and _plugin_flag(name, "sandbox", True)
            for name in ("coding", "gui_control", "cli",
                           "file_read", "file_write", "browser")
        )
        if not any_sandboxed:
            return
        sb = self.config.sandbox
        cfg = SandboxConfig(
            agent_url=sb.agent.url,
            agent_token=sb.agent.token,
            guest_os=sb.guest_os,
        )
        try:
            info = await preflight(cfg)
        except SandboxUnavailableError as e:
            raise RuntimeError(
                f"sandbox preflight failed: {e}. "
                "Start the guest agent or disable sandboxed tentacles."
            )
        self.log.hb(
            f"sandbox preflight ok: guest_os={info.get('guest_os')} "
            f"agent_version={info.get('agent_version')}"
        )

    async def _refine_bootstrap_from_data(self) -> None:
        """Bootstrap fires only when the workspace is genuinely empty —
        zero GM nodes AND zero (active or archived) KBs. Otherwise the
        agent already has lived experience and shouldn't re-read GENESIS,
        regardless of what self_model.yaml says about bootstrap_complete.

        Override path (`is_bootstrap_override` in __init__) wins for tests.
        """
        if hasattr(self, "_bootstrap_overridden") and self._bootstrap_overridden:
            return
        try:
            n_nodes = await self.gm.count_nodes()
        except Exception:  # noqa: BLE001
            n_nodes = 0
        try:
            kbs = await self.kb_registry.list_kbs(include_archived=True)
        except Exception:  # noqa: BLE001
            kbs = []
        empty = n_nodes == 0 and len(kbs) == 0
        self.is_bootstrap = empty and not self.self_model.get(
            "state", {}).get("bootstrap_complete", False)
        if self.is_bootstrap:
            self.log.hb("bootstrap mode: empty GM + KBs → injecting GENESIS")

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

        # Route inbound user messages through the web_chat sensory
        # (plugin-registered). If the plugin is disabled, no sensory
        # exists — install a noop callback that logs the drop so the
        # dashboard can still show history in monitor-only mode.
        web_chat_sensory = self.sensories._sensories.get("web_chat")  # noqa: SLF001
        if web_chat_sensory is not None:
            _on_user_message = web_chat_sensory.push_user_message
        else:
            async def _on_user_message(text: str,
                                           attachments: list[dict] | None = None
                                           ) -> None:
                self.log.hb_warn(
                    "web_chat plugin disabled — dropping inbound message "
                    f"({len(text or '')} chars)"
                )

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

    def _get_genesis_text(self) -> str:
        """Lazy-load GENESIS.md on first use.

        Bootstrap is the ONLY consumer of this text — after
        bootstrap_complete flips to True, the agent should never see
        GENESIS again. Reading the file unconditionally at startup
        was both wasteful I/O (80% of runs are steady-state) and a
        correctness trap: once stale genesis bytes live on ``self``
        it's too easy to accidentally surface them later.

        Cached on first call so repeat heartbeats during a long
        Bootstrap don't re-read the file 50 times.
        """
        if self._genesis_text is None:
            self._genesis_text = load_genesis(self._genesis_path)
        return self._genesis_text

    def _build_self_prompt(self, stimuli, recall_result,
                              counts: "_GMCounts") -> str:
        # Suppress the [ACTION FORMAT] layer when a hypothalamus
        # Reflect is registered: the translator owns the dispatch
        # path, and teaching Self structured tags would conflict with
        # its job. See docs/design/reflects-and-self-model.md
        # Reflect #1 design.
        in_mind_state = self.reflects.in_mind_state()
        # Standing instruction layer for in_mind. Imported lazily so
        # the IN_MIND_INSTRUCTIONS_LAYER constant only enters memory
        # when an in_mind Reflect is registered — keeps the lazy-load
        # discipline of the plugin folder.
        in_mind_instructions: str | None = None
        if in_mind_state is not None:
            from src.plugins.builtin.default_in_mind.prompt import (
                IN_MIND_INSTRUCTIONS_LAYER,
            )
            in_mind_instructions = IN_MIND_INSTRUCTIONS_LAYER
        prompt = self.builder.build(
            self_model=self.self_model,
            capabilities=self._capabilities(),
            status=self._status(counts.node_count, counts.edge_count,
                                  counts.fatigue_pct, counts.fatigue_hint),
            recall={"nodes": recall_result.nodes,
                    "edges": recall_result.edges},
            window=self.window.get_rounds(),
            stimuli=stimuli,
            current_time=datetime.now(),
            suppress_action_format=self.reflects.has_hypothalamus(),
            in_mind=in_mind_state,
            in_mind_instructions=in_mind_instructions,
        )
        if self.is_bootstrap:
            prompt = (BOOTSTRAP_PROMPT.format(genesis_text=self._get_genesis_text())
                      + "\n\n" + prompt)
        return prompt

    async def _enforce_input_budget(self, stimuli, recall_result,
                                       counts: "_GMCounts"):
        """Overall prompt-budget enforcement (DevSpec \u00a710.2).

        After recall is finalized and we have a candidate prompt, if
        the full prompt exceeds the Self role's ``max_input_tokens``,
        prune the oldest history round into GM (normal compact path)
        and re-run recall (GM changed \u2192 new nodes may be more
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

        self_params = self.config.llm.core_params("self_thinking") or LLMParams()
        budget = int(self_params.max_input_tokens or 128_000)

        async def _recall_fn(text: str):
            return await self.gm.fts_search(text, top_k=10)

        prompt = self._build_self_prompt(stimuli, recall_result, counts)
        max_iters = 10  # safety bound — should never need more than 2-3
        for _ in range(max_iters):
            total = estimate_tokens(prompt)
            if total <= budget:
                return prompt, recall_result
            if not self.window.rounds:
                # Nothing left to borrow from. Log loud and send
                # anyway \u2014 provider will return its own error, which
                # surfaces through the existing Self-LLM-error path.
                self.log.hb_warn(
                    f"prompt {total} > max_input_tokens {budget} and "
                    "window is empty; sending anyway"
                )
                return prompt, recall_result
            oldest = self.window.pop_oldest()
            assert oldest is not None
            self.log.hb(
                f"input budget: prompt {total} > {budget}; pruning oldest "
                f"round (heartbeat #{oldest.heartbeat_id}) into GM"
            )
            try:
                await compact_round(oldest, self.gm, self.compact_llm,
                                      _recall_fn)
            except Exception as e:  # noqa: BLE001 \u2014 never crash the beat
                self.log.hb_warn(
                    f"budget-driven compact failed: {e} \u2014 round "
                    f"#{oldest.heartbeat_id} dropped without GM write"
                )
            # Re-run recall against the (possibly) enriched GM.
            fresh = self._new_recall()
            await fresh.add_stimuli(stimuli)
            recall_result = await fresh.finalize()
            self._recall = fresh
            prompt = self._build_self_prompt(stimuli, recall_result, counts)
        # Hit the iteration ceiling without converging. Ship what we
        # have plus a loud warning; a follow-up beat will try again.
        self.log.hb_warn(
            f"input budget not satisfied after {max_iters} prune "
            f"iterations; sending oversized prompt "
            f"({estimate_tokens(prompt)} > {budget})"
        )
        return prompt, recall_result

    async def _phase_run_self(self, stimuli, recall_result,
                                counts: "_GMCounts"):
        """Build prompt + call Self LLM + parse. Returns None on LLM error
        (sleeps min_interval and short-circuits the heartbeat)."""
        prompt, recall_result = await self._enforce_input_budget(
            stimuli, recall_result, counts,
        )
        self._record_prompt(self.heartbeat_count, prompt)
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
        """Convert Self's response into tentacle calls + dispatch.

        Routes through ``self.reflects.dispatch_decision`` which picks
        the path:
          * Hypothalamus Reflect registered → LLM translation of
            ``parsed.decision`` (existing behavior).
          * No Hypothalamus Reflect → script-only action executor
            scans ``parsed.raw`` for ``[ACTION]...[/ACTION]`` JSONL.

        Returns True iff Self requested sleep (caller short-circuits
        and runs _perform_sleep).
        """
        decision = parsed.decision.strip().lower()
        if not decision or decision in ("no action", "无行动"):
            # Even "no action" decisions can't have ACTION blocks, so
            # short-circuit. (Action executor would also produce empty
            # but skip the work.)
            return False
        try:
            result = await self.reflects.dispatch_decision(
                parsed.raw, parsed.decision,
                self.tentacles.list_descriptions(),
            )
        except Exception as e:  # noqa: BLE001
            # Include exception class — some exceptions (empty str, None
            # deref, bare raise) format to an empty message and mask the
            # real cause.
            err = f"{type(e).__name__}: {e!r}"
            self.log.hb(f"Hypothalamus error: {err}")
            # Self also needs to learn the translation failed — otherwise
            # the decision looks silently dispatched and she can't correct.
            # Push a system_event stimulus with adrenalin so it surfaces on
            # the next heartbeat.
            await self.buffer.push(Stimulus(
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

        # Sleep bookkeeping is a per-process runtime concern, not
        # something Self needs to remember across restarts. Bumping
        # the in-memory counter is enough for /status output + the
        # dashboard's last_sleep stamp (set client-side on the
        # `sleep_done` event).
        self._sleep_cycles += 1

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
            # Catastrophic tentacle crash — worth waking Self regardless of
            # whether the original call was urgent.
            self.log.dispatch(f"{call.tentacle} error: {e}")
            await self.buffer.push(Stimulus(
                type="system_event", source=f"tentacle:{call.tentacle}",
                content=f"error: {e}", timestamp=datetime.now(),
                adrenalin=True,
            ))
            await self.batch_tracker.mark_completed(call_id)
            return

        # Tentacle-feedback stimuli are low-priority receipts by design.
        # The tentacle itself decides whether its outcome is worth
        # interrupting Self's hibernate (failures typically set
        # adrenalin=True in their own return). Do NOT inherit adrenalin
        # from the dispatch: by the time feedback arrives Self has
        # already acted on the urgent upstream signal, and re-waking for
        # the echo just produces avoidable heartbeats.
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
        """Runtime status numbers — changes every beat (heartbeat counter,
        fatigue), so this section is deliberately placed near the end of
        the prompt to preserve the cacheable prefix above it."""
        return {
            "gm_node_count": node_count,
            "gm_edge_count": edge_count,
            "fatigue_pct": pct,
            "fatigue_hint": hint,
            "last_sleep_time": "never",
            "heartbeats_since_sleep": self.heartbeat_count,
        }

    def _capabilities(self) -> list[dict[str, Any]]:
        """Tentacle list for the [CAPABILITIES] layer. Only changes on
        plugin reload, so this gets rendered high in the prompt above the
        cache-breaking volatile layers."""
        return self.tentacles.list_descriptions()


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


# ---------------- production builder ----------------


def resolve_llm_for_tag(
    cfg: Config, tag_name: str | None,
    cache: dict[str, "LLMClient"],
) -> "LLMClient | None":
    """Build (or fetch from cache) the LLMClient for a tag name.

    Shared between the core-purpose loader (build_runtime_from_config)
    and the per-plugin loader (Runtime._register_reflects_from_config)
    so that two purposes pointing at the same tag share one client
    instance — keeps connection state + future rate-limit accounting
    consistent.

    Returns None for: empty tag_name, missing tag in cfg.llm.tags,
    malformed provider field, or provider name not in cfg.llm.providers.
    Each failure mode logs a single stderr warning so the user can
    see what to fix; the runtime continues without that LLM (strictly
    additive plugin model — bad config doesn't crash startup).
    """
    if not tag_name:
        return None
    cached = cache.get(tag_name)
    if cached is not None:
        return cached
    tag = cfg.llm.tags.get(tag_name)
    if tag is None:
        print(f"warning: tag {tag_name!r} not defined in llm.tags",
              file=sys.stderr)
        return None
    try:
        provider_name, model_name = tag.split_provider()
    except ValueError as e:
        print(f"warning: tag {tag_name!r} has bad provider field: {e}",
              file=sys.stderr)
        return None
    provider = cfg.llm.providers.get(provider_name)
    if provider is None:
        print(
            f"warning: tag {tag_name!r} references unknown provider "
            f"{provider_name!r}", file=sys.stderr,
        )
        return None
    client = LLMClient(provider, model_name, params=tag.params)
    cache[tag_name] = client
    return client


def build_runtime_from_config(config_path: str = "config.yaml") -> Runtime:
    """Construct a production Runtime from ``config.yaml``.

    Tag-based resolution path (Samuel 2026-04-26 refactor):
      1. ``cfg.llm.tags`` → name → (provider, model, params)
      2. ``cfg.llm.core_purposes`` → core purpose name → tag name
      3. ``cfg.llm.embedding`` / ``cfg.llm.reranker`` → tag name
         (model-type slots, not "purposes")

    Each LLMClient is built once per tag name and reused across all
    purposes that share it.
    """
    cfg = load_config(config_path)

    # Build LLMClient cache keyed by tag name. Multiple purposes that
    # point at the same tag share a single client. The cache is also
    # passed through to plugin loaders via RuntimeDeps so plugin
    # purposes resolve from the same cache.
    client_cache: dict[str, LLMClient] = {}

    def _client_for_tag(tag_name: str | None) -> LLMClient | None:
        return resolve_llm_for_tag(cfg, tag_name, client_cache)

    # Core purposes (Self / compact / classifier ...). Self is required
    # for the heartbeat to fire; if missing, we DON'T raise — the
    # Runtime drops into "setup mode" (dashboard runs, heartbeat
    # skipped) so the user can complete the config via Web UI without
    # having to hand-edit YAML before they ever see Krakey's UI.
    self_llm = _client_for_tag(cfg.llm.core_purposes.get("self_thinking"))
    compact_llm = _client_for_tag(cfg.llm.core_purposes.get("compact"))
    classify_llm = (
        _client_for_tag(cfg.llm.core_purposes.get("classifier"))
        or compact_llm  # historical reuse
    )
    if compact_llm is None:
        compact_llm = self_llm  # last-resort fallback so sleep doesn't crash
        # (in setup mode self_llm is also None — that's fine, sleep
        # never runs without a heartbeat)

    # Embedding + reranker (model-type slots, not purposes)
    embed_client = _client_for_tag(cfg.llm.embedding)

    async def embedder(text: str) -> list[float]:
        if embed_client is None:
            raise RuntimeError(
                "no embedding tag bound — set llm.embedding to a tag name "
                "in config.yaml (or use the dashboard's LLM section)"
            )
        return await embed_client.embed(text)

    reranker = None
    reranker_client = _client_for_tag(cfg.llm.reranker)
    if reranker_client is not None:
        class _RerankerAdapter:
            async def rerank(self, query, docs):
                return await reranker_client.rerank(query, docs)
        reranker = _RerankerAdapter()

    # hypo_llm is no longer eagerly required at the core level — it's
    # bound through the per-plugin config of `default_hypothalamus`.
    # We still keep the field on RuntimeDeps for back-compat with
    # existing plugin factories that pull `deps.hypo_llm`. Resolve
    # from the dedicated `hypothalamus` core purpose if the user
    # mapped one (compat shim), else None.
    hypo_llm = _client_for_tag(cfg.llm.core_purposes.get("hypothalamus"))

    deps = RuntimeDeps(
        config=cfg, self_llm=self_llm, hypo_llm=hypo_llm,
        compact_llm=compact_llm,
        classify_llm=classify_llm, embedder=embedder, reranker=reranker,
        config_path=str(config_path),
        llm_clients_by_tag=client_cache,  # shared with plugin loader
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
