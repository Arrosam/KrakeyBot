"""Config loader for config.yaml (DevSpec §15).

Parses YAML, substitutes ${VAR} from os.environ at load time,
validates fatigue thresholds vs force_sleep_threshold.

First-run bootstrap: if the target config file is missing, a
defaults-populated file is written at that path and the process
exits with guidance to set LLM providers/API keys. We intentionally
do NOT copy config.yaml.example — the single source of truth for
defaults is the dataclasses below.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class ModelEntry:
    name: str = ""
    capabilities: list[str] = field(default_factory=list)


@dataclass
class Provider:
    type: str = "openai_compatible"
    base_url: str = ""
    api_key: str | None = None
    models: list[ModelEntry] = field(default_factory=list)


@dataclass
class RoleBinding:
    provider: str = ""
    model: str = ""


@dataclass
class LLMSection:
    # Both empty by default — the first-run file will be a usable
    # scaffold and the user fills in providers + role bindings. The
    # runtime bootstrap validates presence of required roles (`self`,
    # `hypothalamus`, `embedding`) and fails loud with guidance.
    providers: dict[str, Provider] = field(default_factory=dict)
    roles: dict[str, RoleBinding] = field(default_factory=dict)


@dataclass
class HibernateSection:
    min_interval: int = 2
    max_interval: int = 300
    default_interval: int = 10


@dataclass
class FatigueSection:
    gm_node_soft_limit: int = 1000
    force_sleep_threshold: int = 1200
    thresholds: dict[int, str] = field(default_factory=lambda: {
        50: "（不繁忙时可以睡眠）",
        75: "（疲劳，需要主动睡眠）",
        100: "（非常疲劳，需要立即找到睡眠的机会）",
    })


@dataclass
class SlidingWindowSection:
    max_tokens: int = 4096


@dataclass
class GraphMemorySection:
    db_path: str = "workspace/data/graph_memory.sqlite"
    auto_ingest_similarity_threshold: float = 0.92
    recall_per_stimulus_k: int = 5
    max_recall_nodes: int = 20
    neighbor_expand_depth: int = 1


@dataclass
class KnowledgeBaseSection:
    dir: str = "workspace/data/knowledge_bases"


@dataclass
class SleepSection:
    max_duration_seconds: int = 7200
    # Communities below this size stay in GM (don't get migrated to a KB).
    # Default 2 = skip pure singletons.
    min_community_size: int = 2
    # KB consolidation: pairwise-merge active KBs whose index vectors
    # (mean of member entry embeddings) are at least this cosine-close.
    kb_consolidation_threshold: float = 0.85
    # When active KB count exceeds this, archive the least-important
    # `kb_archive_pct` percent (importance = entry_count * mean importance).
    # Archived KBs keep their files + entries on disk and their index
    # vector in kb_registry — they just lose their GM index node so they
    # stop bloating recall.
    kb_index_max: int = 30
    kb_archive_pct: int = 10
    # When sleep would create a fresh KB for a new community, first compare
    # the community summary embedding against archived KBs' index vectors;
    # if the cosine similarity to one is at least this, revive that
    # archived KB and write the new entries into it instead. Models the
    # "forgot a topic, then re-encountered it" relearning shortcut.
    kb_revive_threshold: float = 0.80


@dataclass
class SafetySection:
    gm_node_hard_limit: int = 500
    max_consecutive_no_action: int = 50


@dataclass
class DashboardSection:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8765
    # Ring buffer for the "Prompts" tab. Runtime keeps the last N fully
    # built heartbeat prompts so the UI can show a scrollable log rather
    # than only the single latest one. Per-run, not persisted to disk.
    prompt_log_size: int = 20


@dataclass
class SandboxResourcesSection:
    cpu: int = 2
    memory_mb: int = 4096
    disk_gb: int = 40


@dataclass
class SandboxAgentSection:
    url: str = ""
    token: str = ""


@dataclass
class SandboxSection:
    """Sandbox VM configuration. Required when any sandboxed tentacle is
    enabled (coding / gui_control / cli / file_read / file_write / browser).
    Runtime refuses to start if any of those has sandbox=true but the
    required fields here are missing or the agent is unreachable.
    """
    guest_os: str = ""         # "linux" | "macos" | "windows" — REQUIRED
    provider: str = "qemu"     # qemu | virtualbox | utm
    vm_name: str = ""
    # "headed" — user can see the VM's desktop (spice/sdl/vnc window
    # with a display server). "headless" — VM runs with no display.
    # Declarative only for now: the user launches the VM themselves;
    # this flag documents intent + drives lifecycle tooling later.
    display: str = "headed"    # headed | headless
    resources: SandboxResourcesSection = field(
        default_factory=SandboxResourcesSection
    )
    agent: SandboxAgentSection = field(default_factory=SandboxAgentSection)
    # Network model documentation only; enforced in the VM provisioning,
    # not by this config. Stored for clarity + future tooling.
    network_mode: str = "nat_allowlist"  # nat_allowlist | host_only | isolated
    allowlist_domains: list[str] = field(default_factory=list)


@dataclass
class Config:
    llm: LLMSection = field(default_factory=LLMSection)
    hibernate: HibernateSection = field(default_factory=HibernateSection)
    fatigue: FatigueSection = field(default_factory=FatigueSection)
    sliding_window: SlidingWindowSection = field(
        default_factory=SlidingWindowSection
    )
    graph_memory: GraphMemorySection = field(
        default_factory=GraphMemorySection
    )
    knowledge_base: KnowledgeBaseSection = field(
        default_factory=KnowledgeBaseSection
    )
    # Per-project plugin config. Key = project folder name (matches
    # src/plugins/builtin/<name>/ or workspace/plugins/<name>/). A
    # project can carry one tentacle, one sensory, or a bundle of both
    # that share state (e.g. Telegram: sensory + reply tentacle
    # sharing one HttpTelegramClient).
    #
    # DEPRECATED: kept for backwards compatibility only. Phase 2 of the
    # config overhaul moves plugin settings to per-plugin files under
    # workspace/plugin-configs/<project>.yaml; this central dict stays
    # so existing configs still load until the migration lands.
    plugins: dict[str, dict[str, Any]] = field(default_factory=dict)
    sleep: SleepSection = field(default_factory=SleepSection)
    safety: SafetySection = field(default_factory=SafetySection)
    dashboard: DashboardSection = field(default_factory=DashboardSection)
    sandbox: SandboxSection = field(default_factory=SandboxSection)


# ---------------- env substitution ----------------


def _substitute_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            return os.environ.get(m.group(1), m.group(0))
        out = _ENV_PATTERN.sub(repl, value)
        # If still contains unresolved placeholder → treat as None for api keys etc.
        if _ENV_PATTERN.search(out):
            return None
        return out
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    return value


# ---------------- section builders ----------------
#
# Each builder overlays raw YAML on top of the dataclass's defaults so
# sparse configs still load. Absent keys fall back to defaults; an
# explicit empty value (e.g. `thresholds: {}`) is honored as empty.


def _build_llm(raw: dict[str, Any]) -> LLMSection:
    providers: dict[str, Provider] = {}
    for pname, pdata in (raw.get("providers") or {}).items():
        models = [
            ModelEntry(
                name=m.get("name", ""),
                capabilities=list(m.get("capabilities", [])),
            )
            for m in (pdata.get("models") or [])
        ]
        providers[pname] = Provider(
            type=pdata.get("type", "openai_compatible"),
            base_url=pdata.get("base_url", ""),
            api_key=pdata.get("api_key"),
            models=models,
        )
    roles: dict[str, RoleBinding] = {}
    for rname, rdata in (raw.get("roles") or {}).items():
        roles[rname] = RoleBinding(
            provider=rdata.get("provider", ""),
            model=rdata.get("model", ""),
        )
    return LLMSection(providers=providers, roles=roles)


def _build_hibernate(raw: dict[str, Any]) -> HibernateSection:
    d = HibernateSection()
    return HibernateSection(
        min_interval=int(raw.get("min_interval", d.min_interval)),
        max_interval=int(raw.get("max_interval", d.max_interval)),
        default_interval=int(raw.get("default_interval",
                                       d.default_interval)),
    )


def _build_fatigue(raw: dict[str, Any]) -> FatigueSection:
    d = FatigueSection()
    if "thresholds" in raw:
        thresholds = {
            int(k): str(v) for k, v in (raw["thresholds"] or {}).items()
        }
    else:
        thresholds = d.thresholds
    return FatigueSection(
        gm_node_soft_limit=int(raw.get("gm_node_soft_limit",
                                         d.gm_node_soft_limit)),
        force_sleep_threshold=int(raw.get("force_sleep_threshold",
                                             d.force_sleep_threshold)),
        thresholds=thresholds,
    )


def _build_sliding_window(raw: dict[str, Any]) -> SlidingWindowSection:
    d = SlidingWindowSection()
    return SlidingWindowSection(
        max_tokens=int(raw.get("max_tokens", d.max_tokens)),
    )


def _build_graph_memory(raw: dict[str, Any]) -> GraphMemorySection:
    d = GraphMemorySection()
    return GraphMemorySection(
        db_path=str(raw.get("db_path", d.db_path)),
        auto_ingest_similarity_threshold=float(
            raw.get("auto_ingest_similarity_threshold",
                     d.auto_ingest_similarity_threshold)
        ),
        recall_per_stimulus_k=int(raw.get("recall_per_stimulus_k",
                                              d.recall_per_stimulus_k)),
        max_recall_nodes=int(raw.get("max_recall_nodes",
                                         d.max_recall_nodes)),
        neighbor_expand_depth=int(raw.get("neighbor_expand_depth",
                                              d.neighbor_expand_depth)),
    )


def _build_kb(raw: dict[str, Any]) -> KnowledgeBaseSection:
    d = KnowledgeBaseSection()
    return KnowledgeBaseSection(dir=str(raw.get("dir", d.dir)))


def _build_sleep(raw: dict[str, Any]) -> SleepSection:
    d = SleepSection()
    return SleepSection(
        max_duration_seconds=int(raw.get("max_duration_seconds",
                                              d.max_duration_seconds)),
        min_community_size=int(raw.get("min_community_size",
                                           d.min_community_size)),
        kb_consolidation_threshold=float(
            raw.get("kb_consolidation_threshold",
                     d.kb_consolidation_threshold)
        ),
        kb_index_max=int(raw.get("kb_index_max", d.kb_index_max)),
        kb_archive_pct=int(raw.get("kb_archive_pct", d.kb_archive_pct)),
        kb_revive_threshold=float(raw.get("kb_revive_threshold",
                                               d.kb_revive_threshold)),
    )


def _build_safety(raw: dict[str, Any]) -> SafetySection:
    d = SafetySection()
    return SafetySection(
        gm_node_hard_limit=int(raw.get("gm_node_hard_limit",
                                           d.gm_node_hard_limit)),
        max_consecutive_no_action=int(
            raw.get("max_consecutive_no_action", d.max_consecutive_no_action)
        ),
    )


def _build_dashboard(raw: dict[str, Any] | None) -> DashboardSection:
    raw = raw or {}
    d = DashboardSection()
    return DashboardSection(
        enabled=bool(raw.get("enabled", d.enabled)),
        host=str(raw.get("host", d.host)),
        port=int(raw.get("port", d.port)),
        prompt_log_size=max(1, int(raw.get("prompt_log_size",
                                               d.prompt_log_size))),
    )


def _build_sandbox(raw: dict[str, Any] | None) -> SandboxSection:
    raw = raw or {}
    d = SandboxSection()
    res_raw = raw.get("resources") or {}
    agent_raw = raw.get("agent") or {}
    display = str(raw.get("display", d.display)).lower()
    if display not in ("headed", "headless"):
        print(
            f"warning: sandbox.display={display!r} not recognised; "
            "falling back to 'headed'. Valid values: headed | headless.",
            file=sys.stderr,
        )
        display = "headed"
    return SandboxSection(
        guest_os=str(raw.get("guest_os", d.guest_os)),
        provider=str(raw.get("provider", d.provider)),
        vm_name=str(raw.get("vm_name", d.vm_name)),
        display=display,
        resources=SandboxResourcesSection(
            cpu=int(res_raw.get("cpu", d.resources.cpu)),
            memory_mb=int(res_raw.get("memory_mb", d.resources.memory_mb)),
            disk_gb=int(res_raw.get("disk_gb", d.resources.disk_gb)),
        ),
        agent=SandboxAgentSection(
            url=str(agent_raw.get("url", d.agent.url)),
            token=str(agent_raw.get("token", d.agent.token)),
        ),
        network_mode=str(raw.get("network_mode", d.network_mode)),
        allowlist_domains=list(raw.get("allowlist_domains")
                                 or d.allowlist_domains),
    )


# ---------------- dump / ensure ----------------


def dump_config(cfg: Config) -> str:
    """Serialize a Config dataclass to the YAML text we'd write to disk.

    Round-trips cleanly through load_config: `dump_config(Config())` is
    a valid minimal config that load_config accepts without error.

    fatigue.thresholds uses int keys in memory; YAML tolerates that but
    some downstream tools don't, so we normalize to string keys on the
    way out. load_config casts them back to int on the way in.
    """
    data: dict[str, Any] = asdict(cfg)
    ft = data.get("fatigue") or {}
    if "thresholds" in ft:
        ft["thresholds"] = {str(k): v for k, v in ft["thresholds"].items()}
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)


def ensure_config(path: str | Path = "config.yaml") -> bool:
    """Create a defaults-populated config at `path` if it does not exist.

    Returns True iff a new file was written. Parent directories are
    created as needed.
    """
    p = Path(path)
    if p.exists():
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dump_config(Config()), encoding="utf-8")
    return True


# ---------------- loader ----------------


class _ConfigBootstrapExit(SystemExit):
    """SystemExit subclass raised after first-run config generation so
    tests can distinguish it from unrelated exits."""
    pass


def load_config(path: str | Path = "config.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        ensure_config(p)
        print(
            f"✨ Generated default config at {p}\n"
            f"   Next steps:\n"
            f"     1. Add at least one provider under llm.providers with a\n"
            f"        valid api_key.\n"
            f"     2. Bind the required roles under llm.roles: self,\n"
            f"        hypothalamus, embedding (compact/reranker optional).\n"
            f"     3. Re-run Krakey.",
            file=sys.stderr,
        )
        raise _ConfigBootstrapExit(1)

    raw_text = p.read_text(encoding="utf-8")
    raw: dict[str, Any] = yaml.safe_load(raw_text) or {}
    raw = _substitute_env(raw)

    fatigue = _build_fatigue(raw.get("fatigue") or {})
    _validate_fatigue_thresholds(fatigue)

    return Config(
        llm=_build_llm(raw.get("llm") or {}),
        hibernate=_build_hibernate(raw.get("hibernate") or {}),
        fatigue=fatigue,
        sliding_window=_build_sliding_window(raw.get("sliding_window") or {}),
        graph_memory=_build_graph_memory(raw.get("graph_memory") or {}),
        knowledge_base=_build_kb(raw.get("knowledge_base") or {}),
        plugins=raw.get("plugins") or {},
        sleep=_build_sleep(raw.get("sleep") or {}),
        safety=_build_safety(raw.get("safety") or {}),
        dashboard=_build_dashboard(raw.get("dashboard")),
        sandbox=_build_sandbox(raw.get("sandbox")),
    )


def _validate_fatigue_thresholds(f: FatigueSection) -> None:
    bad = [t for t in f.thresholds if t >= f.force_sleep_threshold]
    if bad:
        print(
            f"warning: fatigue threshold(s) {bad} >= force_sleep_threshold "
            f"({f.force_sleep_threshold}); force sleep will fire before hint shows.",
            file=sys.stderr,
        )
