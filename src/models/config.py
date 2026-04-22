"""Config loader for config.yaml (DevSpec §15).

Parses YAML, substitutes ${VAR} from os.environ at load time,
validates fatigue thresholds vs force_sleep_threshold.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class ModelEntry:
    name: str
    capabilities: list[str] = field(default_factory=list)


@dataclass
class Provider:
    type: str
    base_url: str
    api_key: str | None = None
    models: list[ModelEntry] = field(default_factory=list)


@dataclass
class RoleBinding:
    provider: str
    model: str


@dataclass
class LLMSection:
    providers: dict[str, Provider]
    roles: dict[str, RoleBinding]


@dataclass
class HibernateSection:
    min_interval: int
    max_interval: int
    default_interval: int


@dataclass
class FatigueSection:
    gm_node_soft_limit: int
    force_sleep_threshold: int
    thresholds: dict[int, str]


@dataclass
class SlidingWindowSection:
    max_tokens: int


@dataclass
class GraphMemorySection:
    db_path: str
    auto_ingest_similarity_threshold: float
    recall_per_stimulus_k: int
    max_recall_nodes: int
    neighbor_expand_depth: int


@dataclass
class KnowledgeBaseSection:
    dir: str


@dataclass
class SleepSection:
    max_duration_seconds: int
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
    gm_node_hard_limit: int
    max_consecutive_no_action: int


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
    llm: LLMSection
    hibernate: HibernateSection
    fatigue: FatigueSection
    sliding_window: SlidingWindowSection
    graph_memory: GraphMemorySection
    knowledge_base: KnowledgeBaseSection
    # Per-project plugin config. Key = project folder name (matches
    # src/plugins/builtin/<name>/ or workspace/plugins/<name>/). A
    # project can carry one tentacle, one sensory, or a bundle of both
    # that share state (e.g. Telegram: sensory + reply tentacle
    # sharing one HttpTelegramClient).
    plugins: dict[str, dict[str, Any]]
    sleep: SleepSection
    safety: SafetySection
    dashboard: DashboardSection = field(default_factory=DashboardSection)
    sandbox: SandboxSection = field(default_factory=SandboxSection)


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


def _build_llm(raw: dict[str, Any]) -> LLMSection:
    providers: dict[str, Provider] = {}
    for pname, pdata in (raw.get("providers") or {}).items():
        models = [ModelEntry(name=m["name"], capabilities=list(m.get("capabilities", [])))
                  for m in (pdata.get("models") or [])]
        providers[pname] = Provider(
            type=pdata["type"],
            base_url=pdata["base_url"],
            api_key=pdata.get("api_key"),
            models=models,
        )
    roles: dict[str, RoleBinding] = {}
    for rname, rdata in (raw.get("roles") or {}).items():
        roles[rname] = RoleBinding(provider=rdata["provider"], model=rdata["model"])
    return LLMSection(providers=providers, roles=roles)


def load_config(path: str | Path = "config.yaml") -> Config:
    raw_text = Path(path).read_text(encoding="utf-8")
    raw: dict[str, Any] = yaml.safe_load(raw_text)
    raw = _substitute_env(raw)

    fatigue_raw = raw["fatigue"]
    thresholds = {int(k): str(v) for k, v in (fatigue_raw.get("thresholds") or {}).items()}
    fatigue = FatigueSection(
        gm_node_soft_limit=fatigue_raw["gm_node_soft_limit"],
        force_sleep_threshold=fatigue_raw["force_sleep_threshold"],
        thresholds=thresholds,
    )

    _validate_fatigue_thresholds(fatigue)

    hib = raw["hibernate"]
    gm = raw["graph_memory"]
    return Config(
        llm=_build_llm(raw["llm"]),
        hibernate=HibernateSection(
            min_interval=hib["min_interval"],
            max_interval=hib["max_interval"],
            default_interval=hib["default_interval"],
        ),
        fatigue=fatigue,
        sliding_window=SlidingWindowSection(max_tokens=raw["sliding_window"]["max_tokens"]),
        graph_memory=GraphMemorySection(
            db_path=gm["db_path"],
            auto_ingest_similarity_threshold=gm["auto_ingest_similarity_threshold"],
            recall_per_stimulus_k=gm["recall_per_stimulus_k"],
            max_recall_nodes=gm["max_recall_nodes"],
            neighbor_expand_depth=gm["neighbor_expand_depth"],
        ),
        knowledge_base=KnowledgeBaseSection(dir=raw["knowledge_base"]["dir"]),
        plugins=raw.get("plugins") or {},
        sleep=_build_sleep(raw["sleep"]),
        safety=SafetySection(
            gm_node_hard_limit=raw["safety"]["gm_node_hard_limit"],
            max_consecutive_no_action=raw["safety"]["max_consecutive_no_action"],
        ),
        dashboard=_build_dashboard(raw.get("dashboard")),
        sandbox=_build_sandbox(raw.get("sandbox")),
    )


def _build_sandbox(raw: dict[str, Any] | None) -> SandboxSection:
    raw = raw or {}
    res_raw = raw.get("resources") or {}
    agent_raw = raw.get("agent") or {}
    display = str(raw.get("display", "headed")).lower()
    if display not in ("headed", "headless"):
        print(
            f"warning: sandbox.display={display!r} not recognised; "
            "falling back to 'headed'. Valid values: headed | headless.",
            file=sys.stderr,
        )
        display = "headed"
    return SandboxSection(
        guest_os=str(raw.get("guest_os", "")),
        provider=str(raw.get("provider", "qemu")),
        vm_name=str(raw.get("vm_name", "")),
        display=display,
        resources=SandboxResourcesSection(
            cpu=int(res_raw.get("cpu", 2)),
            memory_mb=int(res_raw.get("memory_mb", 4096)),
            disk_gb=int(res_raw.get("disk_gb", 40)),
        ),
        agent=SandboxAgentSection(
            url=str(agent_raw.get("url", "")),
            token=str(agent_raw.get("token", "")),
        ),
        network_mode=str(raw.get("network_mode", "nat_allowlist")),
        allowlist_domains=list(raw.get("allowlist_domains") or []),
    )


def _build_sleep(raw: dict[str, Any]) -> SleepSection:
    return SleepSection(
        max_duration_seconds=raw["max_duration_seconds"],
        min_community_size=int(raw.get("min_community_size", 2)),
        kb_consolidation_threshold=float(raw.get("kb_consolidation_threshold", 0.85)),
        kb_index_max=int(raw.get("kb_index_max", 30)),
        kb_archive_pct=int(raw.get("kb_archive_pct", 10)),
        kb_revive_threshold=float(raw.get("kb_revive_threshold", 0.80)),
    )


def _build_dashboard(raw: dict[str, Any] | None) -> DashboardSection:
    raw = raw or {}
    return DashboardSection(
        enabled=bool(raw.get("enabled", True)),
        host=str(raw.get("host", "127.0.0.1")),
        port=int(raw.get("port", 8765)),
        prompt_log_size=max(1, int(raw.get("prompt_log_size", 20))),
    )


def _validate_fatigue_thresholds(f: FatigueSection) -> None:
    bad = [t for t in f.thresholds if t >= f.force_sleep_threshold]
    if bad:
        print(
            f"warning: fatigue threshold(s) {bad} >= force_sleep_threshold "
            f"({f.force_sleep_threshold}); force sleep will fire before hint shows.",
            file=sys.stderr,
        )
