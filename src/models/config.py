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


@dataclass
class SafetySection:
    gm_node_hard_limit: int
    max_consecutive_no_action: int


@dataclass
class DashboardSection:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass
class Config:
    llm: LLMSection
    hibernate: HibernateSection
    fatigue: FatigueSection
    sliding_window: SlidingWindowSection
    graph_memory: GraphMemorySection
    knowledge_base: KnowledgeBaseSection
    sensory: dict[str, dict[str, Any]]
    tentacle: dict[str, dict[str, Any]]
    sleep: SleepSection
    safety: SafetySection
    dashboard: DashboardSection = field(default_factory=DashboardSection)


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
        sensory=raw.get("sensory") or {},
        tentacle=raw.get("tentacle") or {},
        sleep=SleepSection(max_duration_seconds=raw["sleep"]["max_duration_seconds"]),
        safety=SafetySection(
            gm_node_hard_limit=raw["safety"]["gm_node_hard_limit"],
            max_consecutive_no_action=raw["safety"]["max_consecutive_no_action"],
        ),
        dashboard=_build_dashboard(raw.get("dashboard")),
    )


def _build_dashboard(raw: dict[str, Any] | None) -> DashboardSection:
    raw = raw or {}
    return DashboardSection(
        enabled=bool(raw.get("enabled", False)),
        host=str(raw.get("host", "127.0.0.1")),
        port=int(raw.get("port", 8765)),
    )


def _validate_fatigue_thresholds(f: FatigueSection) -> None:
    bad = [t for t in f.thresholds if t >= f.force_sleep_threshold]
    if bad:
        print(
            f"warning: fatigue threshold(s) {bad} >= force_sleep_threshold "
            f"({f.force_sleep_threshold}); force sleep will fire before hint shows.",
            file=sys.stderr,
        )
