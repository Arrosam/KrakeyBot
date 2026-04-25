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
from dataclasses import asdict, dataclass, field, fields
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
class LLMParams:
    """Per-role LLM call parameters.

    These are overlaid on `LLMClient` requests. Fields default to values
    that work for a general-purpose chat role; specific roles (self,
    hypothalamus, compact, classifier, embedding) get their own sensible
    defaults applied on top via `_ROLE_DEFAULTS` before the user's YAML
    overrides are merged in.

    Provider adaptation is handled inside `LLMClient`:
      * `reasoning_mode` is translated to the provider-native field
        (Anthropic `thinking.budget_tokens`, OpenAI `reasoning_effort`).
        Set to "off" to disable.
      * `response_format="json_object"` becomes
        `response_format={"type":"json_object"}` on OpenAI-compatible;
        Anthropic has no native JSON-mode so the field is ignored there.
      * Fields the provider cannot accept are silently dropped rather
        than sent (e.g. `temperature` on DeepSeek-Reasoner).

    Token fields intentionally spell out their direction:
      * ``max_output_tokens`` \u2014 upper bound on generation. Translated
        to Anthropic ``max_tokens``, OpenAI classic ``max_tokens``,
        OpenAI reasoning ``max_completion_tokens``, Gemini
        ``maxOutputTokens``.
      * ``max_input_tokens`` \u2014 input-context-window budget for the
        backing model. If left None at load time, the config loader
        resolves it via ``src.utils.model_context.resolve_max_input_tokens``
        (known-prefix lookup, default 128_000). This is **active**:
        used by the runtime for sliding-window history budget, recall
        budget, and overall-prompt enforcement.

    Prompt-budget allocation (only meaningful for Self's role, since
    Self is the only consumer of the sliding window + GM recall):
      * ``history_token_fraction`` \u2014 fraction of ``max_input_tokens``
        reserved for [HISTORY]. Default 0.4. When the window's token
        total exceeds this fraction the compactor pops oldest rounds
        into GM.
      * ``recall_token_budget`` \u2014 ABSOLUTE token cap for the
        [GRAPH MEMORY] section (not a fraction). Too many recall nodes
        pollute context with marginal relevance, so this scales poorly
        with bigger context \u2014 a model with 2M tokens doesn't want 500
        recall items. Default 3000.

    None means "do not send this field" (use provider's own default).
    """
    # Generation bounds
    max_output_tokens: int | None = 4096
    # Input-context budget. None at construction time; the loader
    # resolves it via model_context lookup before the runtime starts.
    max_input_tokens: int | None = None
    # Prompt-budget allocation (Self role only in practice)
    history_token_fraction: float = 0.4
    recall_token_budget: int = 3000
    temperature: float | None = 0.7
    top_p: float | None = None
    stop_sequences: list[str] = field(default_factory=list)
    response_format: str | None = None   # None | "json_object"
    seed: int | None = None

    # Reasoning / thinking (provider-abstracted)
    # off | low | medium | high
    reasoning_mode: str = "off"
    reasoning_budget_tokens: int | None = None

    # Transport-level knobs
    timeout_seconds: float = 120.0
    max_retries: int = 3
    retry_on_status: list[int] = field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )


# Role → default param overrides applied before YAML overlay. Any field
# not mentioned here falls back to LLMParams' universal defaults.
#
# Note on `response_format`: deliberately *not* defaulted to "json_object"
# for Hypothalamus / compact / classifier. Many OpenAI-compatible
# providers (xunfei / zhipu / moonshot / baichuan / …) either ignore the
# field or crash their backends with "EngineInternalError:Unexpected EOF"
# style 500s when they see it. Hypothalamus already has a robust JSON
# extractor (markdown fence, regex, smart-quote/trailing-comma fixups)
# that works without JSON mode, so turning it on is a cost/benefit loss
# in the general case. Users whose provider supports it (Anthropic does
# not, OpenAI + Gemini do) can opt in via YAML per-role.
_ROLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "self": {
        "max_output_tokens": 8192,
        "temperature": 0.7,
        "reasoning_mode": "medium",
        "reasoning_budget_tokens": 4096,
        "timeout_seconds": 180.0,
    },
    "hypothalamus": {
        # 2048 not 512 — multi-action decisions with Chinese tentacle
        # intents blow past 512 easily and get truncated mid-JSON,
        # which some providers surface as an "Unexpected EOF" 500.
        "max_output_tokens": 2048,
        "temperature": 0.0,
        "reasoning_mode": "off",
        "timeout_seconds": 20.0,
    },
    "compact": {
        "max_output_tokens": 2048,
        "temperature": 0.2,
        "reasoning_mode": "off",
        "timeout_seconds": 60.0,
    },
    "classifier": {
        "max_output_tokens": 1024,
        "temperature": 0.0,
        "reasoning_mode": "off",
        "timeout_seconds": 30.0,
    },
    "embedding": {
        # Embedding endpoints don't use generation params; keep a short
        # timeout since embedding calls should be fast.
        "timeout_seconds": 20.0,
    },
    "reranker": {
        "timeout_seconds": 20.0,
    },
}


# Human-readable descriptions used both for docstrings and for the
# `/api/config/schema` endpoint that feeds the dashboard UI. Keep this
# in sync with LLMParams fields above.
_LLM_PARAM_HELP: dict[str, str] = {
    "max_output_tokens": "生成 (输出) token 上限。按 provider 自动翻译: Anthropic max_tokens, OpenAI 经典 max_tokens, OpenAI reasoning max_completion_tokens, Gemini maxOutputTokens。Anthropic 必填。",
    "max_input_tokens": "输入 (prompt) 上下文 token 预算。留空则启动时按 model 名字自动查表 (未知模型默认 128000)。驱动 sliding window 压缩阈值、GM recall 预算、整体 prompt 超限自动裁剪 history。",
    "history_token_fraction": "Self role 独有。[HISTORY] 层占用 max_input_tokens 的比例。默认 0.4 = 40%。超过这个比例就触发 compact 把最老 round 收入 GM。",
    "recall_token_budget": "Self role 独有。[GRAPH MEMORY] 召回节点的总 token 预算 (绝对值, 不是比例)。默认 3000。太多 recall 会污染 context, 所以不跟 context 规模线性增长。",
    "temperature": "采样温度。0 = 确定性，越大越发散。部分 reasoning 模型 (OpenAI o-series, DeepSeek Reasoner) 不支持，会被自动忽略。",
    "top_p": "nucleus sampling 阈值 (0-1)。通常和 temperature 二选一。留空 = 不发送此字段。",
    "stop_sequences": "停止序列列表。遇到任一即停止生成。",
    "response_format": "响应格式。json_object = 强制 JSON 输出 (OpenAI 兼容有效; Anthropic 无原生 JSON 模式, 自动忽略; 国产兼容端口 xunfei/zhipu/moonshot 等常不支持, 可能触发 500)。留空 = 自由文本。",
    "seed": "随机种子，用于可复现实验。仅 OpenAI / Gemini 支持；Anthropic 无此字段。",
    "reasoning_mode": "推理强度: off / low / medium / high。Anthropic 翻译为 thinking.budget_tokens，OpenAI 翻译为 reasoning_effort。",
    "reasoning_budget_tokens": "Anthropic thinking 预算 token 数 (≥ 1024 且 < max_output_tokens)。只在 reasoning_mode != off 时生效。留空 = 按模式自动推算。",
    "timeout_seconds": "单次 HTTP 请求超时秒数。Self 建议 180, Hypothalamus 20。",
    "max_retries": "HTTP 失败时的最大重试次数。指数退避 + jitter。仅 5xx 和 429 会触发重试，4xx 不重试。",
    "retry_on_status": "触发重试的 HTTP 状态码列表。默认 [429, 500, 502, 503, 504]。",
}


@dataclass
class RoleBinding:
    provider: str = ""
    model: str = ""
    params: LLMParams = field(default_factory=LLMParams)


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
class GraphMemorySection:
    db_path: str = "workspace/data/graph_memory.sqlite"
    auto_ingest_similarity_threshold: float = 0.92
    # Top-K candidate nodes per stimulus during vec_search. A SEARCH
    # cap, not a prompt cap — the prompt cap is the per-role
    # `recall_token_budget` in LLMParams. Keeping this separate avoids
    # blowing compute on vec_search results we'd only truncate anyway.
    recall_per_stimulus_k: int = 5
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
    # `sliding_window` removed in the prompt-budget refactor — the
    # window's size is now derived from the Self role's LLMParams
    # (`max_input_tokens * history_token_fraction`). See
    # `_warn_about_removed_sections` in the loader for the deprecation
    # message on old configs.
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
    # Ordered list of Reflect plugin names to register at startup.
    # Order = execution order within a kind (chain semantics in
    # ``ReflectRegistry``). ``None`` is a migration sentinel: the
    # loader fell back to it because the YAML had no ``reflects:``
    # key, which means "old config not yet updated" — the runtime
    # then registers the legacy defaults + emits a deprecation
    # warning. An empty ``[]`` is the user's explicit choice for
    # zero Reflects (honored, no warning).
    # See docs/design/reflects-and-self-model.md for the full design.
    reflects: list[str] | None = None
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


def _build_llm_params(
    role_name: str, raw_params: dict[str, Any] | None,
) -> LLMParams:
    """Build LLMParams for `role_name`.

    Precedence (highest to lowest):
      1. User-supplied fields in config YAML (`raw_params`)
      2. Role-specific defaults from `_ROLE_DEFAULTS[role_name]`
      3. Universal defaults on the LLMParams dataclass itself

    Unknown keys in raw_params are ignored (forward compatibility — a
    future LLMParams field can appear in configs without errors).
    """
    # Normalize the legacy alias on the user's raw input first, BEFORE
    # merging with role defaults. Older configs may say `max_tokens`;
    # the current dataclass field is `max_output_tokens` (direction
    # made explicit). If both appear in the user's block, the explicit
    # new name wins and the alias is discarded.
    user: dict[str, Any] = dict(raw_params or {})
    if "max_tokens" in user:
        if "max_output_tokens" not in user:
            user["max_output_tokens"] = user["max_tokens"]
        user.pop("max_tokens")

    merged: dict[str, Any] = {}
    merged.update(_ROLE_DEFAULTS.get(role_name, {}))
    for k, v in user.items():
        merged[k] = v
    # Filter to known fields so unknown keys don't crash dataclass init.
    known = {f.name for f in fields(LLMParams)}
    safe = {k: v for k, v in merged.items() if k in known}
    # Light coercion: YAML lists for list fields, numeric strings for numbers.
    if "stop_sequences" in safe and safe["stop_sequences"] is not None:
        safe["stop_sequences"] = list(safe["stop_sequences"])
    if "retry_on_status" in safe and safe["retry_on_status"] is not None:
        safe["retry_on_status"] = [int(x) for x in safe["retry_on_status"]]
    return LLMParams(**safe)


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
        model = rdata.get("model", "")
        params = _build_llm_params(rname, rdata.get("params"))
        # Resolve max_input_tokens NOW (rather than lazily at call
        # time) so the runtime sees a concrete int everywhere and
        # budget math doesn't have to keep checking for None.
        # Order: YAML-pinned value > model_context lookup > default.
        if params.max_input_tokens is None:
            from src.utils.model_context import resolve_max_input_tokens
            params.max_input_tokens = resolve_max_input_tokens(model)
        roles[rname] = RoleBinding(
            provider=rdata.get("provider", ""),
            model=model,
            params=params,
        )
    return LLMSection(providers=providers, roles=roles)


# ---------------- schema introspection (for dashboard UI) ----------------


def llm_params_schema() -> list[dict[str, Any]]:
    """Return a list of field descriptors for LLMParams.

    Shape matches the per-plugin ``config_schema`` contract already
    consumed by the dashboard JS (`renderRow` + the plugin card
    renderer): each entry is ``{field, type, default, help}``.

    The dashboard fetches this via ``GET /api/config/schema`` and renders
    a dynamic "Params" sub-form under each LLM role so the UI stays in
    lockstep with the Python dataclass — adding a field to LLMParams
    automatically surfaces it in the UI without touching JavaScript.
    """
    out: list[dict[str, Any]] = []
    defaults = LLMParams()
    for f in fields(LLMParams):
        t = f.type
        # Normalize annotation to a UI type string.
        ui_type = "text"
        ann = t if isinstance(t, str) else getattr(t, "__name__", str(t))
        ann_lower = ann.lower()
        if "bool" in ann_lower:
            ui_type = "bool"
        elif "int" in ann_lower and "float" not in ann_lower:
            ui_type = "number"
        elif "float" in ann_lower:
            ui_type = "number_float"
        elif "list" in ann_lower:
            ui_type = "list"
        else:
            ui_type = "text"
        # Enum-like: reasoning_mode / response_format
        choices: list[str] | None = None
        if f.name == "reasoning_mode":
            choices = ["off", "low", "medium", "high"]
            ui_type = "enum"
        elif f.name == "response_format":
            choices = ["", "json_object"]
            ui_type = "enum"
        entry: dict[str, Any] = {
            "field": f.name,
            "type": ui_type,
            "default": getattr(defaults, f.name),
            "help": _LLM_PARAM_HELP.get(f.name, ""),
        }
        if choices is not None:
            entry["choices"] = choices
        out.append(entry)
    return out


def role_default_params(role_name: str) -> dict[str, Any]:
    """Expose the role-specific default overrides for the dashboard.

    The UI can use this to pre-fill the params form when the user first
    opens a role (so the Self role shows max_tokens=8192 etc. rather
    than the universal 4096). Returning a fresh dict each time so
    callers can safely mutate.
    """
    return dict(_ROLE_DEFAULTS.get(role_name, {}))


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

    _warn_about_removed_sections(raw)

    fatigue = _build_fatigue(raw.get("fatigue") or {})
    _validate_fatigue_thresholds(fatigue)

    return Config(
        llm=_build_llm(raw.get("llm") or {}),
        hibernate=_build_hibernate(raw.get("hibernate") or {}),
        fatigue=fatigue,
        graph_memory=_build_graph_memory(raw.get("graph_memory") or {}),
        knowledge_base=_build_kb(raw.get("knowledge_base") or {}),
        plugins=raw.get("plugins") or {},
        reflects=_build_reflects(raw),
        sleep=_build_sleep(raw.get("sleep") or {}),
        safety=_build_safety(raw.get("safety") or {}),
        dashboard=_build_dashboard(raw.get("dashboard")),
        sandbox=_build_sandbox(raw.get("sandbox")),
    )


def _build_reflects(raw: dict[str, Any]) -> list[str] | None:
    """Parse the ``reflects:`` field.

    Three states:
      * key absent       → return None (migration sentinel; runtime
                           falls back to legacy defaults + warns)
      * key empty list   → return [] (explicit "no reflects")
      * key string list  → return that list, in order

    Non-string entries are dropped with a warning so a typo or stray
    YAML mapping doesn't cause a registration crash later.
    """
    if "reflects" not in raw:
        return None
    val = raw.get("reflects")
    if val is None:
        # Explicit ``reflects: null`` is "I want no reflects" — same
        # as empty list. Treat both equally.
        return []
    if not isinstance(val, list):
        print(
            f"warning: `reflects:` should be a list, got "
            f"{type(val).__name__}; ignoring and falling back to "
            "legacy defaults.",
            file=sys.stderr,
        )
        return None
    cleaned: list[str] = []
    for item in val:
        if not isinstance(item, str) or not item.strip():
            print(
                f"warning: `reflects:` entry {item!r} is not a non-empty "
                "string; skipping.",
                file=sys.stderr,
            )
            continue
        cleaned.append(item.strip())
    return cleaned


def _warn_about_removed_sections(raw: dict[str, Any]) -> None:
    """Loud deprecation warnings for the prompt-budget refactor.

    Two fields were removed in favor of per-role LLMParams budgets:
      * ``sliding_window.max_tokens``     → derived from
        ``llm.roles.self.params.max_input_tokens *
        history_token_fraction``.
      * ``graph_memory.max_recall_nodes`` → replaced by
        ``llm.roles.self.params.recall_token_budget`` (absolute
        token cap, not a node count).

    We don't silently map them — the semantics changed (nodes → tokens;
    global → per-role) so silent mapping would produce surprising
    behavior. Users get one explicit stderr line per stale field.
    """
    sw = raw.get("sliding_window") or {}
    if isinstance(sw, dict) and "max_tokens" in sw:
        print(
            "deprecated: `sliding_window.max_tokens` is no longer used.\n"
            "  History budget is now derived from "
            "`llm.roles.self.params.max_input_tokens * "
            "history_token_fraction`. Remove the sliding_window section "
            "from your config. Your previous value is being ignored.",
            file=sys.stderr,
        )
    gm = raw.get("graph_memory") or {}
    if isinstance(gm, dict) and "max_recall_nodes" in gm:
        print(
            "deprecated: `graph_memory.max_recall_nodes` is no longer "
            "used.\n  Recall size is now capped by tokens via "
            "`llm.roles.self.params.recall_token_budget`. Remove the "
            "key from your config. Your previous value is being ignored.",
            file=sys.stderr,
        )


def _validate_fatigue_thresholds(f: FatigueSection) -> None:
    bad = [t for t in f.thresholds if t >= f.force_sleep_threshold]
    if bad:
        print(
            f"warning: fatigue threshold(s) {bad} >= force_sleep_threshold "
            f"({f.force_sleep_threshold}); force sleep will fire before hint shows.",
            file=sys.stderr,
        )
