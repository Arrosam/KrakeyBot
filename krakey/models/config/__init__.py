"""Config loader for config.yaml (DevSpec §15).

Parses YAML, substitutes ``${VAR}`` from os.environ at load time,
validates fatigue thresholds vs force_sleep_threshold.

If the target config file is missing, ``load_config`` raises
``FileNotFoundError`` with a hint pointing the user at the standalone
onboarding wizard (``python -m src.onboarding``). No silent
auto-generation — fresh installs are expected to run the wizard so
they get providers + tags + plugin choices wired up before the first
runtime boot.

Submodule layout (split out of the original 872-line monolith):

  * ``llm``       — providers / tags / core_purposes / embedding /
                    reranker. Largest section by far.
  * ``heartbeat`` — idle cadence + fatigue thresholds.
  * ``memory``    — graph_memory + knowledge_base + sleep + safety.
  * ``infra``     — dashboard + sandbox.

Everything users import from ``src.models.config`` is re-exported
here so existing call sites keep working unchanged.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Re-exports: every dataclass + builder the rest of the codebase uses.
from krakey.models.config.heartbeat import (  # noqa: F401
    FatigueSection,
    IdleSection,
    SlidingWindowSection,
    _build_fatigue,
    _build_idle,
    _build_sliding_window,
    _validate_fatigue_thresholds,
)
from krakey.models.config.environments import (  # noqa: F401
    EnvironmentsSection,
    LocalEnvironmentConfig,
    SandboxEnvironmentConfig,
    _build_environments,
)
from krakey.models.config.infra import (  # noqa: F401
    SandboxAgentSection,
    SandboxResourcesSection,
)
from krakey.models.config.llm import (  # noqa: F401
    LLMParams,
    LLMSection,
    ModelEntry,
    Provider,
    TagBinding,
    _build_llm,
    _build_llm_params_for_tag,
    llm_params_schema,
)
from krakey.models.config.memory import (  # noqa: F401
    GraphMemorySection,
    KnowledgeBaseSection,
    SafetySection,
    SleepSection,
    _build_graph_memory,
    _build_kb,
    _build_safety,
    _build_sleep,
)
from krakey.models.config.core_impls import (  # noqa: F401
    CoreImplementations,
    _build_core_implementations,
)


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


# ---------------- top-level Config dataclass ----------------


@dataclass
class Config:
    llm: LLMSection = field(default_factory=LLMSection)
    idle: IdleSection = field(default_factory=IdleSection)
    fatigue: FatigueSection = field(default_factory=FatigueSection)
    # The sliding window's SIZE budget is derived from the Self
    # role's LLMParams (max_input_tokens * history_token_fraction)
    # — that's why `sliding_window.max_tokens` is not a config
    # field. ``sliding_window`` here only carries persistence
    # settings (state_path) — see SlidingWindowSection.
    sliding_window: SlidingWindowSection = field(
        default_factory=SlidingWindowSection
    )
    graph_memory: GraphMemorySection = field(
        default_factory=GraphMemorySection
    )
    knowledge_base: KnowledgeBaseSection = field(
        default_factory=KnowledgeBaseSection
    )
    # Ordered list of unified-format plugin names to enable at
    # startup. A plugin can declare any mix of modifier / tool /
    # channel components in its meta.yaml (Samuel 2026-04-26
    # unification). Order = chain execution order for same-kind
    # modifier components.
    #
    # ``None`` (field absent) and ``[]`` are both honored as "zero
    # plugins" — the runtime never enables anything implicitly
    # (strictly additive). ``None`` triggers a one-line stderr
    # nudge so users notice; ``[]`` is silent.
    plugins: list[str] | None = None
    sleep: SleepSection = field(default_factory=SleepSection)
    safety: SafetySection = field(default_factory=SafetySection)
    environments: EnvironmentsSection = field(
        default_factory=EnvironmentsSection
    )
    # Optional dotted-path overrides for built-in core services
    # (memory, prompt builder, embedder, ...). See
    # krakey/models/config/core_impls.py + krakey/runtime/service_resolver.py.
    # Empty fields → built-in defaults; non-empty → import + Protocol-validate
    # the user's class at startup.
    core_implementations: CoreImplementations = field(
        default_factory=CoreImplementations
    )
    # Per-engine user config keyed by ``(slot, short_name)``. Each
    # selected engine's dict is passed to its constructor as a
    # ``config`` kwarg (engines that don't take one ignore it via
    # ``EngineRegistry._filter_kwargs``). Schema for each engine
    # comes from ``EngineImpl.config_schema`` in the slot's
    # ``engines/<slot>/meta.yaml`` (or, for plugin engines, from the
    # plugin's top-level ``config_schema:``). Dashboard renders the
    # form under the slot's dropdown when a schema-bearing impl is
    # selected.
    engine_configs: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=dict,
    )


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


# ---------------- dump / ensure ----------------


def dump_config(cfg: Config) -> str:
    """Serialize a Config dataclass to the YAML text we'd write to disk.

    Round-trips cleanly through load_config: ``dump_config(Config())`` is
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
    """Create a defaults-populated config at ``path`` if it does not exist.

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
    """SystemExit subclass raised on unrecoverable config-shape errors
    (currently: detected the deprecated ``llm.roles:`` schema). Lets
    tests distinguish a config-bootstrap exit from unrelated exits."""
    pass


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load + parse ``config.yaml``.

    Raises ``FileNotFoundError`` if the file is missing. The CLI's
    ``run`` / ``start`` commands catch this and auto-launch the
    onboarding wizard rather than surfacing the error to the user.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"config not found at {p} — run `krakey onboard` to "
            "generate one (the wizard walks you through providers, "
            "tags, and plugin selection)."
        )

    raw_text = p.read_text(encoding="utf-8")
    raw: dict[str, Any] = yaml.safe_load(raw_text) or {}
    raw = _substitute_env(raw)

    _warn_about_removed_sections(raw)

    fatigue = _build_fatigue(raw.get("fatigue") or {})
    _validate_fatigue_thresholds(fatigue)

    return Config(
        llm=_build_llm(raw.get("llm") or {}),
        idle=_build_idle(raw.get("idle") or {}),
        fatigue=fatigue,
        sliding_window=_build_sliding_window(
            raw.get("sliding_window") or {}
        ),
        graph_memory=_build_graph_memory(raw.get("graph_memory") or {}),
        knowledge_base=_build_kb(raw.get("knowledge_base") or {}),
        plugins=_build_plugins(raw),
        sleep=_build_sleep(raw.get("sleep") or {}),
        safety=_build_safety(raw.get("safety") or {}),
        environments=_build_environments(raw.get("environments")),
        core_implementations=_build_core_implementations(
            raw.get("core_implementations") or {}
        ),
        engine_configs=_build_engine_configs(
            raw.get("engine_configs") or {}
        ),
    )


def _build_engine_configs(
    raw: Any,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Parse the ``engine_configs:`` block — ``{slot: {short_name:
    {field: value}}}``. Tolerates a missing block or wrong-shape
    nesting; the registry only ever consults the leaf dict so a
    ragged structure just degrades to "no config" silently. Strict
    schema validation against each engine's ``config_schema`` is
    handled at engine-construction time, not here."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for slot, slot_cfg in raw.items():
        if not isinstance(slot_cfg, dict):
            continue
        slot_out: dict[str, dict[str, Any]] = {}
        for short_name, impl_cfg in slot_cfg.items():
            if isinstance(impl_cfg, dict):
                slot_out[str(short_name)] = dict(impl_cfg)
        out[str(slot)] = slot_out
    return out


def _build_plugins(raw: dict[str, Any]) -> list[str] | None:
    """Parse the ``plugins:`` + ``modifiers:`` fields into a single
    ordered enable-list.

    The dashboard intentionally maintains TWO lists in central
    config.yaml:
      * ``modifiers:`` — ordered list of plugin names that contribute
        a modifier component. The order is the heartbeat-chain order
        (each Modifier runs in turn each beat).
      * ``plugins:``   — set-style list of plugin names that contribute
        a tool or channel component.

    A plugin with both kinds shows up in both lists; a modifier-only
    plugin shows up ONLY in ``modifiers:``. The loader merges both
    lists so modifier-only plugins still register.

    Three states:
      * neither field present → return None (one-line stderr nudge at
                                runtime so users notice they have no
                                plugins enabled)
      * both fields empty     → return [] (explicit "zero plugins")
      * any names present     → merged + de-duplicated list, modifier
                                chain order first

    Migration: a config with ONLY ``modifiers:`` (no ``plugins:``) is a
    pre-2026-04-26 layout. We still load it but emit a one-time
    deprecation note so the user knows to add the ``plugins:`` key.
    """
    has_plugins   = "plugins"   in raw
    has_modifiers = "modifiers" in raw

    if not has_plugins and not has_modifiers:
        return None

    if has_modifiers and not has_plugins:
        print(
            "config: only `modifiers:` is set — pre-2026-04-26 layout. "
            "Treating it as the active plugin list. Add an empty "
            "`plugins: []` (or a list of tool/channel plugin names) to "
            "silence this notice; the loader merges both fields.",
            file=sys.stderr,
        )

    plugins_list   = _coerce_name_list(raw.get("plugins"),   "plugins")
    modifiers_list = _coerce_name_list(raw.get("modifiers"), "modifiers")

    # Modifier chain order first (the chain is order-sensitive), then
    # any service plugins not already covered. De-dup preserves the
    # earlier slot — so a plugin in both lists keeps its modifiers-list
    # position in the chain.
    merged: list[str] = []
    seen: set[str] = set()
    for name in modifiers_list + plugins_list:
        if name in seen:
            continue
        seen.add(name)
        merged.append(name)
    return merged


def _coerce_name_list(val: Any, field_name: str) -> list[str]:
    """Validate one of the plugin enable-lists. Drops non-string /
    empty entries with a per-entry warning so a typo doesn't silently
    disable the rest of the list."""
    if val is None:
        return []
    if not isinstance(val, list):
        print(
            f"warning: `{field_name}:` should be a list, got "
            f"{type(val).__name__}; treating as empty.",
            file=sys.stderr,
        )
        return []
    cleaned: list[str] = []
    for item in val:
        if not isinstance(item, str) or not item.strip():
            print(
                f"warning: `{field_name}:` entry {item!r} is not a "
                "non-empty string; skipping.",
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
        # Note (2026-05-07): `sliding_window` is back as a section
        # (carrying state_path for persistence), but `max_tokens`
        # specifically is still derived, never user-configured.
        print(
            "deprecated: `sliding_window.max_tokens` is no longer used.\n"
            "  History budget is derived from "
            "`llm.tags.<self_tag>.params.max_input_tokens * "
            "history_token_fraction`. Remove just the `max_tokens` "
            "key — `sliding_window.state_path` is still honored.",
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
    if "dashboard" in raw:
        print(
            "deprecated: top-level `dashboard:` section is no longer "
            "read.\n  The dashboard is now a regular plugin: add "
            "'dashboard' to your `plugins:` list, and put host/port/"
            "history_path in `workspace/plugins/dashboard/config.yaml`.\n"
            "  Your previous values are being ignored.",
            file=sys.stderr,
        )
    if "sandbox" in raw:
        print(
            "deprecated: top-level `sandbox:` section is no longer "
            "read.\n  Sandbox is now an Environment under the new "
            "top-level `environments:` block. Move your guest_os / "
            "agent.url / agent.token (and any other fields) under "
            "`environments.sandbox`, and add an `allowed_plugins:` "
            "list naming each plugin that may dispatch through it.\n"
            "  Your previous values are being ignored.",
            file=sys.stderr,
        )
