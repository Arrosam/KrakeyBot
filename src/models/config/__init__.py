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
  * ``heartbeat`` — hibernate cadence + fatigue thresholds.
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
from src.models.config.heartbeat import (  # noqa: F401
    FatigueSection,
    HibernateSection,
    _build_fatigue,
    _build_hibernate,
    _validate_fatigue_thresholds,
)
from src.models.config.infra import (  # noqa: F401
    SandboxAgentSection,
    SandboxResourcesSection,
    SandboxSection,
    _build_sandbox,
)
from src.models.config.llm import (  # noqa: F401
    LLMParams,
    LLMSection,
    ModelEntry,
    Provider,
    TagBinding,
    _build_llm,
    _build_llm_params_for_tag,
    llm_params_schema,
)
from src.models.config.memory import (  # noqa: F401
    GraphMemorySection,
    KnowledgeBaseSection,
    SafetySection,
    SleepSection,
    _build_graph_memory,
    _build_kb,
    _build_safety,
    _build_sleep,
)


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


# ---------------- top-level Config dataclass ----------------


@dataclass
class Config:
    llm: LLMSection = field(default_factory=LLMSection)
    hibernate: HibernateSection = field(default_factory=HibernateSection)
    fatigue: FatigueSection = field(default_factory=FatigueSection)
    # `sliding_window` removed in the prompt-budget refactor — the
    # window's size is now derived from the Self role's LLMParams
    # (`max_input_tokens * history_token_fraction`). See
    # `_warn_about_removed_sections` below for the deprecation
    # message on old configs.
    graph_memory: GraphMemorySection = field(
        default_factory=GraphMemorySection
    )
    knowledge_base: KnowledgeBaseSection = field(
        default_factory=KnowledgeBaseSection
    )
    # Ordered list of unified-format plugin names to enable at
    # startup. A plugin can declare any mix of reflect / tentacle /
    # sensory components in its meta.yaml (Samuel 2026-04-26
    # unification). Order = chain execution order for same-kind
    # reflect components.
    #
    # ``None`` (field absent) and ``[]`` are both honored as "zero
    # plugins" — the runtime never enables anything implicitly
    # (strictly additive). ``None`` triggers a one-line stderr
    # nudge so users notice; ``[]`` is silent.
    plugins: list[str] | None = None
    sleep: SleepSection = field(default_factory=SleepSection)
    safety: SafetySection = field(default_factory=SafetySection)
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

    Raises ``FileNotFoundError`` if the file is missing — fresh
    installs are expected to run ``python -m src.onboarding`` first.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"config not found at {p} — run `python -m src.onboarding` "
            "to generate one (the wizard walks you through providers, "
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
        hibernate=_build_hibernate(raw.get("hibernate") or {}),
        fatigue=fatigue,
        graph_memory=_build_graph_memory(raw.get("graph_memory") or {}),
        knowledge_base=_build_kb(raw.get("knowledge_base") or {}),
        plugins=_build_plugins(raw),
        sleep=_build_sleep(raw.get("sleep") or {}),
        safety=_build_safety(raw.get("safety") or {}),
        sandbox=_build_sandbox(raw.get("sandbox")),
    )


def _build_plugins(raw: dict[str, Any]) -> list[str] | None:
    """Parse the ``plugins:`` field.

    Three states:
      * key absent       → return None (one-line stderr nudge at
                           runtime so users notice they have no
                           plugins enabled)
      * key empty list   → return [] (explicit "zero plugins")
      * key string list  → return that list, in order

    Migration: the OLD ``reflects:`` field is silently translated to
    ``plugins:`` for one release window so users mid-migration still
    boot. Non-string entries are dropped with a warning.
    """
    # Old `reflects:` field migration alias
    if "plugins" not in raw and "reflects" in raw:
        print(
            "config: `reflects:` is deprecated — renamed to `plugins:` "
            "in the unified plugin refactor (2026-04-26). Treating your "
            "`reflects:` list as the plugin list this run; please rename "
            "the key to silence this warning.",
            file=sys.stderr,
        )
        raw = {**raw, "plugins": raw["reflects"]}

    if "plugins" not in raw:
        return None
    val = raw.get("plugins")
    if val is None:
        return []
    if not isinstance(val, list):
        print(
            f"warning: `plugins:` should be a list, got "
            f"{type(val).__name__}; treating as empty.",
            file=sys.stderr,
        )
        return []
    cleaned: list[str] = []
    for item in val:
        if not isinstance(item, str) or not item.strip():
            print(
                f"warning: `plugins:` entry {item!r} is not a non-empty "
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
    if "dashboard" in raw:
        print(
            "deprecated: top-level `dashboard:` section is no longer "
            "read.\n  The dashboard is now a regular plugin: add "
            "'dashboard' to your `plugins:` list, and put host/port/"
            "history_path in `workspace/plugins/dashboard/config.yaml`.\n"
            "  Your previous values are being ignored.",
            file=sys.stderr,
        )
