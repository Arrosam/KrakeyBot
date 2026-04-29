"""LLM section of config.yaml — providers, tags, core_purposes,
embedding/reranker slots.

Three layers of indirection:

  1. ``providers`` — physical API connections (URL + key). Sensitive;
     only the runtime ever sees these. Plugins **never** receive
     provider config.
  2. ``tags`` — user-named pools each binding a single
     ``(provider, model, params)`` triple. Tags are how users give
     human-readable names to model choices and reuse one choice
     across many purposes.
  3. ``core_purposes`` — runtime-internal use cases (Self's thinking,
     compaction, ...) mapped to a tag name. Plugins have their own
     purposes declared in their own per-folder ``meta.yaml`` and
     bound in ``workspace/plugins/<name>/config.yaml`` — those live
     OUTSIDE this section so plugin code can't reach into central
     config and read API keys.

Embedding + reranker are NOT purposes (they're capabilities intrinsic
to specific models), so they get dedicated fields holding a tag name
directly.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from typing import Any


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
      * ``max_output_tokens`` — upper bound on generation. Translated
        to Anthropic ``max_tokens``, OpenAI classic ``max_tokens``,
        OpenAI reasoning ``max_completion_tokens``, Gemini
        ``maxOutputTokens``.
      * ``max_input_tokens`` — input-context-window budget for the
        backing model. If left None at load time, the config loader
        resolves it via ``src.utils.model_context.resolve_max_input_tokens``
        (known-prefix lookup, default 128_000). This is **active**:
        used by the runtime for sliding-window history budget, recall
        budget, and overall-prompt enforcement.

    Prompt-budget allocation (only meaningful for Self's role, since
    Self is the only consumer of the sliding window + GM recall):
      * ``history_token_fraction`` — fraction of ``max_input_tokens``
        reserved for [HISTORY]. Default 0.4. When the window's token
        total exceeds this fraction the compactor pops oldest rounds
        into GM.
      * ``recall_token_budget`` — ABSOLUTE token cap for the
        [GRAPH MEMORY] section (not a fraction). Too many recall nodes
        pollute context with marginal relevance, so this scales poorly
        with bigger context — a model with 2M tokens doesn't want 500
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


# Human-readable descriptions used both for docstrings and for the
# ``/api/config/schema`` endpoint that feeds the dashboard UI. Keep
# this in sync with LLMParams fields above.
_LLM_PARAM_HELP: dict[str, str] = {
    "max_output_tokens": "Output (generation) token cap. Translated per provider: Anthropic max_tokens, OpenAI classic max_tokens, OpenAI reasoning max_completion_tokens, Gemini maxOutputTokens. Required for Anthropic.",
    "max_input_tokens": "Input (prompt) context token budget. If empty, resolved at startup by model-name lookup (unknown models default to 128000). Drives sliding-window compaction threshold, GM recall budget, and oversized-prompt history trimming.",
    "history_token_fraction": "Self role only. Fraction of max_input_tokens reserved for the [HISTORY] layer. Default 0.4 = 40%. When the window exceeds this fraction, compact pops the oldest round into GM.",
    "recall_token_budget": "Self role only. Absolute (not fractional) token budget for the [GRAPH MEMORY] recall section. Default 3000. Too many recall items pollute context, so this does not scale linearly with context size.",
    "temperature": "Sampling temperature. 0 = deterministic; higher = more diverse. Some reasoning models (OpenAI o-series, DeepSeek Reasoner) do not support it and the field is silently dropped.",
    "top_p": "Nucleus-sampling threshold (0-1). Usually paired with OR temperature, not both. Empty = do not send this field.",
    "stop_sequences": "Stop-sequence list. Generation halts on the first match.",
    "response_format": "Response format. json_object = force JSON output (effective on OpenAI-compatible; Anthropic has no native JSON mode and ignores it; some Chinese-vendor compatibility ports such as xunfei/zhipu/moonshot reject it and may 500). Empty = free text.",
    "seed": "Random seed for reproducible runs. Supported by OpenAI / Gemini only; Anthropic has no such field.",
    "reasoning_mode": "Reasoning intensity: off / low / medium / high. Translated to Anthropic thinking.budget_tokens or OpenAI reasoning_effort.",
    "reasoning_budget_tokens": "Anthropic thinking-budget tokens (≥ 1024 and < max_output_tokens). Active only when reasoning_mode != off. Empty = auto-derived from mode.",
    "timeout_seconds": "Per-request HTTP timeout in seconds. Suggested: 180 for Self, 20 for Hypothalamus.",
    "max_retries": "Max retries on HTTP failure. Exponential backoff + jitter. Only 5xx and 429 retry; 4xx does not.",
    "retry_on_status": "List of HTTP status codes that trigger a retry. Default [429, 500, 502, 503, 504].",
}


@dataclass
class TagBinding:
    """A named tag mapping a logical purpose to a concrete model.

    ``provider`` is a combined ``"<provider_name>/<model_name>"`` string
    so YAML stays compact. Splits on the FIRST ``/`` — provider names
    must not contain ``/``, but model names may (e.g. ``BAAI/bge-m3``).
    """
    provider: str = ""   # e.g. "One API/qwen3.6-9b"
    params: LLMParams = field(default_factory=LLMParams)

    def split_provider(self) -> tuple[str, str]:
        """Return ``(provider_name, model_name)``. Raises ``ValueError``
        when the value lacks a ``/`` separator."""
        provider, sep, model = self.provider.partition("/")
        if not sep:
            raise ValueError(
                f"tag provider must be '<provider>/<model>'; got "
                f"{self.provider!r}"
            )
        return provider, model


@dataclass
class LLMSection:
    providers: dict[str, Provider] = field(default_factory=dict)
    tags: dict[str, TagBinding] = field(default_factory=dict)
    # Core (runtime-owned, non-plugin) chat purpose → tag name
    core_purposes: dict[str, str] = field(default_factory=dict)
    # Special model-type slots
    embedding: str | None = None  # name of an embedding-capable tag
    reranker: str | None = None   # name of a rerank-capable tag (optional)

    # ---- helpers ---------------------------------------------------

    def tag(self, name: str) -> TagBinding | None:
        return self.tags.get(name)

    def core_tag(self, purpose: str) -> TagBinding | None:
        tag_name = self.core_purposes.get(purpose)
        if tag_name is None:
            return None
        return self.tags.get(tag_name)

    def core_params(self, purpose: str) -> LLMParams | None:
        """Convenience for callers that only care about the params
        of a core purpose (e.g. Self's max_input_tokens)."""
        tag = self.core_tag(purpose)
        return tag.params if tag is not None else None


# ---- builder ----------------------------------------------------------


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
    # Detect old `roles:` shape — Samuel's tag-based refactor 2026-04-26
    # removed it. Legacy configs must migrate; we exit loud rather than
    # silently doing the wrong thing.
    if "roles" in raw:
        _raise_old_roles_migration_error()

    tags: dict[str, TagBinding] = {}
    for tag_name, tdata in (raw.get("tags") or {}).items():
        tdata = tdata or {}
        params = _build_llm_params_for_tag(tdata.get("params"))
        # Resolve max_input_tokens NOW from model name (split out of
        # the combined "provider/model" field) so the runtime sees a
        # concrete int everywhere.
        if params.max_input_tokens is None:
            from krakey.utils.model_context import resolve_max_input_tokens
            provider_field = str(tdata.get("provider", ""))
            _, _, model_name = provider_field.partition("/")
            params.max_input_tokens = resolve_max_input_tokens(model_name)
        tags[tag_name] = TagBinding(
            provider=str(tdata.get("provider", "")),
            params=params,
        )
    core_purposes: dict[str, str] = {}
    for purpose, tag_name in (raw.get("core_purposes") or {}).items():
        if not isinstance(tag_name, str):
            continue
        core_purposes[purpose] = tag_name
    embedding = raw.get("embedding")
    if embedding is not None and not isinstance(embedding, str):
        embedding = None
    reranker = raw.get("reranker")
    if reranker is not None and not isinstance(reranker, str):
        reranker = None
    return LLMSection(
        providers=providers, tags=tags,
        core_purposes=core_purposes,
        embedding=embedding, reranker=reranker,
    )


def _build_llm_params_for_tag(
    raw_params: dict[str, Any] | None,
) -> LLMParams:
    """Build an LLMParams with no per-purpose default injection — every
    tag stands on its own; users specify the params they want."""
    user: dict[str, Any] = dict(raw_params or {})
    # Legacy alias: older configs may say `max_tokens`. Translate
    # silently — same behavior as the previous _build_llm_params.
    if "max_tokens" in user:
        if "max_output_tokens" not in user:
            user["max_output_tokens"] = user["max_tokens"]
        user.pop("max_tokens")
    known = {f.name for f in fields(LLMParams)}
    safe = {k: v for k, v in user.items() if k in known}
    return LLMParams(**safe)


def _raise_old_roles_migration_error() -> None:
    """Loud failure for users on the pre-2026-04-26 ``llm.roles:`` shape.
    We exit rather than silently mis-parsing.
    """
    msg = (
        "config.yaml uses the deprecated `llm.roles:` shape that was\n"
        "removed in the tag-based LLM refactor (2026-04-26). Migrate by:\n\n"
        "  OLD:\n"
        "    llm:\n"
        "      roles:\n"
        "        self:\n"
        "          provider: \"One API\"\n"
        "          model: \"astron\"\n"
        "          params: {...}\n\n"
        "  NEW:\n"
        "    llm:\n"
        "      tags:\n"
        "        my_self_tag:\n"
        "          provider: \"One API/astron\"   # combined provider/model\n"
        "          params: {...}\n"
        "      core_purposes:\n"
        "        self_thinking: my_self_tag\n"
        "        compact: my_compact_tag\n"
        "        classifier: my_compact_tag\n"
        "      embedding: my_embed_tag\n"
        "      reranker: my_rerank_tag    # optional\n\n"
        "  Per-plugin (workspace/plugins/<name>/config.yaml):\n"
        "    llm_purposes:\n"
        "      <purpose_name>: <tag_name>\n\n"
        "See docs/design/modifiers-and-self-model.md for the full spec.\n"
        "Krakey will not start until this migrates."
    )
    print(msg, file=sys.stderr)
    # Lazy import to avoid circular dep with the package __init__.
    from krakey.models.config import _ConfigBootstrapExit
    raise _ConfigBootstrapExit(2)


# ---- schema introspection (for dashboard UI) ------------------------


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
