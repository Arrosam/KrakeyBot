"""Model → input-context-window lookup (DevSpec §14.2).

Resolves `max_input_tokens` for a role when the user hasn't pinned it
in YAML. Precedence handled by the caller:

    user YAML value  >  `resolve_max_input_tokens(model)`  >  DEFAULT

We intentionally do NOT perform a runtime HTTP probe of the provider:
Anthropic's REST API has no `/v1/models` endpoint that reports context
window, OpenAI's `/v1/models` omits the field, Gemini's metadata API
needs auth. A static lookup keeps startup fast + offline-safe; the
cost is that when a provider ships a new model we need to add a row
here. That's a trade we're happy with for a solo-dev project.

Matching is **longest-prefix**: `claude-sonnet-4-5-20250101` matches
`claude-sonnet-4`. Callers hit `resolve_max_input_tokens(model_name)`
and get an int — never None, never an exception. Unknown models get
`DEFAULT_CONTEXT_WINDOW` with a single INFO log so we don't silently
misconfigure without a breadcrumb.
"""
from __future__ import annotations

import logging
from functools import lru_cache

_log = logging.getLogger(__name__)


DEFAULT_CONTEXT_WINDOW = 128_000  # Safe middle-ground default (2026).


# Longest-prefix table. Keys are case-insensitive model-name prefixes;
# order within the dict is irrelevant — the resolver sorts by key
# length descending so `claude-sonnet-4` beats `claude`.
#
# Last updated: 2026-04. Keep in sync with provider docs when you
# notice a surprise — this table is load-bearing for budget math.
_CONTEXT_WINDOW_LOOKUP: dict[str, int] = {
    # Anthropic (Claude) — all current models are 200K context
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "claude-haiku-4": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-2": 100_000,
    "claude": 200_000,  # generic fallback for claude-*

    # OpenAI
    "gpt-5": 400_000,
    "gpt-4.1": 1_000_000,
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 128_000,
    "gpt-3.5": 16_000,
    "o1-mini": 128_000,
    "o1": 200_000,
    "o3-mini": 200_000,
    "o3": 200_000,
    "o4-mini": 200_000,
    "o4": 200_000,

    # DeepSeek
    "deepseek-reasoner": 64_000,
    "deepseek-chat": 128_000,
    "deepseek-v3": 128_000,
    "deepseek-r1": 64_000,
    "deepseek": 128_000,

    # Google Gemini
    "gemini-3": 2_000_000,
    "gemini-2": 2_000_000,
    "gemini-1.5-pro": 2_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-1.5": 1_000_000,
    "gemini-pro": 128_000,
    "gemini": 1_000_000,

    # Alibaba Qwen
    "qwen2.5": 128_000,
    "qwen2": 128_000,
    "qwen-max": 32_000,
    "qwen": 32_000,

    # Moonshot / Kimi
    "kimi-k2": 256_000,
    "kimi": 200_000,
    "moonshot-v1-128k": 128_000,
    "moonshot-v1-32k": 32_000,
    "moonshot-v1-8k": 8_000,
    "moonshot": 128_000,

    # Zhipu GLM
    "glm-4.6": 128_000,
    "glm-4.5": 128_000,
    "glm-4": 128_000,
    "glm": 128_000,

    # Xunfei Spark
    "spark-max": 128_000,
    "spark-pro": 128_000,
    "spark": 32_000,

    # Meta Llama
    "llama-4": 1_000_000,
    "llama-3.3": 128_000,
    "llama-3.1": 128_000,
    "llama-3": 8_000,
    "llama": 8_000,

    # Mistral
    "mistral-large": 128_000,
    "mistral-medium": 32_000,
    "mistral-small": 32_000,
    "mistral": 32_000,
    "mixtral": 32_000,
}


@lru_cache(maxsize=256)
def _sorted_keys() -> tuple[str, ...]:
    """Return prefix keys sorted by length descending so longest-prefix
    match wins. Cached — the table is effectively immutable at runtime.
    """
    return tuple(sorted(_CONTEXT_WINDOW_LOOKUP.keys(),
                         key=len, reverse=True))


def resolve_max_input_tokens(model: str | None) -> int:
    """Return the declared input-context window for ``model``.

    Returns ``DEFAULT_CONTEXT_WINDOW`` (128K) when:
      * ``model`` is empty / None
      * no prefix in ``_CONTEXT_WINDOW_LOOKUP`` matches

    Unknown-model case emits a single INFO-level log so the user has
    a breadcrumb to add an entry here if they care about precision.
    """
    if not model:
        return DEFAULT_CONTEXT_WINDOW
    needle = model.lower()
    for prefix in _sorted_keys():
        if needle.startswith(prefix):
            return _CONTEXT_WINDOW_LOOKUP[prefix]
    _log.info(
        "model %r not in context-window lookup; using default %d. "
        "Add to src/utils/model_context.py if precision matters.",
        model, DEFAULT_CONTEXT_WINDOW,
    )
    return DEFAULT_CONTEXT_WINDOW
