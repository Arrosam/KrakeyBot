"""Token estimation (DevSpec §10.3).

Single source of truth for "how many tokens is this string". Replaces
the `len(text) // 4` heuristic that used to live in
``sliding_window._approx_tokens``: that was wildly inaccurate for
CJK text (one ideograph ≈ 1-2 tokens, under-counted ~4-8×) which would
have poisoned any budget-based enforcement.

Backend:
  * Primary — `tiktoken` with the `cl100k_base` encoding. That's
    the GPT-4 / GPT-3.5 encoding, and Anthropic publicly recommends
    it as a good-enough approximation for Claude too (Anthropic's own
    tokenizer isn't pip-installable). Close enough that budget
    decisions don't swing on tokenizer choice.
  * Fallback — if tiktoken import or encoding-load somehow fails,
    drop to the legacy char/4 heuristic + log once so we don't crash
    a bot over a missing vocab file.

We do not bother with per-provider encodings (gpt-2 / p50k_base / etc.)
— the overhead of juggling encodings is not worth the marginal
accuracy, and cl100k is the common denominator the industry has
standardized on for this kind of back-of-envelope work.
"""
from __future__ import annotations

import logging
from functools import lru_cache

_log = logging.getLogger(__name__)

_FALLBACK_WARN_EMITTED = False


@lru_cache(maxsize=1)
def _encoding():
    """Return the cl100k_base tiktoken encoder, or None if unavailable.

    Cached so we don't re-load the 1MB vocab on every call. LRU of 1
    is effectively a module-level singleton with lazy init — cleaner
    than a module global that's mutated on first call.
    """
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:  # noqa: BLE001
        _log.warning(
            "tiktoken cl100k_base unavailable (%s); falling back to "
            "char/4 estimator. Budget enforcement will under-count "
            "non-ASCII text.", e,
        )
        return None


def estimate_tokens(text: str) -> int:
    """Return an integer token estimate for ``text``.

    Empty/None → 0. Never raises: a broken tokenizer drops to the
    char/4 fallback rather than poisoning a heartbeat.
    """
    if not text:
        return 0
    enc = _encoding()
    if enc is None:
        global _FALLBACK_WARN_EMITTED
        if not _FALLBACK_WARN_EMITTED:
            _log.warning("estimate_tokens: using char/4 fallback")
            _FALLBACK_WARN_EMITTED = True
        return max(1, len(text) // 4)
    try:
        return len(enc.encode(text))
    except Exception as e:  # noqa: BLE001 — tokenizer shouldn't crash a beat
        _log.warning("tiktoken.encode failed (%s); falling back", e)
        return max(1, len(text) // 4)


def estimate_tokens_many(texts: list[str]) -> int:
    """Sum ``estimate_tokens`` over a list. Convenience for callers
    that want the total of several fields (e.g. a SlidingWindowRound's
    stimulus_summary + decision_text + note_text) without writing the
    sum themselves.
    """
    return sum(estimate_tokens(t) for t in texts)
