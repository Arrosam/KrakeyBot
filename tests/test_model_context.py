"""Tests for the known-model context-window lookup
(src/utils/model_context.py)."""
from krakey.utils.model_context import (
    DEFAULT_CONTEXT_WINDOW, resolve_max_input_tokens,
)


def test_claude_sonnet_is_200k():
    assert resolve_max_input_tokens("claude-sonnet-4-5-20250101") == 200_000
    assert resolve_max_input_tokens("claude-sonnet-4") == 200_000


def test_gpt4o_is_128k():
    assert resolve_max_input_tokens("gpt-4o") == 128_000
    assert resolve_max_input_tokens("gpt-4o-mini") == 128_000


def test_deepseek_reasoner_is_64k():
    assert resolve_max_input_tokens("deepseek-reasoner") == 64_000


def test_deepseek_chat_is_128k():
    assert resolve_max_input_tokens("deepseek-chat") == 128_000


def test_gemini_15_pro_is_2m():
    assert resolve_max_input_tokens("gemini-1.5-pro-001") == 2_000_000


def test_unknown_model_falls_back_to_default():
    assert resolve_max_input_tokens("some-random-unknown-model") == (
        DEFAULT_CONTEXT_WINDOW
    )


def test_empty_model_returns_default():
    assert resolve_max_input_tokens("") == DEFAULT_CONTEXT_WINDOW
    assert resolve_max_input_tokens(None) == DEFAULT_CONTEXT_WINDOW


def test_case_insensitive_match():
    """Model names come from various provider catalogs with
    inconsistent casing; the resolver should match regardless."""
    assert resolve_max_input_tokens("CLAUDE-SONNET-4-5") == 200_000
    assert resolve_max_input_tokens("GPT-4o-Mini") == 128_000


def test_longest_prefix_wins():
    """`claude-sonnet-4` is more specific than `claude`, so a model
    that matches both keys must get the more specific entry's value.
    """
    # Both map to 200k so equality doesn't distinguish — but the
    # resolver contract says the more specific prefix wins. Verified
    # implicitly when future tables diverge.
    assert resolve_max_input_tokens("claude-sonnet-4-5") == 200_000
    assert resolve_max_input_tokens("claude-2") == 100_000
    # Generic "claude" fallback should give the 200k bucket, not
    # the 100k claude-2 bucket.
    assert resolve_max_input_tokens("claude-4-future-variant") == 200_000
