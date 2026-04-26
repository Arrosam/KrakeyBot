"""Tests for the token-estimation utility (src/utils/tokens.py)."""
import src.utils.tokens as tokens_mod
from src.utils.tokens import estimate_tokens, estimate_tokens_many


def test_empty_string_is_zero():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0  # type: ignore[arg-type]


def test_short_ascii_has_reasonable_count():
    """'hello world' is 2 tokens under cl100k_base. Exact value is
    tokenizer-dependent so we check a loose range rather than a hard
    equal."""
    t = estimate_tokens("hello world")
    assert 1 <= t <= 4


def test_chinese_is_counted_realistically():
    """Chinese chars are 1-2 tokens each under cl100k_base — critical
    that the old `len//4` heuristic is gone, otherwise budget math
    under-counts Chinese contexts by 4-8\u00d7."""
    # 10 Chinese chars should be at least ~5 tokens (not 2.5 from char/4)
    text = "\u4f60\u597d\u4e16\u754c\u6211\u662f\u4e00\u4e2a\u6838\u5fc3"  # 10 chars
    t = estimate_tokens(text)
    assert t >= 5, f"expected \u22655 tokens for 10 Chinese chars, got {t}"


def test_estimate_many_sums():
    parts = ["hello", "world", "foo"]
    assert estimate_tokens_many(parts) == sum(
        estimate_tokens(p) for p in parts
    )
    assert estimate_tokens_many([]) == 0


def test_falls_back_gracefully_without_tokenizer(monkeypatch):
    """If tiktoken is somehow unavailable, estimate_tokens must NOT
    raise \u2014 it degrades to char/4 so a missing vocab never crashes
    a heartbeat."""
    # Clear the cached encoder and force the loader to return None.
    # monkeypatch auto-reverts the attribute at test end, so the
    # next test that wants the real tokenizer gets it back.
    tokens_mod._encoding.cache_clear()
    monkeypatch.setattr(tokens_mod, "_encoding", lambda: None)
    tokens_mod._FALLBACK_WARN_EMITTED = False
    t = estimate_tokens("hello world")
    # Fallback is char/4 of 11 chars \u2192 2 tokens
    assert t == max(1, len("hello world") // 4)
