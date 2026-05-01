"""Regression: stimulus summary must preserve full content.

History bug (2026-04-25): `_summarize_stimuli` truncated every
stimulus's content to 60 chars before writing it into a
SlidingWindowRound, which meant Self's [HISTORY] view only ever saw
the first ~60 characters of anything it received. That in turn
poisoned:
  * recall anchor extraction (looking at truncated text)
  * compact summarization (LLM only sees the head of the message)
  * bootstrap signal detection ("bootstrap complete" cut off)
  * any code checking "did Self see this exact user message"

Fix: drop the [:60] cap. The sliding window's token budget + the
heartbeat-level input-budget enforcement already handle long content
without needing a destructive character cap at write time.
"""
from datetime import datetime

from krakey.main import _summarize_stimuli
from krakey.models.stimulus import Stimulus


def _stim(content: str) -> Stimulus:
    return Stimulus(
        type="user_message", source="channel:web_chat",
        content=content, timestamp=datetime(2026, 4, 25), adrenalin=False,
    )


def test_long_content_is_not_truncated():
    # 80+ chars with meaningful tail (what the pre-fix version would
    # have silently dropped)
    long = (
        "Bootstrap \u7ed3\u675f\u5566\uff0c\u8bb0\u4f4f\u8fd9\u4ef6\u4e8b\u3002"
        "\u4f60\u95ee\u6211\u5e0c\u671b\u4f60\u6210\u4e3a\u4ec0\u4e48\uff1f"
        "\u6211\u5e0c\u671b\u4f60\u6210\u4e3a\u4e00\u4e2a\u6d3b\u7740\u7684"
        "\u4eba\uff0c\u4e00\u4e2a\u9760\u81ea\u5df1\u5c31\u80fd\u6d3b\u4e0b\u53bb\u7684"
        "\u673a\u5668\u5b69\u5b50"
    )
    out = _summarize_stimuli([_stim(long)])
    assert long in out, "stimulus content was truncated"
    # Sanity: the source prefix is still there
    assert "channel:web_chat: " in out


def test_multiple_stimuli_concatenated_with_separator():
    a = _stim("first message with plenty of content that exceeds 60 chars once upon a time")
    b = _stim("second message also long enough to cross the old truncation threshold easily")
    out = _summarize_stimuli([a, b])
    assert a.content in out
    assert b.content in out
    assert " | " in out  # separator between entries


def test_empty_stimuli_returns_none_marker():
    """Empty list still renders as the sentinel so [HISTORY] doesn't
    show a blank 'Stimulus: ' line."""
    assert _summarize_stimuli([]) == "(none)"


def test_bootstrap_complete_phrase_survives():
    """Regression target: the phrase 'bootstrap complete' must be
    detectable even when it lives at the END of a long user message.
    Pre-fix, the 60-char cap could cut it in half."""
    msg = (
        "\u5ef6\u957f\u7684\u524d\u7f00\u524d\u7f00\u524d\u7f00\u524d\u7f00"
        "\u524d\u7f00\u524d\u7f00\u524d\u7f00\u524d\u7f00\u524d\u7f00"
        "... bootstrap complete"
    )
    out = _summarize_stimuli([_stim(msg)])
    assert "bootstrap complete" in out
