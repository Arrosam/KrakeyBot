"""Part A — SemanticAssociationEnricher contract tests.

Tests the public surface of:
    krakey.engines.recall._internal.enrich.SemanticAssociationEnricher

The class does NOT exist yet; these tests define acceptance criteria.

Techniques applied per method:
  enrich():
    - Positive / equivalence-partition  (6 tests)
    - Boundary value analysis           (8 tests)
    - State transitions                 (2 tests)
    - Negative / error-guessing         (9 tests)
"""
from __future__ import annotations

from datetime import datetime

import pytest

from krakey.engines.recall._internal.enrich import SemanticAssociationEnricher


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _ScriptedChatClient:
    """Duck-types ChatLike. Returns a fixed string on every call.
    Records all calls so tests can inspect the messages sent."""

    def __init__(self, response: str):
        self._response = response
        self.calls: list[list[dict]] = []

    async def chat(self, messages, **kwargs) -> str:
        self.calls.append(list(messages))
        return self._response


class _RaisingChatClient:
    """Duck-types ChatLike. Always raises on chat()."""

    def __init__(self, exc: Exception | None = None):
        self._exc = exc or RuntimeError("network error")
        self.calls: list[list[dict]] = []

    async def chat(self, messages, **kwargs) -> str:
        self.calls.append(list(messages))
        raise self._exc


class _ReturningNonStrClient:
    """Duck-types ChatLike. Returns None instead of a string."""

    async def chat(self, messages, **kwargs):
        return None  # type: ignore[return-value]


class _ReturningEmptyClient:
    """Duck-types ChatLike. Returns empty string."""

    async def chat(self, messages, **kwargs) -> str:
        return ""


class _ReturningWhitespaceClient:
    """Duck-types ChatLike. Returns whitespace-only string."""

    async def chat(self, messages, **kwargs) -> str:
        return "   \n\t  \n  "


_NOW = datetime(2026, 5, 17, 12, 0, 0)
_PAST = datetime(2024, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# Positive tests — valid inputs and expected outputs
# ---------------------------------------------------------------------------

class TestEnrichPositive:

    async def test_typical_text_returns_list_of_strings(self):
        """Standard text with a useful LLM response returns a non-empty list."""
        client = _ScriptedChatClient("Tom (key person)\nTravelling to London")
        e = SemanticAssociationEnricher(client)
        result = await e.enrich("Hello from Tom", now=_NOW)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(s, str) for s in result)

    async def test_worked_example_from_spec(self):
        """Spec's concrete worked example: 3-phrase LLM response parses to 3 items."""
        raw_response = (
            "Tom (key person)\n"
            "Travelling to London on 2026-05-18 (key event)\n"
            "Visiting Sam in London"
        )
        client = _ScriptedChatClient(raw_response)
        e = SemanticAssociationEnricher(client)
        text = 'A message from Tom: "I\'m heading to London tomorrow, going to meet Sam."'
        result = await e.enrich(text, now=datetime(2026, 5, 17, 0, 0, 0))
        assert result == [
            "Tom (key person)",
            "Travelling to London on 2026-05-18 (key event)",
            "Visiting Sam in London",
        ]

    async def test_now_isoformat_appears_in_prompt(self):
        """The caller-supplied now value's isoformat must appear in the
        user message sent to the client (determinism guarantee)."""
        fixed_now = datetime(2026, 5, 17, 8, 30, 0)
        client = _ScriptedChatClient("some phrase")
        e = SemanticAssociationEnricher(client)
        await e.enrich("test content", now=fixed_now)

        assert len(client.calls) == 1
        messages = client.calls[0]
        # Find the user message
        user_msg = next(m for m in messages if m["role"] == "user")
        assert fixed_now.isoformat() in user_msg["content"]

    async def test_raw_text_appears_in_user_message(self):
        """The raw input text must appear verbatim in the user message."""
        client = _ScriptedChatClient("phrase one")
        e = SemanticAssociationEnricher(client)
        raw_text = "Meeting at the office on Friday"
        await e.enrich(raw_text, now=_NOW)

        messages = client.calls[0]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert raw_text in user_msg["content"]

    async def test_messages_has_system_then_user_order(self):
        """Contract: messages is a 2-element list — system first, user second."""
        client = _ScriptedChatClient("phrase")
        e = SemanticAssociationEnricher(client)
        await e.enrich("some text", now=_NOW)

        messages = client.calls[0]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    async def test_past_datetime_appears_in_prompt(self):
        """A past datetime (not today) still propagates correctly to the prompt."""
        past = datetime(2020, 3, 15, 9, 0, 0)
        client = _ScriptedChatClient("result")
        e = SemanticAssociationEnricher(client)
        await e.enrich("old content", now=past)

        user_msg = next(m for m in client.calls[0] if m["role"] == "user")
        assert past.isoformat() in user_msg["content"]


# ---------------------------------------------------------------------------
# Boundary value analysis
# ---------------------------------------------------------------------------

class TestEnrichBVA:

    async def test_empty_string_returns_none_no_llm_call(self):
        """Empty string must return None without calling client.chat."""
        client = _ScriptedChatClient("phrase")
        e = SemanticAssociationEnricher(client)
        result = await e.enrich("", now=_NOW)
        assert result is None
        assert client.calls == []

    async def test_whitespace_only_returns_none_no_llm_call(self):
        """Whitespace-only string must return None without calling client.chat."""
        client = _ScriptedChatClient("phrase")
        e = SemanticAssociationEnricher(client)
        result = await e.enrich("   \t\n  ", now=_NOW)
        assert result is None
        assert client.calls == []

    async def test_single_char_text_calls_llm(self):
        """Single-character (non-whitespace) text is valid — LLM is called."""
        client = _ScriptedChatClient("one phrase")
        e = SemanticAssociationEnricher(client)
        result = await e.enrich("x", now=_NOW)
        assert len(client.calls) == 1
        assert result == ["one phrase"]

    async def test_single_line_response_returns_one_item(self):
        """LLM returning exactly one non-empty line → list of one string."""
        client = _ScriptedChatClient("single association")
        e = SemanticAssociationEnricher(client)
        result = await e.enrich("text", now=_NOW)
        assert result == ["single association"]

    async def test_response_with_empty_lines_drops_them(self):
        """Empty lines in LLM output must be dropped from result."""
        client = _ScriptedChatClient("phrase one\n\nphrase two\n\n")
        e = SemanticAssociationEnricher(client)
        result = await e.enrich("text", now=_NOW)
        assert result == ["phrase one", "phrase two"]

    async def test_response_lines_are_stripped(self):
        """Leading/trailing whitespace on each line is stripped."""
        client = _ScriptedChatClient("  phrase one  \n  phrase two  ")
        e = SemanticAssociationEnricher(client)
        result = await e.enrich("text", now=_NOW)
        assert result == ["phrase one", "phrase two"]

    async def test_response_with_only_code_fence_returns_none(self):
        """A response that is only a markdown code-fence line → None."""
        client = _ScriptedChatClient("```")
        e = SemanticAssociationEnricher(client)
        result = await e.enrich("text", now=_NOW)
        assert result is None

    async def test_response_with_code_fences_around_phrases_drops_fences(self):
        """Code-fence lines (```) are stripped; content lines are kept."""
        client = _ScriptedChatClient("```\nphrase one\nphrase two\n```")
        e = SemanticAssociationEnricher(client)
        result = await e.enrich("text", now=_NOW)
        assert result == ["phrase one", "phrase two"]

    async def test_large_response_all_lines_returned(self):
        """Many-line response returns all usable lines (no artificial cap)."""
        lines = [f"phrase {i}" for i in range(50)]
        client = _ScriptedChatClient("\n".join(lines))
        e = SemanticAssociationEnricher(client)
        result = await e.enrich("text", now=_NOW)
        assert result == lines


# ---------------------------------------------------------------------------
# State transition tests — non-idempotent call accumulation
# ---------------------------------------------------------------------------

class TestEnrichStateTransitions:

    async def test_multiple_calls_are_independent(self):
        """Each call to enrich() is independent — earlier results don't
        bleed into later ones. The enricher holds no mutable session state."""
        client1 = _ScriptedChatClient("result alpha")
        client2 = _ScriptedChatClient("result beta")
        e1 = SemanticAssociationEnricher(client1)
        e2 = SemanticAssociationEnricher(client2)

        r1 = await e1.enrich("text a", now=_NOW)
        r2 = await e2.enrich("text b", now=_NOW)

        assert r1 == ["result alpha"]
        assert r2 == ["result beta"]

    async def test_same_enricher_called_twice_makes_two_llm_calls(self):
        """Calling enrich() twice on the same enricher instance makes
        exactly two LLM calls — the method is not cached/memoized."""
        client = _ScriptedChatClient("phrase")
        e = SemanticAssociationEnricher(client)
        await e.enrich("first call", now=_NOW)
        await e.enrich("second call", now=_NOW)
        assert len(client.calls) == 2


# ---------------------------------------------------------------------------
# Negative tests — error-guessing
# ---------------------------------------------------------------------------

class TestEnrichNegative:

    async def test_chat_raises_returns_none(self):
        """Any exception from client.chat → None, never propagates."""
        client = _RaisingChatClient(RuntimeError("network error"))
        e = SemanticAssociationEnricher(client)
        result = await e.enrich("text", now=_NOW)
        assert result is None

    async def test_chat_raises_connection_error_returns_none(self):
        """ConnectionError specifically is caught and returns None."""
        client = _RaisingChatClient(ConnectionError("refused"))
        e = SemanticAssociationEnricher(client)
        result = await e.enrich("text", now=_NOW)
        assert result is None

    async def test_chat_raises_value_error_returns_none(self):
        """ValueError from the client is also caught; returns None."""
        client = _RaisingChatClient(ValueError("bad value"))
        e = SemanticAssociationEnricher(client)
        result = await e.enrich("text", now=_NOW)
        assert result is None

    async def test_chat_returns_none_gives_none(self):
        """client.chat() returning None (non-str) → enrich returns None."""
        client = _ReturningNonStrClient()
        e = SemanticAssociationEnricher(client)
        result = await e.enrich("text", now=_NOW)
        assert result is None

    async def test_chat_returns_empty_string_gives_none(self):
        """client.chat() returning empty string → enrich returns None."""
        client = _ReturningEmptyClient()
        e = SemanticAssociationEnricher(client)
        result = await e.enrich("text", now=_NOW)
        assert result is None

    async def test_chat_returns_whitespace_gives_none(self):
        """client.chat() returning whitespace-only string → enrich returns None."""
        client = _ReturningWhitespaceClient()
        e = SemanticAssociationEnricher(client)
        result = await e.enrich("text", now=_NOW)
        assert result is None

    async def test_response_only_json_brackets_returns_none(self):
        """Lines that are pure JSON bracket chars ({, }, [, ]) are dropped.
        If that leaves nothing, result is None."""
        client = _ScriptedChatClient("{\n}\n[\n]")
        e = SemanticAssociationEnricher(client)
        result = await e.enrich("text", now=_NOW)
        assert result is None

    async def test_response_mixed_fence_and_json_brackets_returns_none(self):
        """A response of only fences and JSON brackets → None after filtering."""
        client = _ScriptedChatClient("```\n{\n}\n```")
        e = SemanticAssociationEnricher(client)
        result = await e.enrich("text", now=_NOW)
        assert result is None

    async def test_enricher_does_not_call_datetime_now_internally(self):
        """The enricher must NOT call datetime.now() itself — determinism
        requirement. Verified indirectly: passing two different fixed
        datetimes to the same enricher produces two different prompts
        containing those exact timestamps."""
        client = _ScriptedChatClient("phrase")
        e = SemanticAssociationEnricher(client)
        dt1 = datetime(2020, 1, 1, 0, 0, 0)
        dt2 = datetime(2030, 12, 31, 23, 59, 59)

        await e.enrich("text", now=dt1)
        await e.enrich("text", now=dt2)

        user1 = next(m for m in client.calls[0] if m["role"] == "user")
        user2 = next(m for m in client.calls[1] if m["role"] == "user")
        assert dt1.isoformat() in user1["content"]
        assert dt2.isoformat() in user2["content"]
        # The two prompts must differ — if the enricher used datetime.now()
        # it could accidentally produce identical prompts here.
        assert user1["content"] != user2["content"]
