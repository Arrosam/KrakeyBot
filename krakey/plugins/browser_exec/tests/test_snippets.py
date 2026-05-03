"""Unit tests for ``browser_exec.snippets`` — pure builders +
validators. No browser required.

Covers:
  * build_session_script: emitted source compiles, SPEC round-trips,
    selector / text / URL values do NOT leak into Python source as
    bare tokens (the load-bearing safety contract).
  * validate_url: accepts http/https, rejects file/data/javascript/
    chrome/about/non-string/empty.
  * validate_action: per-kind required-field enforcement, type
    checks, scroll-direction whitelist, etc. Catches the same shape
    of bug as the gui_exec key-combo collapse — deterministic
    param error at the tool boundary.
"""
from __future__ import annotations

import json
import re

import pytest

from krakey.plugins.browser_exec.snippets import (
    ACTIONS,
    SCROLL_DIRECTIONS,
    URL_SCHEMES_ALLOWED,
    build_session_script,
    validate_action,
    validate_url,
)


# =====================================================================
# build_session_script — emitted-source contract
# =====================================================================


def _baseline_spec() -> dict:
    return {
        "browser": "chromium",
        "headless": True,
        "start_url": "https://example.com",
        "timeout_ms": 30000,
        "output": "a11y",
        "actions": [],
        "return_screenshot": False,
        "screenshot_path": None,
    }


def test_emitted_source_compiles_as_python():
    """Smoke-check: the generated string is valid Python. Catches
    template / brace-doubling / format-arg drift."""
    src = build_session_script(_baseline_spec())
    # ``compile`` raises ``SyntaxError`` on bad source; success
    # means the snippet is parseable.
    compile(src, "<browser_exec_snippet>", "exec")


def test_emitted_source_contains_no_executable_python_for_user_text():
    """Safety contract: a Self-controlled selector value containing
    Python keywords / function calls must NOT appear as Python
    source — it must live inside a JSON string literal that the
    snippet's ``json.loads`` decodes at runtime.

    We embed an obvious payload (``__import__('os').system('rm -rf /')``)
    as a selector and check the emitted source. The payload IS
    present (inside the JSON string literal) but never as bare
    Python tokens — the line containing it must be a quoted string
    arg to ``json.loads``, not a bare expression.
    """
    payload = "__import__('os').system('rm -rf /')"
    spec = _baseline_spec()
    spec["actions"] = [{"action": "click", "selector": payload}]
    src = build_session_script(spec)

    # The payload must appear (as data) somewhere in the emitted
    # source — but only inside the json.loads(...) literal.
    assert payload in src

    # Stronger: the payload appears EXACTLY ONCE (only inside the
    # JSON literal). If it leaked elsewhere we'd see it twice.
    assert src.count(payload) == 1

    # And: the line carrying the payload starts with ``SPEC =
    # json.loads(`` — i.e. it's the JSON literal line, not bare
    # source.
    payload_line = next(
        line for line in src.splitlines() if payload in line
    )
    assert payload_line.lstrip().startswith("SPEC = json.loads(")


def test_emitted_source_round_trips_spec_via_json_loads():
    """Decode the JSON literal embedded in the emitted source and
    confirm it equals the input spec. This is the runtime
    behavior of the script (``json.loads`` recovers the dict)
    minus actually launching Playwright."""
    spec = _baseline_spec()
    spec["actions"] = [
        {"action": "navigate", "url": "https://example.org/a"},
        {"action": "click", "selector": "#submit"},
        {"action": "type", "selector": "input[name='q']",
         "text": "hello \"world\"\nwith newline"},
        {"action": "press", "key": "Enter"},
        {"action": "scroll", "direction": "down", "amount": 500},
        {"action": "wait_for", "selector": ".results",
         "timeout_ms": 5000},
        {"action": "screenshot", "full_page": True},
    ]

    src = build_session_script(spec)

    # Pull the JSON literal out of the source and decode it.
    m = re.search(r"SPEC = json\.loads\((.*)\)", src)
    assert m, "could not locate SPEC = json.loads(...) line"
    inner_literal = m.group(1)
    # The literal is itself a JSON-encoded JSON string. Decoding
    # twice gives back the original dict.
    decoded = json.loads(json.loads(inner_literal))
    assert decoded == spec


def test_emitted_source_uses_static_dispatch_table_no_eval():
    """Tighter safety contract: no ``eval(`` or ``exec(`` calls
    appear in the emitted source. The only ``page.evaluate`` is the
    fixed scroll call. Verifies via substring check."""
    src = build_session_script(_baseline_spec())
    # No bare eval/exec in the snippet (would be a code-injection
    # vector if Self-controlled values were near them).
    assert "eval(" not in src
    assert "exec(" not in src
    # ``page.evaluate`` IS used — but only for the fixed scroll
    # call; check it appears with the literal scroll JS.
    assert (
        "page.evaluate('window.scrollBy(arguments[0], "
        "arguments[1])'"
    ) in src


def test_emitted_source_specifies_chromium_launcher_via_getattr():
    """Browser is picked via ``getattr(p, SPEC['browser'])`` — the
    name comes from the JSON spec, not Python source. Validates the
    pattern is in place (so a future refactor doesn't accidentally
    hardcode chromium)."""
    src = build_session_script(_baseline_spec())
    assert "getattr(p, SPEC['browser'])" in src


def test_emitted_source_size_is_reasonable_for_a_typical_call():
    """Sanity bound: a typical call fits well under the OS argv
    limit (~256KB on Linux, ~32KB on Windows command line). Default
    chromium fixture should be << 4KB."""
    src = build_session_script(_baseline_spec())
    assert len(src) < 4000  # generous; current is ~2KB


# =====================================================================
# validate_url
# =====================================================================


@pytest.mark.parametrize("url", [
    "https://example.com",
    "http://example.com/",
    "https://example.com:8080/path?q=1#frag",
    "http://192.168.1.1/",
])
def test_validate_url_accepts_http_and_https(url):
    assert validate_url(url) == url


@pytest.mark.parametrize("url, expect_substr", [
    ("file:///etc/passwd", "http://"),
    ("data:text/html,<h1>x</h1>", "http://"),
    ("javascript:alert(1)", "http://"),
    ("chrome://settings", "http://"),
    ("about:blank", "http://"),
    ("ftp://example.com", "http://"),
    ("//example.com", "http://"),
    ("example.com", "http://"),  # no scheme at all
])
def test_validate_url_rejects_non_http_schemes(url, expect_substr):
    with pytest.raises(ValueError) as ei:
        validate_url(url)
    assert expect_substr in str(ei.value)


@pytest.mark.parametrize("bad", [None, 0, [], {}, b"https://x", ""])
def test_validate_url_rejects_non_string_or_empty(bad):
    with pytest.raises(ValueError) as ei:
        validate_url(bad)
    assert "non-empty string" in str(ei.value)


def test_validate_url_field_name_appears_in_error_message():
    """Self should learn whether the bad URL was ``start_url`` or
    a per-action ``navigate.url`` so she can fix the right one."""
    with pytest.raises(ValueError) as ei:
        validate_url("file://bad", field="actions[2].url")
    assert "actions[2].url" in str(ei.value)


# =====================================================================
# validate_action — kind dispatch + required fields
# =====================================================================


def test_validate_action_rejects_non_dict():
    with pytest.raises(ValueError) as ei:
        validate_action("click", index=0)  # type: ignore[arg-type]
    assert "actions[0]" in str(ei.value)
    assert "object" in str(ei.value)


@pytest.mark.parametrize("kind", [
    "evaluate", "set_cookie", "download", "", None, 42,
])
def test_validate_action_rejects_unknown_kind(kind):
    with pytest.raises(ValueError) as ei:
        validate_action({"action": kind}, index=3)
    assert "actions[3].action" in str(ei.value)
    assert "must be one of" in str(ei.value)


# --- navigate ---------------------------------------------------------

def test_validate_action_navigate_requires_http_url():
    with pytest.raises(ValueError):
        validate_action({"action": "navigate", "url": "file:///x"}, index=0)
    with pytest.raises(ValueError):
        validate_action({"action": "navigate"}, index=0)
    # Passes:
    validate_action({"action": "navigate", "url": "https://x"}, index=0)


# --- click / press / wait_for / type ----------------------------------

@pytest.mark.parametrize("kind, field", [
    ("click", "selector"),
    ("press", "key"),
    ("wait_for", "selector"),
])
def test_validate_action_requires_string_field(kind, field):
    with pytest.raises(ValueError) as ei:
        validate_action({"action": kind}, index=1)
    assert field in str(ei.value)
    with pytest.raises(ValueError):
        validate_action({"action": kind, field: ""}, index=1)
    with pytest.raises(ValueError):
        validate_action({"action": kind, field: 5}, index=1)
    # Passes:
    validate_action({"action": kind, field: "x"}, index=1)


def test_validate_action_type_allows_empty_text_for_clear():
    """``type`` with text='' is the documented "clear field" use-case
    (Playwright's ``page.fill(selector, '')`` clears the input).
    Must NOT be rejected — only None / wrong-type is rejected."""
    validate_action({
        "action": "type", "selector": "#name", "text": "",
    }, index=0)


def test_validate_action_type_rejects_non_string_text():
    with pytest.raises(ValueError):
        validate_action({
            "action": "type", "selector": "#name", "text": None,
        }, index=0)
    with pytest.raises(ValueError):
        validate_action({
            "action": "type", "selector": "#name", "text": 42,
        }, index=0)


# --- scroll -----------------------------------------------------------

@pytest.mark.parametrize("direction", list(SCROLL_DIRECTIONS))
def test_validate_action_scroll_accepts_each_direction(direction):
    validate_action({
        "action": "scroll", "direction": direction, "amount": 100,
    }, index=0)


@pytest.mark.parametrize("bad_direction", [
    "diagonal", "", None, 0, "DOWN",  # case-sensitive
])
def test_validate_action_scroll_rejects_bad_direction(bad_direction):
    with pytest.raises(ValueError) as ei:
        validate_action({
            "action": "scroll", "direction": bad_direction,
            "amount": 100,
        }, index=0)
    assert "direction" in str(ei.value)


@pytest.mark.parametrize("bad_amount", [
    0, -1, -0.5, "100", None, True, False,
])
def test_validate_action_scroll_rejects_non_positive_amount(bad_amount):
    with pytest.raises(ValueError) as ei:
        validate_action({
            "action": "scroll", "direction": "down",
            "amount": bad_amount,
        }, index=0)
    assert "amount" in str(ei.value)


# --- wait_for timeout_ms ----------------------------------------------

def test_validate_action_wait_for_accepts_optional_timeout():
    # Without timeout_ms: OK.
    validate_action({
        "action": "wait_for", "selector": ".x",
    }, index=0)
    # With positive timeout_ms: OK.
    validate_action({
        "action": "wait_for", "selector": ".x",
        "timeout_ms": 1000,
    }, index=0)


@pytest.mark.parametrize("bad", [0, -1, "1000", None, True])
def test_validate_action_wait_for_rejects_bad_timeout(bad):
    with pytest.raises(ValueError) as ei:
        validate_action({
            "action": "wait_for", "selector": ".x",
            "timeout_ms": bad,
        }, index=0)
    assert "timeout_ms" in str(ei.value)


# --- screenshot full_page ---------------------------------------------

def test_validate_action_screenshot_default_is_ok():
    validate_action({"action": "screenshot"}, index=0)
    validate_action(
        {"action": "screenshot", "full_page": True}, index=0,
    )
    validate_action(
        {"action": "screenshot", "full_page": False}, index=0,
    )


@pytest.mark.parametrize("bad", ["yes", 1, 0, None, "true"])
def test_validate_action_screenshot_rejects_non_bool_full_page(bad):
    with pytest.raises(ValueError) as ei:
        validate_action(
            {"action": "screenshot", "full_page": bad}, index=0,
        )
    assert "full_page" in str(ei.value)


# =====================================================================
# Module-level constants — pin the public surface
# =====================================================================


def test_url_schemes_allowed_is_http_and_https_only():
    assert set(URL_SCHEMES_ALLOWED) == {"http://", "https://"}


def test_actions_constant_matches_validator_branches():
    """If a new action kind is added to ACTIONS but no validator
    branch handles it, ``validate_action`` would silently allow
    malformed payloads. Pin the contract."""
    assert set(ACTIONS) == {
        "navigate", "click", "type", "press",
        "scroll", "wait_for", "screenshot",
    }
