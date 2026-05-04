"""Unit tests for ``browser_exec.snippets`` — pure builders +
validators.

Covers (post-v2 redesign):
  * build_dispatch_script: emitted source compiles, payload
    (op + args) round-trips through embedded JSON, SERVER_SOURCE
    is included, Self-controlled values (selectors, URLs, text)
    travel ONLY as JSON string values inside PAYLOAD, never as
    bare Python tokens.
  * SERVER_SOURCE module constant: non-empty, matches the live
    server.py source so dispatched server source can't drift
    behind the in-tree server module.
  * validate_url: accepts http/https, rejects file/data/javascript/
    chrome/about/non-string/empty.
  * validate_action: per-kind required-field enforcement, type
    checks, scroll-direction whitelist. Catches the same shape
    of bug as the gui_exec key-combo collapse — deterministic
    param error at the tool boundary.
"""
from __future__ import annotations

import inspect
import json
import re

import pytest

from krakey.plugins.browser_exec import server as server_module
from krakey.plugins.browser_exec.snippets import (
    ACTIONS,
    SCROLL_DIRECTIONS,
    SERVER_SOURCE,
    URL_SCHEMES_ALLOWED,
    build_dispatch_script,
    validate_action,
    validate_url,
)


# =====================================================================
# SERVER_SOURCE module constant
# =====================================================================


def test_server_source_is_nonempty_python():
    """SERVER_SOURCE must contain real server.py source so the
    dispatch snippet can write it to disk and spawn it. Empty /
    truncated content would silently break the persistent-server
    architecture."""
    assert isinstance(SERVER_SOURCE, str)
    assert len(SERVER_SOURCE) > 1000
    # Source compiles as Python.
    compile(SERVER_SOURCE, "<server.py>", "exec")
    # Carries the recognizable entry-point markers so we know we
    # didn't accidentally embed a stale or different module.
    assert "def serve(" in SERVER_SOURCE
    assert 'class BrowserWorker' in SERVER_SOURCE
    assert '/rpc' in SERVER_SOURCE


def test_server_source_matches_in_tree_server_module():
    """If someone edits server.py without re-importing snippets,
    the cached SERVER_SOURCE could go stale relative to the
    on-disk module. Pin equality so the test fails first."""
    assert SERVER_SOURCE == inspect.getsource(server_module)


# =====================================================================
# build_dispatch_script — emitted-source contract
# =====================================================================


def _baseline_args() -> dict:
    return {"op": "list_tabs", "args": {}}


def test_emitted_source_compiles_as_python():
    """Smoke-check: the generated string is valid Python."""
    src = build_dispatch_script(
        "list_tabs", {},
        python_cmd="python", browser="chromium", headless=True,
    )
    compile(src, "<browser_exec_dispatch>", "exec")


def test_emitted_source_embeds_payload_op_and_args_via_json():
    """Decode the PAYLOAD JSON literal embedded in the source and
    confirm it equals the input op + args."""
    src = build_dispatch_script(
        "operate",
        {
            "tab_id": "tab_abc",
            "actions": [
                {"action": "click", "selector": "#btn"},
            ],
            "output": "a11y",
        },
        python_cmd="python3", browser="chromium", headless=False,
        rpc_timeout_s=45.0,
    )
    m = re.search(r"PAYLOAD = json\.loads\((.*)\)", src)
    assert m, "could not locate PAYLOAD = json.loads(...) line"
    decoded = json.loads(json.loads(m.group(1)))
    assert decoded == {
        "op": "operate",
        "args": {
            "tab_id": "tab_abc",
            "actions": [{"action": "click", "selector": "#btn"}],
            "output": "a11y",
        },
    }


def test_emitted_source_embeds_full_server_source():
    """The dispatch snippet ships server.py verbatim so the env
    doesn't need a pre-deployed copy. Pull SERVER_SOURCE out of
    the snippet and confirm it round-trips."""
    src = build_dispatch_script(
        "list_tabs", {},
        python_cmd="python", browser="chromium", headless=True,
    )
    # SERVER_SOURCE = json.loads(<literal>) — pull literal,
    # double-decode to recover the original source.
    m = re.search(
        r"SERVER_SOURCE = json\.loads\((.*?)\)$", src, re.MULTILINE,
    )
    assert m, "could not locate SERVER_SOURCE = json.loads(...) line"
    embedded = json.loads(json.loads(m.group(1)))
    assert embedded == SERVER_SOURCE


def test_emitted_source_threads_constants_into_python_vars():
    """python_cmd / browser / headless / rpc_timeout_s appear as
    Python source literals (not Self-controlled). The snippet
    relies on them being plain Python values, not JSON-decoded."""
    src = build_dispatch_script(
        "list_tabs", {},
        python_cmd="/usr/local/bin/python3.11",
        browser="firefox", headless=False, rpc_timeout_s=42.5,
    )
    assert "PYTHON_CMD = '/usr/local/bin/python3.11'" in src
    assert "BROWSER = 'firefox'" in src
    assert "HEADLESS = False" in src
    assert "RPC_TIMEOUT_S = 42.5" in src


def test_self_controlled_payload_does_not_leak_as_bare_python():
    """Safety contract: a Self-controlled value containing Python
    code (selector with backticks/quotes/dunder) must travel as a
    JSON string value inside PAYLOAD, never as a bare Python token.

    We embed an obvious payload (``__import__('os').system('rm -rf /')``)
    as a selector. The string IS present in the source (inside the
    PAYLOAD JSON literal) but ONLY there — never as a bare
    expression."""
    payload = "__import__('os').system('rm -rf /')"
    src = build_dispatch_script(
        "operate",
        {
            "tab_id": "tab_x",
            "actions": [{"action": "click", "selector": payload}],
        },
        python_cmd="python", browser="chromium", headless=True,
    )
    # The payload appears (as data) in the source.
    assert payload in src
    # And ONLY ONCE — the line carrying it must be the
    # PAYLOAD = json.loads(...) line.
    assert src.count(payload) == 1
    payload_line = next(
        line for line in src.splitlines() if payload in line
    )
    assert payload_line.lstrip().startswith("PAYLOAD = json.loads(")


def test_emitted_source_has_no_eval_or_exec_calls():
    """Tighter safety contract: no bare eval(/exec( in the
    dispatch snippet. The snippet only TCP-talks to the server."""
    src = build_dispatch_script(
        "operate",
        {"tab_id": "t", "actions": [], "output": "a11y"},
        python_cmd="python", browser="chromium", headless=True,
    )
    assert "eval(" not in src
    assert "exec(" not in src


def test_emitted_source_size_under_argv_limit():
    """Windows command-line is ~32KB; the snippet (carrying the
    full server source as data) needs to stay under that. Linux
    is much larger (~256KB), so this is the binding constraint."""
    src = build_dispatch_script(
        "list_tabs", {},
        python_cmd="python", browser="chromium", headless=True,
    )
    assert len(src) < 30_000, (
        f"snippet too large for Windows argv: {len(src)} bytes"
    )


def test_emitted_source_chooses_detach_flags_per_platform():
    """Sanity: the snippet must mention BOTH OS-specific spawn
    paths (Linux: start_new_session; Windows: DETACHED_PROCESS /
    CREATE_NEW_PROCESS_GROUP). It picks at runtime via os.name,
    so both branches must be in the source."""
    src = build_dispatch_script(
        "list_tabs", {},
        python_cmd="python", browser="chromium", headless=True,
    )
    assert "start_new_session" in src
    assert "DETACHED_PROCESS" in src or "0x00000008" in src


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
    ("example.com", "http://"),
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


def test_validate_action_navigate_requires_http_url():
    with pytest.raises(ValueError):
        validate_action({"action": "navigate", "url": "file:///x"}, index=0)
    with pytest.raises(ValueError):
        validate_action({"action": "navigate"}, index=0)
    validate_action({"action": "navigate", "url": "https://x"}, index=0)


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
    validate_action({"action": kind, field: "x"}, index=1)


def test_validate_action_type_allows_empty_text_for_clear():
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


@pytest.mark.parametrize("direction", list(SCROLL_DIRECTIONS))
def test_validate_action_scroll_accepts_each_direction(direction):
    validate_action({
        "action": "scroll", "direction": direction, "amount": 100,
    }, index=0)


@pytest.mark.parametrize("bad_direction", [
    "diagonal", "", None, 0, "DOWN",
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


def test_validate_action_wait_for_accepts_optional_timeout():
    validate_action({
        "action": "wait_for", "selector": ".x",
    }, index=0)
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
    assert set(ACTIONS) == {
        "navigate", "click", "type", "press",
        "scroll", "wait_for", "screenshot",
    }
