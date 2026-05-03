"""Pure builders for the ``browser_exec`` Playwright session script.

The tool dispatches one Python source string per call as
``[python_cmd, "-c", snippet]``. The snippet template is fixed at
build time; only the JSON-encoded SPEC dict varies per call.
Selectors / text / URLs from Self travel as JSON string values
inside the snippet's ``SPEC`` dict — they are NEVER interpolated
into Python or JS source. This is the load-bearing safety contract.

Public API:

    build_session_script(spec) -> str
        Return a self-contained Python source string. Raises
        nothing; all validation happens at the tool layer before
        this is called.

    validate_url(url) -> str
        Reject non-string / non-http(s) URLs. Used by the tool to
        gate ``start_url`` and per-action ``navigate.url`` BEFORE
        the snippet is built — deterministic param error, not a
        runtime subprocess failure.

    validate_action(action_dict) -> None
        Validate one action dict's shape (kind + required per-kind
        fields + types). Raises ``ValueError`` with a precise
        message; tool layer catches and converts to error Stimulus.

    URL_SCHEMES_ALLOWED, ACTIONS, SCROLL_DIRECTIONS — module-level
        constants reused by the tool layer.
"""
from __future__ import annotations

import json
from typing import Any


URL_SCHEMES_ALLOWED = ("http://", "https://")
"""Only HTTP(S) URLs cross the tool boundary. ``file://``,
``data:``, ``javascript:``, ``chrome://``, ``about:`` are rejected
— they bypass network policy or grant local-file / privileged-page
access we do not want Self to reach."""

ACTIONS = (
    "navigate", "click", "type", "press",
    "scroll", "wait_for", "screenshot",
)
"""Action kinds Self may include in the ``actions`` array."""

SCROLL_DIRECTIONS = ("up", "down", "left", "right")


def validate_url(url: Any, *, field: str = "start_url") -> str:
    """Reject non-string, non-http(s) URLs. Returns the validated
    string on success.

    The name of the field being validated is included in error
    messages so Self gets a precise pointer (``start_url`` vs
    ``actions[3].url``) when the snippet is rejected at the tool
    boundary.
    """
    if not isinstance(url, str) or not url:
        raise ValueError(
            f"`{field}` must be a non-empty string"
        )
    if not any(url.startswith(s) for s in URL_SCHEMES_ALLOWED):
        raise ValueError(
            f"`{field}` must start with http:// or https:// "
            f"(got: {url[:40]!r})"
        )
    return url


def validate_action(a: Any, *, index: int = 0) -> None:
    """Validate one action dict in-place. Raises ``ValueError`` on
    any shape problem — caller catches and converts to error
    Stimulus.

    Per-kind required fields are enforced strictly so unknown
    action kinds and missing fields are deterministic param errors,
    not runtime subprocess failures (matches the lesson learned
    from the gui_exec key-combo collapse bug)."""
    if not isinstance(a, dict):
        raise ValueError(
            f"`actions[{index}]` must be an object, got "
            f"{type(a).__name__}"
        )
    kind = a.get("action")
    if kind not in ACTIONS:
        raise ValueError(
            f"`actions[{index}].action` must be one of "
            f"{list(ACTIONS)}, got {kind!r}"
        )

    def _str_field(name: str) -> None:
        v = a.get(name)
        if not isinstance(v, str) or not v:
            raise ValueError(
                f"`actions[{index}]` action={kind!r} requires "
                f"non-empty string `{name}`"
            )

    if kind == "navigate":
        validate_url(a.get("url"), field=f"actions[{index}].url")
    elif kind == "click":
        _str_field("selector")
    elif kind == "type":
        _str_field("selector")
        # `text` may be empty string (legitimate "clear field"
        # use-case) but must be a string, not None / int / list.
        if not isinstance(a.get("text"), str):
            raise ValueError(
                f"`actions[{index}]` action='type' requires "
                f"string `text` (empty OK to clear)"
            )
    elif kind == "press":
        _str_field("key")
    elif kind == "scroll":
        if a.get("direction") not in SCROLL_DIRECTIONS:
            raise ValueError(
                f"`actions[{index}]` action='scroll' requires "
                f"`direction` in {list(SCROLL_DIRECTIONS)}, got "
                f"{a.get('direction')!r}"
            )
        amt = a.get("amount")
        if (
            not isinstance(amt, (int, float))
            or isinstance(amt, bool)
            or amt <= 0
        ):
            raise ValueError(
                f"`actions[{index}]` action='scroll' requires "
                f"positive number `amount`"
            )
    elif kind == "wait_for":
        _str_field("selector")
        if "timeout_ms" in a:
            tm = a["timeout_ms"]
            if (
                not isinstance(tm, (int, float))
                or isinstance(tm, bool)
                or tm <= 0
            ):
                raise ValueError(
                    f"`actions[{index}]` action='wait_for' "
                    f"`timeout_ms` must be a positive number"
                )
    elif kind == "screenshot":
        if "full_page" in a and not isinstance(a["full_page"], bool):
            raise ValueError(
                f"`actions[{index}]` action='screenshot' "
                f"`full_page` must be a boolean if provided"
            )


# The snippet template. Kept as a module-level string so it's
# easy to inspect and safe-guards against accidental f-string
# interpolation of caller-controlled values (only ``{spec_literal}``
# is substituted, and that's already a Python str literal of JSON).
#
# Doubled braces (``{{ ... }}``) survive .format() as single braces
# in the emitted source.
_SCRIPT_TEMPLATE = """\
import json, os, sys
from playwright.sync_api import sync_playwright
SPEC = json.loads({spec_literal})
with sync_playwright() as p:
    launcher = getattr(p, SPEC['browser'])
    browser = launcher.launch(headless=SPEC['headless'])
    completed = 0
    try:
        page = browser.new_page()
        timeout_ms = int(SPEC['timeout_ms'])
        page.goto(SPEC['start_url'], timeout=timeout_ms)
        for a in SPEC['actions']:
            kind = a['action']
            if kind == 'navigate':
                page.goto(a['url'], timeout=timeout_ms)
            elif kind == 'click':
                page.click(a['selector'], timeout=timeout_ms)
            elif kind == 'type':
                page.fill(a['selector'], a['text'], timeout=timeout_ms)
            elif kind == 'press':
                page.keyboard.press(a['key'])
            elif kind == 'scroll':
                d = a['direction']
                amt = a['amount']
                if d == 'down':    dx, dy = 0, amt
                elif d == 'up':    dx, dy = 0, -amt
                elif d == 'right': dx, dy = amt, 0
                else:              dx, dy = -amt, 0
                page.evaluate('window.scrollBy(arguments[0], arguments[1])', [dx, dy])
            elif kind == 'wait_for':
                tm = int(a.get('timeout_ms', timeout_ms))
                page.wait_for_selector(a['selector'], timeout=tm)
            elif kind == 'screenshot':
                sp = SPEC['screenshot_path']
                os.makedirs(os.path.dirname(sp) or '.', exist_ok=True)
                page.screenshot(path=sp, full_page=bool(a.get('full_page', False)))
            else:
                raise ValueError('unknown action: ' + repr(kind))
            completed += 1
        if SPEC['output'] == 'a11y':
            result = page.accessibility.snapshot()
        elif SPEC['output'] == 'text':
            result = page.inner_text('body')
        elif SPEC['output'] == 'html':
            result = page.content()
        else:
            raise ValueError('unknown output format: ' + repr(SPEC['output']))
        sys.stdout.write(json.dumps({{
            'final_url':         page.url,
            'output_format':     SPEC['output'],
            'output':            result,
            'screenshot_path':   SPEC.get('screenshot_path') if SPEC.get('return_screenshot') else None,
            'actions_completed': completed,
            'actions_total':     len(SPEC['actions']),
        }}))
    finally:
        browser.close()
"""


def build_session_script(spec: dict[str, Any]) -> str:
    """Build the Python source for one browser session.

    ``spec`` is expected to already be validated by the tool layer
    (URLs / action shapes / browser name / output format / etc.).
    This function does not re-validate; it only encodes ``spec`` as
    JSON, wraps that JSON inside a Python string literal, and
    interpolates it into the fixed template.

    Round-trip:
      1. JSON-encode ``spec`` → JSON string.
      2. JSON-encode the JSON string → a Python-safe string literal
         (escapes quotes, backslashes, control chars).
      3. Substitute that literal into ``_SCRIPT_TEMPLATE`` at
         ``{spec_literal}``.
      4. The emitted script does ``json.loads(<literal>)`` to get
         back the original dict.

    No selector / text / URL value reaches Python source as a bare
    token — they're all string values inside the JSON dict. This is
    the safety contract the snippet relies on.
    """
    # Outer json.dumps wraps the inner JSON string in valid Python
    # source (a quoted string literal). The inner json.dumps
    # produces the JSON document itself.
    spec_json = json.dumps(spec, ensure_ascii=False)
    spec_literal = json.dumps(spec_json)
    return _SCRIPT_TEMPLATE.format(spec_literal=spec_literal)
