"""``browser_exec`` Tool — drive a Playwright browser inside a
Self-chosen Environment.

Mirrors ``cli_exec`` and ``gui_exec``'s envelope (capture
``ctx.environment`` in factory, validate Self params, soft-fail on
every error path). Each call is one *session script*: the tool
builds a Python source string from Self's spec, dispatches it as
``[python, "-c", snippet]`` to ``env.run``, parses one JSON object
back from stdout, and wraps it as a ``tool_feedback`` Stimulus.

The script template is fixed (all action dispatch logic lives
inside the snippet); only the JSON-encoded SPEC varies per call.
Selectors / text values / URLs travel as JSON string values, never
interpolated into Python or JS source — this is the safety contract.

Skeleton: this module currently exposes the factory and the class
shell. ``execute`` raises ``NotImplementedError``; the dispatch
body lands in step 3 of the implementation plan once
``snippets.py`` is in place. The stable surface (name / description /
parameters_schema) is finalized so the dashboard catalog and the
hypothalamus translator can pin against it from day one.
"""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Callable

from krakey.interfaces.tool import Tool
from krakey.models.stimulus import Stimulus

if TYPE_CHECKING:
    from krakey.interfaces.environment import Environment
    from krakey.interfaces.plugin_context import PluginContext


DEFAULT_TIMEOUT_S = 30.0
"""Per-action timeout cap (page.goto / click / wait_for). Browser
launch + JS execution + network round-trips push this longer than
gui_exec's 15s. Self can override per call via ``timeout_s`` and the
operator can override the default via ``default_timeout_s`` in
plugin config."""

DEFAULT_PYTHON_CMD = "python"
"""Default interpreter name on the env's PATH. Same caveat as
``gui_exec``: many Linux distros only expose ``python3`` — set
``python_cmd: python3`` in plugin config when needed."""

DEFAULT_BROWSER = "chromium"
"""Default Playwright launcher. The operator must have run
``playwright install <browser>`` for whichever is selected; missing
binaries surface as a non-zero rc with a Playwright error in stderr."""

DEFAULT_HEADLESS = True
"""Default headless mode. Operator can override via plugin config
(e.g. on a sandbox with ``display: headed`` they may want visible
windows for debugging)."""

BROWSERS = ("chromium", "firefox", "webkit")
"""Allowed Playwright launcher names."""

OUTPUT_FORMATS = ("a11y", "text", "html")
"""Allowed extraction formats. ``a11y`` is the default (semantic
tree, token-efficient); ``text`` strips tags; ``html`` returns the
post-JS rendered HTML verbatim."""

ACTIONS = (
    "navigate", "click", "type", "press",
    "scroll", "wait_for", "screenshot",
)
"""Action kinds Self may include in the ``actions`` array."""

SCROLL_DIRECTIONS = ("up", "down", "left", "right")

OUTPUT_TRUNCATE_CHARS = 4000
"""Per-stream truncation cap on stderr (and on stringified output
values when the tool falls back to a textual error path). Matches
``cli_exec``'s cap so Self learns one number, not two."""

SCREENSHOT_DIR = PurePosixPath("workspace/data/screenshots")
"""Relative path inside the env's filesystem where screenshots
land. ``PurePosixPath`` so the snippet always emits ``/`` joins
regardless of host OS."""


def build_tool(ctx: "PluginContext") -> "BrowserExecTool":
    """Factory for the single ``tool`` component declared in
    ``meta.yaml``. Captures the per-plugin env resolver and reads
    optional config overrides. Invalid / missing config values fall
    back to the module defaults rather than raising — preserves the
    additive-plugin invariant (a misconfigured plugin degrades, not
    crashes)."""
    cfg = ctx.config
    py = cfg.get("python_cmd")
    python_cmd = py if isinstance(py, str) and py.strip() else DEFAULT_PYTHON_CMD

    hl = cfg.get("headless", DEFAULT_HEADLESS)
    headless = hl if isinstance(hl, bool) else DEFAULT_HEADLESS

    br = cfg.get("default_browser")
    default_browser = br if isinstance(br, str) and br in BROWSERS else DEFAULT_BROWSER

    to = cfg.get("default_timeout_s", DEFAULT_TIMEOUT_S)
    default_timeout_s = (
        float(to) if isinstance(to, (int, float)) and not isinstance(to, bool) and to > 0
        else DEFAULT_TIMEOUT_S
    )

    return BrowserExecTool(
        env_resolver=ctx.environment,
        python_cmd=python_cmd,
        headless=headless,
        default_browser=default_browser,
        default_timeout_s=default_timeout_s,
    )


class BrowserExecTool(Tool):
    """Self-facing tool that runs one browser session-script per
    call. Skeleton — ``execute`` raises ``NotImplementedError``
    until the dispatch body lands."""

    def __init__(
        self,
        env_resolver: Callable[[str], "Environment"],
        python_cmd: str = DEFAULT_PYTHON_CMD,
        headless: bool = DEFAULT_HEADLESS,
        default_browser: str = DEFAULT_BROWSER,
        default_timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self._env_resolver = env_resolver
        self._python_cmd = python_cmd
        self._headless = headless
        self._default_browser = default_browser
        self._default_timeout_s = default_timeout_s

    @property
    def name(self) -> str:
        return "browser_exec"

    @property
    def description(self) -> str:
        return (
            "Run a real browser (Playwright) inside a target "
            "Environment and execute one session script per call: "
            "`start_url` + an ordered `actions` list "
            "(navigate / click / type / press / scroll / wait_for / "
            "screenshot). The browser opens at call start and closes "
            "at call end — no state survives across calls; pass the "
            "returned `final_url` as the next call's `start_url` to "
            "continue from a post-click page. `env` selects the "
            "Environment by name (e.g. \"local\" or \"sandbox\"); the "
            "plugin must be allow-listed for that env in config. The "
            "env's Python interpreter (default \"python\"; configurable "
            "via the plugin's `python_cmd` config field) must have "
            "`playwright` installed and bundled browsers downloaded "
            "via `playwright install`. Default extraction is the "
            "page's accessibility tree (`output: \"a11y\"`); "
            "`\"text\"` and `\"html\"` are also available."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["env", "start_url", "actions"],
            "properties": {
                "env": {
                    "type": "string",
                    "description": (
                        "Environment name; e.g. \"local\" or "
                        "\"sandbox\". Must be allow-listed for this "
                        "plugin in `config.environments.<env>"
                        ".allowed_plugins`."
                    ),
                },
                "start_url": {
                    "type": "string",
                    "description": (
                        "Initial URL to load. Must start with "
                        "`http://` or `https://`; other schemes "
                        "(`file://`, `data:`, `javascript:`, etc.) "
                        "are rejected at the tool boundary."
                    ),
                },
                "actions": {
                    "type": "array",
                    "description": (
                        "Ordered list of action objects executed "
                        "inside the same browser session. Each "
                        "object has an `action` key plus per-kind "
                        "fields. Supported kinds: navigate {url}, "
                        "click {selector}, type {selector, text}, "
                        "press {key}, scroll {direction, amount}, "
                        "wait_for {selector, timeout_ms?}, "
                        "screenshot {full_page?}."
                    ),
                    "items": {"type": "object"},
                },
                "timeout_s": {
                    "type": "number",
                    "description": (
                        "Per-action timeout cap in seconds. Default "
                        f"{int(DEFAULT_TIMEOUT_S)}; configurable "
                        "by the operator via `default_timeout_s`."
                    ),
                },
                "output": {
                    "type": "string",
                    "enum": list(OUTPUT_FORMATS),
                    "description": (
                        "Final-state extraction format. `a11y` "
                        "(default) returns the page's accessibility "
                        "tree (semantic, token-efficient). `text` "
                        "strips tags; `html` returns post-JS HTML."
                    ),
                },
                "return_screenshot": {
                    "type": "boolean",
                    "description": (
                        "When true, append a `screenshot` action at "
                        "the end of the chain to capture the final "
                        "page state. Default false."
                    ),
                },
                "headless": {
                    "type": "boolean",
                    "description": (
                        "Override the configured headless setting "
                        "for this call only."
                    ),
                },
                "browser": {
                    "type": "string",
                    "enum": list(BROWSERS),
                    "description": (
                        "Override the configured default browser "
                        "for this call only."
                    ),
                },
            },
        }

    async def execute(
        self, intent: str, params: dict[str, Any],
    ) -> Stimulus:
        # Body lands in step 3 of the implementation plan, after
        # ``snippets.py`` is in place. Until then this raises so
        # accidentally-enabled deployments fail loud instead of
        # silently returning empty stimuli.
        raise NotImplementedError(
            "BrowserExecTool.execute will be implemented in the "
            "next commit (step 3 of the plan); the plugin is a "
            "skeleton today."
        )
