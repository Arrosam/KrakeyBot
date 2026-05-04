"""``browser_exec`` Tool — drive a persistent Playwright browser
inside a Self-chosen Environment.

The plugin's runtime model (plan v2) is a long-running browser
RPC server inside the env that survives across heartbeats. Each
tool call dispatches a small Python *client snippet* via
``env.run([python_cmd, "-c", snippet])``; the snippet talks to
the server over localhost TCP and prints one JSON envelope to
stdout. The tool parses that envelope and renders a
``tool_feedback`` Stimulus.

Tool surface is one-tab-per-call:
  - ``action: "list_tabs"`` — read the live tab map, no browser
    work beyond querying URLs/titles.
  - ``action: "new_tab"`` — open a new tab in the persistent
    browser, navigate to ``start_url``, return its tab_id.
  - ``action: "close_tab"`` — close a specific tab.
  - ``action: "operate"`` — run an in-tab action chain
    (click/type/press/scroll/wait_for/screenshot) on a specific
    tab. The browser instance and the tab's DOM/JS state survive
    across calls.

Every successful response includes the current ``tabs`` list so
Self always knows what's open.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Callable

from krakey.interfaces.environment import (
    EnvironmentDenied,
    EnvironmentUnavailableError,
)
from krakey.interfaces.tool import Tool
from krakey.models.stimulus import Stimulus
from krakey.plugins.browser_exec import snippets as snip

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

TOP_LEVEL_ACTIONS = ("list_tabs", "new_tab", "close_tab", "operate")
"""Top-level ``action`` enum. Each call is exactly one of these."""

ACTIONS = (
    "navigate", "click", "type", "press",
    "scroll", "wait_for", "screenshot",
)
"""In-tab action kinds Self may include in the ``operate`` action's
``actions`` array."""

SCROLL_DIRECTIONS = ("up", "down", "left", "right")

OUTPUT_TRUNCATE_CHARS = 4000
"""Per-stream truncation cap on stderr (and on stringified output
values when the tool falls back to a textual error path). Matches
``cli_exec``'s cap so Self learns one number, not two."""

SCREENSHOT_DIR = PurePosixPath("workspace/data/screenshots")
"""Relative path inside the env's filesystem where screenshots
land. ``PurePosixPath`` so the snippet always emits ``/`` joins
regardless of host OS."""

ENV_RUN_OVERHEAD_S = 60.0
"""Extra wall-clock budget on top of per-action timeouts to cover
browser cold-launch (chromium first-run is slow) + RPC overhead
when the snippet has to spawn the server. Subsequent calls hit a
warm server and rarely use this much."""


def _now_ts() -> str:
    """Filename-safe timestamp for screenshot output paths.

    Microsecond precision so tight-loop screenshot calls don't
    collide. Module-level (not inlined) so tests can monkeypatch
    it for deterministic path assertions."""
    return datetime.now().strftime("%Y%m%dT%H%M%S_%f")


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n...[truncated, total {len(s)} chars]"


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
            "Drive a persistent browser (Playwright) inside a "
            "target Environment. A long-running browser instance "
            "lives in the env across heartbeats; each call dispatches "
            "one tab-management op or one in-tab action chain. "
            "`env` picks the Environment (e.g. \"local\" or "
            "\"sandbox\"); the plugin must be allow-listed there. "
            "`action` is one of: list_tabs, new_tab, close_tab, "
            "operate. `new_tab` requires `start_url` (http/https only) "
            "and optional `label`. `close_tab` and `operate` require "
            "`tab_id` (from a previous response's `tabs` list). "
            "`operate` takes `actions` (a list of in-tab action "
            "objects: navigate / click / type / press / scroll / "
            "wait_for / screenshot) and runs them on the chosen tab. "
            "Every successful response includes the current `tabs` "
            "list so you always see what's open. The env's Python "
            "interpreter (default \"python\"; configurable via the "
            "plugin's `python_cmd` config field) must have "
            "`playwright` installed with bundled browsers downloaded "
            "via `playwright install`. Default extraction format is "
            "the page's accessibility tree (`output: \"a11y\"`); "
            "`\"text\"` and `\"html\"` are also available."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["env", "action"],
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
                "action": {
                    "type": "string",
                    "enum": list(TOP_LEVEL_ACTIONS),
                    "description": (
                        "Top-level op. `list_tabs` returns the "
                        "current tab list with no other side effect. "
                        "`new_tab` opens a fresh tab and navigates "
                        "to `start_url`. `close_tab` closes the "
                        "named tab. `operate` runs an in-tab "
                        "`actions` chain on `tab_id`."
                    ),
                },
                "start_url": {
                    "type": "string",
                    "description": (
                        "URL to navigate the new tab to (required "
                        "when `action=new_tab`). Must start with "
                        "`http://` or `https://`; other schemes "
                        "are rejected at the tool boundary."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Optional human-friendly name for the new "
                        "tab (used by `action=new_tab`); echoed "
                        "back in the `tabs` list to help you "
                        "identify tabs by purpose."
                    ),
                },
                "tab_id": {
                    "type": "string",
                    "description": (
                        "Target tab id from a previous response's "
                        "`tabs` list. Required when "
                        "`action ∈ {close_tab, operate}`."
                    ),
                },
                "actions": {
                    "type": "array",
                    "description": (
                        "Required when `action=operate`. Ordered "
                        "list of in-tab action objects, each with "
                        "an `action` key plus per-kind fields. "
                        "Supported kinds: navigate {url}, "
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
                        "Final-state extraction format for "
                        "`action=operate`. `a11y` (default) returns "
                        "the page's accessibility tree (semantic, "
                        "token-efficient). `text` strips tags; "
                        "`html` returns post-JS HTML."
                    ),
                },
                "return_screenshot": {
                    "type": "boolean",
                    "description": (
                        "When true and `action=operate`, append a "
                        "`screenshot` action at the end of the chain "
                        "to capture the final page state. Default "
                        "false. Path is returned in the response."
                    ),
                },
                "headless": {
                    "type": "boolean",
                    "description": (
                        "Override the configured headless setting. "
                        "Note: this is locked in at the server's "
                        "FIRST launch in the env; subsequent calls' "
                        "overrides are ignored until the server is "
                        "restarted (e.g. by env restart)."
                    ),
                },
                "browser": {
                    "type": "string",
                    "enum": list(BROWSERS),
                    "description": (
                        "Override the configured default browser. "
                        "Same first-launch-wins caveat as `headless`."
                    ),
                },
            },
        }

    async def execute(
        self, intent: str, params: dict[str, Any],
    ) -> Stimulus:
        # ---- 1. validate top-level params ----
        env_name = params.get("env")
        if not isinstance(env_name, str) or not env_name:
            return self._err("missing or invalid `env` parameter")

        action = params.get("action")
        if action not in TOP_LEVEL_ACTIONS:
            return self._err(
                f"`action` must be one of {list(TOP_LEVEL_ACTIONS)}",
            )

        timeout_raw = params.get("timeout_s", self._default_timeout_s)
        if (
            not isinstance(timeout_raw, (int, float))
            or isinstance(timeout_raw, bool)
            or timeout_raw <= 0
        ):
            return self._err(
                "`timeout_s` must be a positive number",
            )
        per_action_s = float(timeout_raw)

        headless_raw = params.get("headless", self._headless)
        if not isinstance(headless_raw, bool):
            return self._err(
                "`headless` must be a boolean if provided",
            )

        browser_name = params.get("browser", self._default_browser)
        if browser_name not in BROWSERS:
            return self._err(
                f"`browser` must be one of {list(BROWSERS)}",
            )

        # ---- 2. per-op validation + args assembly ----
        try:
            args = self._build_op_args(action, params, per_action_s)
        except ValueError as e:
            return self._err(str(e))

        # ---- 3. resolve env (catches denial / lookup failures BEFORE
        #         we build the dispatch snippet) ----
        try:
            env = self._env_resolver(env_name)
        except EnvironmentDenied as e:
            return self._err(
                f"environment {env_name!r} denied: {e}",
            )
        except Exception as e:  # noqa: BLE001
            return self._err(
                f"env resolver error: {type(e).__name__}: {e}",
            )

        # ---- 4. build dispatch snippet + dispatch ----
        n_actions = len(args.get("actions") or [])
        # env.run timeout = ENV_RUN_OVERHEAD_S (covers cold-start +
        # server spawn) + per-action cap × (n_actions + 1).
        # Capped above so a runaway call doesn't tie up the env.
        env_timeout = min(
            ENV_RUN_OVERHEAD_S + per_action_s * (n_actions + 1),
            10.0 * per_action_s + 90.0,
        )
        snippet = snip.build_dispatch_script(
            action,
            args,
            python_cmd=self._python_cmd,
            browser=browser_name,
            headless=bool(headless_raw),
            rpc_timeout_s=env_timeout,
        )
        cmd = [self._python_cmd, "-c", snippet]

        try:
            rc, out, err = await env.run(
                cmd,
                cwd=Path("."),
                timeout=env_timeout,
                stdin=None,
            )
        except asyncio.TimeoutError:
            return self._err(
                f"dispatch timed out after {env_timeout:.1f}s "
                f"(env={env_name!r}, action={action!r})",
            )
        except EnvironmentUnavailableError as e:
            return self._err(
                f"environment {env_name!r} unavailable: {e}",
            )
        except Exception as e:  # noqa: BLE001
            return self._err(
                f"env.run error: {type(e).__name__}: {e}",
            )

        # ---- 5. parse RPC envelope ----
        if rc != 0:
            return self._err(
                f"dispatch returned rc={rc} (env={env_name!r}); "
                f"stderr: {_truncate(err, OUTPUT_TRUNCATE_CHARS)}",
            )
        try:
            envelope = json.loads(out)
        except (json.JSONDecodeError, TypeError) as e:
            return self._err(
                f"could not parse dispatch JSON output: {e}; "
                f"stdout head: {_truncate(out[:400], 400)!r}; "
                f"stderr: {_truncate(err, OUTPUT_TRUNCATE_CHARS)}",
            )
        if not isinstance(envelope, dict) or "ok" not in envelope:
            return self._err(
                "dispatch output missing expected keys (ok, tabs); "
                f"got: {_truncate(str(envelope), 400)!r}",
            )

        tabs = envelope.get("tabs") or []

        if not envelope.get("ok"):
            err_msg = envelope.get("error") or "unspecified server error"
            log_tail = envelope.get("log_tail")
            content = (
                f"browser_exec error: env={env_name} "
                f"action={action} — {err_msg}"
            )
            if log_tail:
                content += (
                    f"\n--- server.log tail ---\n"
                    f"{_truncate(log_tail, OUTPUT_TRUNCATE_CHARS)}"
                )
            content += "\n" + _format_tabs(tabs)
            return Stimulus(
                type="tool_feedback",
                source=f"tool:{self.name}",
                content=content,
                timestamp=datetime.now(),
                adrenalin=False,
            )

        # ---- 6. success — per-op rendering ----
        result = envelope.get("result") or {}
        return Stimulus(
            type="tool_feedback",
            source=f"tool:{self.name}",
            content=self._render_success(
                env_name, action, result, tabs,
            ),
            timestamp=datetime.now(),
            adrenalin=False,
        )

    # ---- arg assembly per top-level action ----

    def _build_op_args(
        self,
        action: str,
        params: dict[str, Any],
        per_action_s: float,
    ) -> dict[str, Any]:
        """Per-op validation + RPC args assembly. Raises
        ``ValueError`` on bad input — caller wraps as error
        Stimulus."""
        timeout_ms = int(per_action_s * 1000)

        if action == "list_tabs":
            return {}

        if action == "new_tab":
            start_url = snip.validate_url(
                params.get("start_url"), field="start_url",
            )
            label = params.get("label", "")
            if not isinstance(label, str):
                raise ValueError("`label` must be a string if provided")
            return {
                "start_url": start_url,
                "label":     label,
                "timeout_ms": timeout_ms,
            }

        if action == "close_tab":
            tab_id = params.get("tab_id")
            if not isinstance(tab_id, str) or not tab_id:
                raise ValueError(
                    "`tab_id` must be a non-empty string for "
                    "action=close_tab",
                )
            return {"tab_id": tab_id}

        if action == "operate":
            tab_id = params.get("tab_id")
            if not isinstance(tab_id, str) or not tab_id:
                raise ValueError(
                    "`tab_id` must be a non-empty string for "
                    "action=operate",
                )
            actions = params.get("actions")
            if not isinstance(actions, list):
                raise ValueError(
                    "`actions` must be an array for action=operate",
                )
            for i, a in enumerate(actions):
                snip.validate_action(a, index=i)

            output_fmt = params.get("output", "a11y")
            if output_fmt not in OUTPUT_FORMATS:
                raise ValueError(
                    f"`output` must be one of {list(OUTPUT_FORMATS)}",
                )

            return_screenshot = params.get("return_screenshot", False)
            if not isinstance(return_screenshot, bool):
                raise ValueError(
                    "`return_screenshot` must be a boolean "
                    "if provided",
                )

            effective_actions = list(actions)
            if return_screenshot and not any(
                a.get("action") == "screenshot"
                for a in effective_actions
            ):
                effective_actions.append({"action": "screenshot"})

            screenshot_path: str | None = None
            if any(
                a.get("action") == "screenshot"
                for a in effective_actions
            ):
                screenshot_path = str(
                    SCREENSHOT_DIR / f"{_now_ts()}.png",
                )

            return {
                "tab_id":          tab_id,
                "actions":         effective_actions,
                "output":          output_fmt,
                "screenshot_path": screenshot_path,
                "timeout_ms":      timeout_ms,
            }

        # Unreachable — top-level validation already gated by
        # TOP_LEVEL_ACTIONS membership.
        raise ValueError(f"unknown action {action!r}")

    # ---- success rendering ----

    def _render_success(
        self,
        env_name: str,
        action: str,
        result: dict[str, Any],
        tabs: list[dict[str, Any]],
    ) -> str:
        header = f"browser_exec env={env_name} action={action} ok"
        body_parts = [header]

        if action == "new_tab":
            body_parts.append(
                f"opened tab_id={result.get('tab_id')!r} "
                f"url={result.get('url')!r} "
                f"title={result.get('title')!r}"
            )
        elif action == "close_tab":
            body_parts.append("tab closed")
        elif action == "operate":
            sub = (
                f"tab_id={result.get('_tab_id', '')!r} "
                if result.get("_tab_id") else ""
            )
            body_parts.append(
                f"{sub}final_url={result.get('final_url')!r} "
                f"format={result.get('output_format')} "
                f"actions={result.get('actions_completed')}/"
                f"{result.get('actions_total')}"
            )
            sp = result.get("screenshot_path")
            if sp:
                body_parts.append(f"screenshot={sp}")
            output_str = json.dumps(
                result.get("output"),
                ensure_ascii=False,
                indent=2,
            )
            body_parts.append(
                "--- output ---\n"
                + _truncate(output_str, OUTPUT_TRUNCATE_CHARS)
            )

        body_parts.append(_format_tabs(tabs))
        return "\n".join(body_parts)

    def _err(self, msg: str) -> Stimulus:
        return Stimulus(
            type="tool_feedback",
            source=f"tool:{self.name}",
            content=f"browser_exec error: {msg}",
            timestamp=datetime.now(),
            adrenalin=False,
        )


def _format_tabs(tabs: list[dict[str, Any]]) -> str:
    """Always-included tab listing in every response. Self uses
    this to know what's open without an explicit ``list_tabs``
    call (mirrors how ``in_mind_note`` injects state)."""
    if not tabs:
        return "--- tabs ---\n(no tabs open)"
    lines = ["--- tabs ---"]
    for t in tabs:
        label = t.get("label") or ""
        suffix = f" [{label}]" if label else ""
        lines.append(
            f"  {t.get('id')}: {t.get('url')!r} — "
            f"{t.get('title')!r}{suffix}"
        )
    return "\n".join(lines)
