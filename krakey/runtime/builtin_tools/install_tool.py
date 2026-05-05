"""``InstallTool`` — built-in tool that lets Self self-repair
missing plugin dependencies without operator intervention.

Why built-in: dep installation is a runtime-lifecycle capability,
not a plugin concern. If browser_exec is missing playwright, Self
should be able to call ``install`` without first needing
browser_exec to work — bootstrapping problem dissolves.

Why a Tool (not a slash-command or other special path): it
surfaces in ``[CAPABILITIES]`` through the same mechanism every
other tool uses, so Self learns about it through the prompt
rather than a separate teaching layer. Self's contract:

    <tool_call>
    {"name": "install"}                          ← install everything pending
    </tool_call>

    <tool_call>
    {"name": "install", "arguments":
       {"plugins": ["browser_exec"]}}            ← scoped to one plugin
    </tool_call>

    <tool_call>
    {"name": "install", "arguments": {"upgrade": true}}
    </tool_call>

The tool runs the same ``krakey.cli.install.install`` code-path
``krakey install`` from a shell does — same pip command, same
post_install hooks, same install_state.json bookkeeping. Result
flows back as a tool_feedback Stimulus with rc + a tail of
stdout/stderr so Self can:

  * Continue using the now-installed plugin if rc==0.
  * Report the failure to the user via her outbound channel
    (telegram, web_chat) when rc!=0 and the error suggests
    operator intervention (no network, PyPI unreachable, etc.).
  * Try ``upgrade=True`` once if the first attempt failed
    on a stale wheel.

The tool intentionally does NOT block on the install — it runs
synchronously in the dispatcher's task pool and returns when
pip is done. For a typical browser_exec install (~1 minute
including chromium binary download) Self's next heartbeat may
be delayed by that much. Acceptable: Self is pausing other
work to fix herself.
"""
from __future__ import annotations

import argparse
import contextlib
import io
from datetime import datetime
from typing import Any

from krakey.interfaces.tool import Tool
from krakey.models.stimulus import Stimulus


INSTALL_TOOL_NAME = "install"

_OUTPUT_TRUNCATE = 4000


class InstallTool(Tool):
    """Reserved built-in tool that runs ``krakey install`` from
    inside the runtime."""

    @property
    def name(self) -> str:
        return INSTALL_TOOL_NAME

    @property
    def description(self) -> str:
        return (
            "Install (or repair) plugin dependencies. Use when a "
            "tool you need reports ModuleNotFoundError or another "
            "missing-dependency error, or when the runtime tells "
            "you plugin deps are out-of-date. Runs the same logic "
            "as the operator's `krakey install` command — pip "
            "install of every plugin's declared deps + each "
            "plugin's `post_install` hooks (e.g. browser_exec's "
            "`playwright install chromium`). Optional arguments: "
            "`plugins` (list of names; default = all), `upgrade` "
            "(bool; default false; passes --upgrade to pip). "
            "Returns rc + a tail of pip's stdout / stderr — if "
            "rc!=0, decide whether to retry (transient: PyPI "
            "rate-limit) or escalate to the user (no network / "
            "policy-blocked CDN)."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "plugins": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of plugin names to scope "
                        "the install to. When omitted, install "
                        "everything pending. (Note: pip resolves "
                        "the union; scoping is advisory and may "
                        "still pull shared deps.)"
                    ),
                },
                "upgrade": {
                    "type": "boolean",
                    "description": (
                        "Pass --upgrade to pip so already-installed "
                        "deps get re-resolved. Default false."
                    ),
                },
            },
            "additionalProperties": False,
        }

    async def execute(
        self, intent: str, params: dict[str, Any],
    ) -> Stimulus:
        from krakey.cli import install as install_mod

        upgrade = bool(params.get("upgrade", False))
        # ``plugins`` filter is advisory — pip resolves the union
        # anyway. We keep it in the schema so Self can express
        # intent ("install browser_exec specifically"), and we
        # echo it back in the response for traceability.
        plugins_filter = params.get("plugins") or []
        if plugins_filter and not all(
            isinstance(p, str) for p in plugins_filter
        ):
            return self._err(
                "`plugins` must be a list of strings if provided",
            )

        # Capture stdout + stderr so the dashboard log + Self's
        # feedback Stimulus see what pip / post_install actually
        # did. The CLI install function prints freely; we
        # redirect to our own buffer.
        out_buf = io.StringIO()
        err_buf = io.StringIO()

        try:
            with (
                contextlib.redirect_stdout(out_buf),
                contextlib.redirect_stderr(err_buf),
            ):
                rc = install_mod.install(
                    argparse.Namespace(
                        dry_run=False, upgrade=upgrade,
                    ),
                )
        except Exception as e:  # noqa: BLE001
            return self._err(
                f"install crashed: {type(e).__name__}: {e}\n"
                f"stdout:\n{out_buf.getvalue()[:_OUTPUT_TRUNCATE]}\n"
                f"stderr:\n{err_buf.getvalue()[:_OUTPUT_TRUNCATE]}",
            )

        out = _truncate(out_buf.getvalue(), _OUTPUT_TRUNCATE)
        err = _truncate(err_buf.getvalue(), _OUTPUT_TRUNCATE)

        if rc == 0:
            content = (
                f"install rc=0 ok"
                + (f" (scope: {plugins_filter})" if plugins_filter else "")
                + (" (upgrade=true)" if upgrade else "")
                + f"\n--- stdout ---\n{out}\n"
                f"--- stderr ---\n{err}"
            )
            return Stimulus(
                type="tool_feedback",
                source=f"tool:{INSTALL_TOOL_NAME}",
                content=content,
                timestamp=datetime.now(),
                adrenalin=False,
            )

        # rc != 0 — surface as a tool_feedback Stimulus with
        # adrenalin=True so Self prioritizes deciding what to
        # do (retry / report-to-user / abandon).
        return Stimulus(
            type="tool_feedback",
            source=f"tool:{INSTALL_TOOL_NAME}",
            content=(
                f"install rc={rc} FAILED. State NOT updated; "
                "next startup will warn the operator again. "
                "Decide: retry, report to user, or use a different "
                "tool path.\n"
                f"--- stdout ---\n{out}\n"
                f"--- stderr ---\n{err}"
            ),
            timestamp=datetime.now(),
            adrenalin=True,
        )

    def _err(self, msg: str) -> Stimulus:
        return Stimulus(
            type="tool_feedback",
            source=f"tool:{INSTALL_TOOL_NAME}",
            content=f"install error: {msg}",
            timestamp=datetime.now(),
            adrenalin=True,
        )


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n...[truncated, total {len(s)} chars]"
