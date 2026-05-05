"""Unit tests for the built-in ``install`` tool — Self's
self-repair surface for missing plugin deps.

Pinned behaviors:
  * Tool name + description / schema visible to Self.
  * execute() runs the same krakey.cli.install.install code-path
    a shell ``krakey install`` does, with stdout / stderr
    captured into the returned Stimulus.
  * rc == 0 → success Stimulus, adrenalin=False.
  * rc != 0 → error Stimulus, adrenalin=True (so Self prioritises
    deciding what to do — retry / report-to-user / abandon).
  * ``upgrade`` flag plumbs through to install.
  * Crash inside install() returns an error Stimulus instead of
    propagating the exception (additive-plugin invariant).

Tests stub install_mod.install so pip is never actually run.
"""
from __future__ import annotations

import argparse
from datetime import datetime

import pytest

from krakey.cli import install as install_mod
from krakey.runtime.builtin_tools import (
    INSTALL_TOOL_NAME,
    InstallTool,
)


# =====================================================================
# Static surface
# =====================================================================


def test_tool_name():
    assert InstallTool().name == INSTALL_TOOL_NAME == "install"


def test_tool_description_mentions_self_facing_use_cases():
    desc = InstallTool().description
    # Description must explicitly tell Self when to call this so
    # she doesn't have to guess from the name alone.
    assert "ModuleNotFoundError" in desc
    assert "playwright install chromium" in desc
    assert "post_install" in desc
    assert "rc" in desc


def test_tool_schema_advertises_plugins_and_upgrade():
    schema = InstallTool().parameters_schema
    assert schema["type"] == "object"
    assert "plugins" in schema["properties"]
    assert schema["properties"]["plugins"]["type"] == "array"
    assert "upgrade" in schema["properties"]
    assert schema["properties"]["upgrade"]["type"] == "boolean"
    # No required fields — Self can call install with no args
    # to get the default "install everything pending" behaviour.
    assert "required" not in schema or not schema.get("required")


# =====================================================================
# execute() — happy path + failure path
# =====================================================================


async def test_execute_success_returns_low_priority_feedback(
    monkeypatch,
):
    monkeypatch.setattr(
        install_mod, "install",
        lambda args: (print("pip ok"), 0)[1],
    )
    s = await InstallTool().execute("repair browser_exec", {})
    assert s.type == "tool_feedback"
    assert s.source == f"tool:{INSTALL_TOOL_NAME}"
    assert s.adrenalin is False
    assert "rc=0" in s.content
    # stdout from the inner install() run is captured into the
    # Stimulus so Self can read what pip did.
    assert "pip ok" in s.content


async def test_execute_failure_returns_adrenalin_feedback(
    monkeypatch,
):
    monkeypatch.setattr(
        install_mod, "install",
        lambda args: (print("pip rc=1 retry"), 1)[1],
    )
    s = await InstallTool().execute("repair", {})
    assert s.adrenalin is True
    assert "rc=1" in s.content
    assert "FAILED" in s.content


async def test_execute_threads_upgrade_flag(monkeypatch):
    captured: list[argparse.Namespace] = []

    def fake_install(args):
        captured.append(args)
        return 0

    monkeypatch.setattr(install_mod, "install", fake_install)
    await InstallTool().execute("force-refresh", {"upgrade": True})
    assert len(captured) == 1
    assert captured[0].upgrade is True
    assert captured[0].dry_run is False


async def test_execute_default_upgrade_is_false(monkeypatch):
    captured: list[argparse.Namespace] = []
    monkeypatch.setattr(
        install_mod, "install",
        lambda args: (captured.append(args), 0)[1],
    )
    await InstallTool().execute("default", {})
    assert captured[0].upgrade is False


async def test_execute_validates_plugins_list_type(monkeypatch):
    """`plugins` must be a list of strings if provided. Wrong
    types should error at the tool boundary rather than crash
    deep inside install()."""
    # If validation passes the install_mod patch isn't consulted,
    # but stub it anyway so we don't hit the real install on
    # accidental fall-through.
    monkeypatch.setattr(install_mod, "install", lambda args: 0)
    s = await InstallTool().execute(
        "scoped",
        {"plugins": ["browser_exec", 7]},  # type: ignore[arg-type]
    )
    assert s.adrenalin is True
    assert "must be a list of strings" in s.content


async def test_execute_swallows_install_crash(monkeypatch):
    """If install() raises (bug, not a clean rc!=0 exit), the
    tool returns an error Stimulus rather than letting the
    exception propagate. Additive-plugin invariant covers
    advisory tools too."""
    def boom(args):
        raise RuntimeError("install module bug")

    monkeypatch.setattr(install_mod, "install", boom)
    s = await InstallTool().execute("repair", {})
    assert s.adrenalin is True
    assert "install crashed" in s.content
    assert "RuntimeError" in s.content


async def test_execute_truncates_giant_output(monkeypatch):
    """A pip install that prints megabytes (e.g. on a build-
    from-source dep) shouldn't blow up Self's prompt. Output
    is capped at 4000 chars per stream."""
    huge = "x" * 10_000

    def fake(args):
        print(huge)
        return 0

    monkeypatch.setattr(install_mod, "install", fake)
    s = await InstallTool().execute("repair", {})
    assert s.content.count("x") < 5000  # truncated
    assert "truncated" in s.content


# =====================================================================
# Tool is registered on Runtime.tools BEFORE plugins
# =====================================================================


def test_install_tool_registered_in_runtime_tools():
    """Sanity: the runtime composition root registers InstallTool
    alongside SleepTool. If a plugin tries to register a
    ``install`` tool, the registry's duplicate-name guard
    refuses it."""
    from tests._runtime_helpers import build_runtime_with_fakes

    class _StubLLM:
        async def chat(self, messages, **kw):
            return ""

    runtime = build_runtime_with_fakes(
        self_llm=_StubLLM(), hypo_llm=_StubLLM(),
    )
    tool = runtime.tools.get(INSTALL_TOOL_NAME)
    assert tool is not None
    assert isinstance(tool, InstallTool)


# =====================================================================
# Startup advisory — Self gets a Stimulus on first heartbeat when
# install state is stale
# =====================================================================


def _runtime_with_advisory_on(self_llm):
    """build_runtime_with_fakes defaults to advisory OFF (so
    existing buffer-state tests aren't surprised). The advisory
    tests explicitly flip it back ON via direct attribute set
    after construction."""
    from tests._runtime_helpers import build_runtime_with_fakes

    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=self_llm,
    )
    # Re-enable the advisory for this specific test.
    runtime._enable_install_advisory = True
    return runtime


async def test_runtime_pushes_install_advisory_when_state_missing(
    monkeypatch, tmp_path,
):
    """No install_state.json on disk → has_pending_deps=True →
    ``run()`` pushes a system:install Stimulus before the first
    heartbeat fires."""
    class _StubLLM:
        async def chat(self, messages, **kw):
            return "[DECISION]\nNo action.\n[IDLE]\n1"

    runtime = _runtime_with_advisory_on(_StubLLM())

    monkeypatch.setattr(
        install_mod, "has_pending_deps",
        lambda: (True, {"browser_exec": ["playwright>=1.40"]}),
    )

    # Run zero iterations (just startup) so the advisory push
    # happens but the heartbeat doesn't actually consume it.
    await runtime.run(iterations=0)

    drained = runtime.buffer.drain()
    install_advisories = [
        s for s in drained if s.source == "system:install"
    ]
    assert len(install_advisories) == 1
    s = install_advisories[0]
    assert s.adrenalin is True
    assert "install" in s.content.lower()
    assert "browser_exec" in s.content


async def test_runtime_silent_when_install_state_current(
    monkeypatch, tmp_path,
):
    """When has_pending_deps=False, no advisory Stimulus fires —
    Self isn't bothered every startup if everything is in order."""
    class _StubLLM:
        async def chat(self, messages, **kw):
            return "[DECISION]\nNo action.\n[IDLE]\n1"

    runtime = _runtime_with_advisory_on(_StubLLM())

    monkeypatch.setattr(
        install_mod, "has_pending_deps", lambda: (False, {}),
    )

    await runtime.run(iterations=0)

    drained = runtime.buffer.drain()
    advisories = [s for s in drained if s.source == "system:install"]
    assert advisories == []


async def test_runtime_install_advisory_swallows_check_exception(
    monkeypatch, tmp_path,
):
    """If has_pending_deps raises (corrupt workspace, weird env),
    runtime startup must NOT crash — the advisory is best-effort."""
    class _StubLLM:
        async def chat(self, messages, **kw):
            return "[DECISION]\nNo action.\n[IDLE]\n1"

    runtime = _runtime_with_advisory_on(_StubLLM())

    def boom():
        raise RuntimeError("workspace blew up")

    monkeypatch.setattr(install_mod, "has_pending_deps", boom)

    # Must not raise.
    await runtime.run(iterations=0)
    drained = runtime.buffer.drain()
    # No advisory pushed (the check failed silently).
    assert not any(s.source == "system:install" for s in drained)


async def test_runtime_install_advisory_off_by_default_in_helper(
    monkeypatch, tmp_path,
):
    """Sanity: build_runtime_with_fakes defaults the advisory OFF
    so existing tests with buffer-state assertions aren't broken
    by the new startup push."""
    from tests._runtime_helpers import build_runtime_with_fakes

    class _StubLLM:
        async def chat(self, messages, **kw):
            return "[DECISION]\nNo action.\n[IDLE]\n1"

    runtime = build_runtime_with_fakes(
        self_llm=_StubLLM(), hypo_llm=_StubLLM(),
    )
    monkeypatch.setattr(
        install_mod, "has_pending_deps",
        lambda: (True, {"x": ["pkg"]}),
    )
    await runtime.run(iterations=0)
    assert not any(
        s.source == "system:install" for s in runtime.buffer.drain()
    )