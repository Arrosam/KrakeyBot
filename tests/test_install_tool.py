"""Unit tests for the built-in ``install`` tool — Self's
self-repair surface for missing plugin deps.

Pinned behaviors:
  * Tool name + description / schema visible to Self.
  * execute() delegates to the injected ``InstallService`` Protocol
    impl, with rc + stdout + stderr captured into the returned
    Stimulus.
  * rc == 0 → success Stimulus, adrenalin=False.
  * rc != 0 → error Stimulus, adrenalin=True (so Self prioritises
    deciding what to do — retry / report-to-user / abandon).
  * ``upgrade`` flag plumbs through to the service.
  * Crash inside service.install returns an error Stimulus
    instead of propagating the exception (additive-plugin
    invariant).
  * No InstallService injected → tool returns a clean error
    Stimulus rather than crashing.

Tests inject a fake ``InstallService`` that records calls + returns
canned ``InstallResult`` objects — no monkeypatching of module
internals, no real pip subprocess.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from krakey.interfaces.install_service import InstallResult, InstallService
from krakey.runtime.builtin_tools import (
    INSTALL_TOOL_NAME,
    InstallTool,
)


class FakeInstallService:
    """In-memory ``InstallService`` for tests. Records every
    install() call so assertions can verify what the tool
    forwarded; returns whatever ``InstallResult`` was queued."""

    def __init__(
        self,
        result: InstallResult | None = None,
        has_pending: tuple[bool, dict] = (False, {}),
        deps_status_value: dict | None = None,
        crash: BaseException | None = None,
    ):
        self._result = result or InstallResult(
            rc=0, stdout="", stderr="",
        )
        self._has_pending = has_pending
        self._deps_status_value = deps_status_value or {}
        self._crash = crash
        self.install_calls: list[dict] = []

    def has_pending_deps(self):
        return self._has_pending

    def collect_plugin_dependencies(self):
        return {}

    def collect_plugin_post_install(self):
        return {}

    def deps_status(self):
        return self._deps_status_value

    def install(self, *, upgrade=False, dry_run=False):
        self.install_calls.append(
            {"upgrade": upgrade, "dry_run": dry_run},
        )
        if self._crash is not None:
            raise self._crash
        return self._result


# =====================================================================
# Static surface
# =====================================================================


def test_tool_name():
    assert InstallTool().name == INSTALL_TOOL_NAME == "install"


def test_tool_description_mentions_self_facing_use_cases():
    desc = InstallTool().description
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
    assert "required" not in schema or not schema.get("required")


# =====================================================================
# execute() — happy path + failure path (via injected fake service)
# =====================================================================


async def test_execute_success_returns_low_priority_feedback():
    svc = FakeInstallService(
        result=InstallResult(rc=0, stdout="pip ok", stderr=""),
    )
    s = await InstallTool(install_service=svc).execute(
        "repair browser_exec", {},
    )
    assert s.type == "tool_feedback"
    assert s.source == f"tool:{INSTALL_TOOL_NAME}"
    assert s.adrenalin is False
    assert "rc=0" in s.content
    assert "pip ok" in s.content


async def test_execute_failure_returns_adrenalin_feedback():
    svc = FakeInstallService(
        result=InstallResult(
            rc=1, stdout="", stderr="pip rc=1 retry",
        ),
    )
    s = await InstallTool(install_service=svc).execute("repair", {})
    assert s.adrenalin is True
    assert "rc=1" in s.content
    assert "FAILED" in s.content


async def test_execute_threads_upgrade_flag():
    svc = FakeInstallService()
    await InstallTool(install_service=svc).execute(
        "force-refresh", {"upgrade": True},
    )
    assert svc.install_calls == [
        {"upgrade": True, "dry_run": False},
    ]


async def test_execute_default_upgrade_is_false():
    svc = FakeInstallService()
    await InstallTool(install_service=svc).execute("default", {})
    assert svc.install_calls == [
        {"upgrade": False, "dry_run": False},
    ]


async def test_execute_validates_plugins_list_type():
    svc = FakeInstallService()
    s = await InstallTool(install_service=svc).execute(
        "scoped",
        {"plugins": ["browser_exec", 7]},  # type: ignore[arg-type]
    )
    assert s.adrenalin is True
    assert "must be a list of strings" in s.content
    # Validation runs BEFORE the service is called.
    assert svc.install_calls == []


async def test_execute_swallows_install_crash():
    """If service.install raises (bug, not a clean rc!=0 exit),
    the tool returns an error Stimulus rather than letting the
    exception propagate."""
    svc = FakeInstallService(crash=RuntimeError("install module bug"))
    s = await InstallTool(install_service=svc).execute("repair", {})
    assert s.adrenalin is True
    assert "install service crashed" in s.content
    assert "RuntimeError" in s.content


async def test_execute_truncates_giant_output():
    """A pip install that prints megabytes shouldn't blow up
    Self's prompt. Output is capped at 4000 chars per stream."""
    huge = "x" * 10_000
    svc = FakeInstallService(
        result=InstallResult(rc=0, stdout=huge, stderr=""),
    )
    s = await InstallTool(install_service=svc).execute("repair", {})
    assert s.content.count("x") < 5000  # truncated
    assert "truncated" in s.content


async def test_execute_returns_error_when_no_service_configured():
    """Composition root chose not to inject a service (or test
    built the tool standalone without one). Tool reports
    cleanly rather than raising."""
    s = await InstallTool().execute("repair", {})
    assert s.type == "tool_feedback"
    assert s.adrenalin is True
    assert "install service not configured" in s.content


# =====================================================================
# Tool is registered on Runtime.tools BEFORE plugins
# =====================================================================


def test_install_tool_registered_in_runtime_tools():
    """Sanity: the runtime composition root registers InstallTool
    alongside SleepTool. The build_runtime_with_fakes helper
    doesn't inject an install service, but the tool is still
    constructed (with service=None) and registered."""
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


def _runtime_with_advisory_on(self_llm, install_service=None):
    """build_runtime_with_fakes defaults the advisory OFF (so
    existing buffer-state tests aren't surprised). Tests that
    want the advisory to fire pass an InstallService + flip the
    flag."""
    from tests._runtime_helpers import build_runtime_with_fakes

    runtime = build_runtime_with_fakes(
        self_llm=self_llm, hypo_llm=self_llm,
    )
    runtime._enable_install_advisory = True
    runtime._install_service = install_service
    return runtime


async def test_runtime_pushes_install_advisory_when_state_missing():
    """has_pending_deps=True via the injected service →
    ``run()`` pushes a system:install Stimulus before the first
    heartbeat fires."""
    class _StubLLM:
        async def chat(self, messages, **kw):
            return "[DECISION]\nNo action.\n[IDLE]\n1"

    svc = FakeInstallService(
        has_pending=(True, {"browser_exec": ["playwright>=1.40"]}),
    )
    runtime = _runtime_with_advisory_on(_StubLLM(), install_service=svc)

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


async def test_runtime_silent_when_install_state_current():
    """has_pending_deps=False → no advisory."""
    class _StubLLM:
        async def chat(self, messages, **kw):
            return "[DECISION]\nNo action.\n[IDLE]\n1"

    svc = FakeInstallService(has_pending=(False, {}))
    runtime = _runtime_with_advisory_on(_StubLLM(), install_service=svc)

    await runtime.run(iterations=0)

    drained = runtime.buffer.drain()
    advisories = [s for s in drained if s.source == "system:install"]
    assert advisories == []


async def test_runtime_install_advisory_swallows_check_exception():
    """If service.has_pending_deps raises, runtime startup must
    NOT crash — the advisory is best-effort."""
    class _StubLLM:
        async def chat(self, messages, **kw):
            return "[DECISION]\nNo action.\n[IDLE]\n1"

    class _BadSvc:
        def has_pending_deps(self):
            raise RuntimeError("workspace blew up")

        def collect_plugin_dependencies(self):
            return {}

        def collect_plugin_post_install(self):
            return {}

        def deps_status(self):
            return {}

        def install(self, *, upgrade=False, dry_run=False):
            return InstallResult(rc=0, stdout="", stderr="")

    runtime = _runtime_with_advisory_on(
        _StubLLM(), install_service=_BadSvc(),
    )

    # Must not raise.
    await runtime.run(iterations=0)
    drained = runtime.buffer.drain()
    assert not any(s.source == "system:install" for s in drained)


async def test_runtime_install_advisory_off_by_default_in_helper():
    """Sanity: build_runtime_with_fakes defaults the advisory OFF
    so existing tests with buffer-state assertions aren't broken
    by the new startup push, even when an InstallService IS
    injected later."""
    from tests._runtime_helpers import build_runtime_with_fakes

    class _StubLLM:
        async def chat(self, messages, **kw):
            return "[DECISION]\nNo action.\n[IDLE]\n1"

    runtime = build_runtime_with_fakes(
        self_llm=_StubLLM(), hypo_llm=_StubLLM(),
    )
    # Inject a service that says "yes pending" but leave the
    # advisory flag at its helper-default (False).
    runtime._install_service = FakeInstallService(
        has_pending=(True, {"x": ["pkg"]}),
    )
    await runtime.run(iterations=0)
    assert not any(
        s.source == "system:install" for s in runtime.buffer.drain()
    )
