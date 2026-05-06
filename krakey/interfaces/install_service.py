"""``InstallService`` Protocol — abstract surface for "install +
post_install + state-tracking" of plugin dependencies.

Defined here so consumers (the runtime, the InstallTool built-in,
the dashboard's plugin adapter) can depend ONLY on this Protocol
and never on the concrete CLI implementation. The composition
root (``krakey.main.build_runtime_from_config`` / wherever a
Runtime is constructed) is the ONLY place that knows about the
default implementation in ``krakey.install``.

This honours two architectural rules at once:

  * Direction: ``runtime/`` must not import from ``cli/``. CLI is
    a wrapper around the runtime, so dependency must flow
    cli → runtime, not the reverse. Pre-DIP, ``runtime/runtime.py``
    and ``runtime/builtin_tools/install_tool.py`` both did
    ``from krakey.cli import install`` — that's gone now.

  * Dependency Inversion Principle: consumers depend on this
    Protocol, not on any class that implements it. Tests
    substitute fakes that implement the same Protocol. A future
    second implementation (e.g. an HTTP-backed installer for a
    dashboard-only deployment) drops in without touching any
    consumer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class InstallResult:
    """Outcome of a single install run."""
    rc: int
    stdout: str
    stderr: str


@runtime_checkable
class InstallService(Protocol):
    """The pip-and-post_install workflow, abstracted.

    Implementations must:
      * walk plugin folders to collect declared deps + post_install
        hooks;
      * hash that surface so callers can decide whether install is
        pending;
      * persist + read install state (which plugins were installed
        + when + with what hash);
      * dispatch ``pip install`` and ``post_install`` commands;
      * return enough captured output for callers to render to a
        log / Stimulus / dashboard pane.

    Consumers:
      * ``Runtime._maybe_push_install_advisory`` calls
        ``has_pending_deps()`` at startup and pushes a Stimulus
        when the answer is True.
      * ``InstallTool.execute`` calls ``install(...)`` so Self can
        self-repair plugin deps.
      * ``RuntimePluginsService.deps_status`` / ``.install``
        (dashboard adapter) call ``deps_status()`` + ``install(...)``
        so the operator-facing endpoints fan out the same calls.
    """

    def has_pending_deps(self) -> tuple[bool, dict[str, list[str]]]:
        """``(pending, plugin_deps)``. Pending=True when no
        install_state.json yet OR the recorded deps_hash differs
        from the live one. ``plugin_deps`` is the per-plugin
        ``{name → [pip-spec strings]}`` snapshot from the live
        meta.yaml walk (handed to callers for advisory text)."""
        ...

    def collect_plugin_dependencies(self) -> dict[str, list[str]]:
        """``{plugin_name: [pip-spec-strings]}`` from every plugin
        folder under BUILTIN_ROOT + WORKSPACE_ROOT. Workspace wins
        on name collisions."""
        ...

    def collect_plugin_post_install(self) -> dict[str, list[dict[str, Any]]]:
        """``{plugin_name: [{args, description, optional}]}`` from
        every plugin folder. Same walk semantics as deps."""
        ...

    def deps_status(self) -> dict[str, Any]:
        """Per-plugin install-state snapshot for the dashboard's
        plugin list. Shape: ``{pending, plugins: {name: {...}},
        state: {installed_at, deps_hash, live_hash}}``."""
        ...

    def install(
        self,
        *,
        upgrade: bool = False,
        dry_run: bool = False,
    ) -> InstallResult:
        """Run the full ``pip install`` + per-plugin
        ``post_install`` chain. Returns rc + captured stdout +
        stderr. State is written ONLY on rc==0 + no
        non-optional post_install failures. ``dry_run=True``
        prints discovery without invoking pip; rc=0 always."""
        ...
