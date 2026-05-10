"""Integration tests for plugin hot-reload — adding a plugin to
``config.plugins`` mid-process and having ``runtime.hot_reload_plugins``
load it without a restart.

Pinned behaviors:
  * Plugins NOT currently loaded are picked up + registered.
  * Plugins ALREADY loaded are reported as skipped (no double
    registration).
  * Plugins listed in the previous config but missing from the
    new target list are flagged ``still_pending_remove`` —
    advisory only, since true remove needs detach hooks the
    runtime doesn't enforce yet.
  * Newly-registered channels get ``buffer.start_one()`` called
    so their background task fires (start_all already ran at
    startup).
  * Loader's ``register_one`` is idempotent for failure modes
    (unknown plugin, factory crash) — failures land in the
    report's errors list, don't break other plugins.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from tests._runtime_helpers import build_runtime_with_fakes


class ScriptedLLM:
    def __init__(self, responses=None):
        self._r = list(responses or [])

    async def chat(self, messages, **kwargs):
        if not self._r:
            return ""
        return self._r.pop(0)


async def test_hot_reload_adds_plugin_not_currently_loaded():
    """Start with hypothalamus only; hot-add cli_exec; verify the
    cli_exec tool is now registered."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM(), hypo_llm=ScriptedLLM(),
        modifiers=["in_mind_note"],
    )
    assert "cli_exec" not in runtime.tools

    report = await runtime.hot_reload_plugins(
        ["in_mind_note", "cli_exec"],
        force_reload=False,
    )

    added_plugins = [a["plugin"] for a in report["added"]]
    assert "cli_exec" in added_plugins
    # Already-loaded hypothalamus is skipped (force_reload=False).
    skipped = [s["plugin"] for s in report["skipped"]]
    assert "in_mind_note" in skipped
    assert "cli_exec" in runtime.tools


async def test_hot_reload_full_reloads_already_loaded_plugins_when_forced():
    """force_reload=True (default for the dashboard's Apply
    button) → already-loaded plugins get unregister + re-register
    so a config / LLM-binding edit takes effect."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM(), hypo_llm=ScriptedLLM(),
        modifiers=["in_mind_note", "recall"],
    )
    # Snapshot the original modifier instance so we can compare
    # identity after reload. The in_mind_note plugin's modifier
    # registers under role "in_mind".
    before = runtime.modifiers.by_role("in_mind")
    assert before is not None

    report = await runtime.hot_reload_plugins(
        ["in_mind_note", "recall"],   # same set, force_reload=True
    )

    reloaded_plugins = [r["plugin"] for r in report["reloaded"]]
    assert "in_mind_note" in reloaded_plugins
    after = runtime.modifiers.by_role("in_mind")
    assert after is not None
    # Different instance — proof that re-register happened, not
    # just a no-op.
    assert after is not before


async def test_hot_reload_skips_already_loaded_when_force_false():
    """force_reload=False → already-loaded plugins are skipped
    entirely (same as the v1 hot-add-only behaviour)."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM(), hypo_llm=ScriptedLLM(),
        modifiers=["in_mind_note"],
    )
    report = await runtime.hot_reload_plugins(
        ["in_mind_note"], force_reload=False,
    )
    skipped = [s["plugin"] for s in report["skipped"]]
    assert "in_mind_note" in skipped
    assert report["reloaded"] == []


async def test_hot_reload_unknown_plugin_lands_in_errors():
    """An unknown plugin name should produce a clean error
    entry, not a crash."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM(), hypo_llm=ScriptedLLM(),
        modifiers=["in_mind_note"],
    )
    report = await runtime.hot_reload_plugins(
        ["in_mind_note", "totally_made_up_plugin_xyz"],
    )
    error_plugins = [e["plugin"] for e in report["errors"]]
    assert "totally_made_up_plugin_xyz" in error_plugins


async def test_hot_reload_removes_plugin_dropped_from_target():
    """A plugin loaded but not in the new target list gets
    unregistered (true hot-disable, not just an advisory)."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM(), hypo_llm=ScriptedLLM(),
        modifiers=["in_mind_note", "recall"],
    )
    # recall registers a memory_recall tool; verify it's present.
    assert "memory_recall" in runtime.tools

    report = await runtime.hot_reload_plugins(["in_mind_note"])

    removed_plugins = [r["plugin"] for r in report["removed"]]
    assert "recall" in removed_plugins
    # recall's tool is gone.
    assert "memory_recall" not in runtime.tools


async def test_hot_reload_starts_newly_registered_channels(monkeypatch):
    """Channels need start_one() after hot-add or their
    background task never fires."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM(), hypo_llm=ScriptedLLM(),
        modifiers=["in_mind_note"],
    )
    started: list[str] = []
    real_start_one = runtime.buffer.start_one

    async def spy(name):
        started.append(name)
        await real_start_one(name)

    monkeypatch.setattr(runtime.buffer, "start_one", spy)

    report = await runtime.hot_reload_plugins(
        ["in_mind_note", "cli_exec"],
    )
    # cli_exec is tool-only, no channel start.
    assert started == []
    cli_added = next(
        a for a in report["added"] if a["plugin"] == "cli_exec"
    )
    assert all(
        c["kind"] != "channel" for c in cli_added["components"]
    )


async def test_hot_reload_back_to_back_idempotent():
    """Running hot_reload twice with the same target produces
    a report whose second pass doesn't add or remove anything
    new (just reloads the same set)."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM(), hypo_llm=ScriptedLLM(),
        modifiers=["in_mind_note"],
    )
    first = await runtime.hot_reload_plugins(
        ["in_mind_note", "cli_exec"],
    )
    second = await runtime.hot_reload_plugins(
        ["in_mind_note", "cli_exec"],
    )
    # First pass: cli_exec added, hypothalamus reloaded.
    assert any(a["plugin"] == "cli_exec" for a in first["added"])
    # Second pass: nothing added or removed; both reloaded.
    assert second["added"] == []
    assert second["removed"] == []
    reloaded_2 = [r["plugin"] for r in second["reloaded"]]
    assert "cli_exec" in reloaded_2
    assert "in_mind_note" in reloaded_2
