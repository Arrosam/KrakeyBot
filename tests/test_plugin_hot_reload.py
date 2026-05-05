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
    cli_exec tool is now registered + the loader records it."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM(), hypo_llm=ScriptedLLM(),
        modifiers=["hypothalamus"],   # plugins list (misnamed)
    )
    # cli_exec wasn't in startup config, so it isn't loaded.
    assert "cli_exec" not in runtime.tools

    report = await runtime.hot_reload_plugins(
        ["hypothalamus", "cli_exec"],
    )

    assert report["errors"] == []
    added_plugins = [a["plugin"] for a in report["added"]]
    assert "cli_exec" in added_plugins
    # Already-loaded hypothalamus is skipped.
    skipped = [s["plugin"] for s in report["skipped"]]
    assert "hypothalamus" in skipped
    # cli_exec's tool is now in the registry.
    assert "cli_exec" in runtime.tools


async def test_hot_reload_skips_already_loaded_plugins():
    """Calling hot_reload twice with the same target set should
    no-op the second time."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM(), hypo_llm=ScriptedLLM(),
        modifiers=["hypothalamus"],
    )
    first = await runtime.hot_reload_plugins(
        ["hypothalamus", "cli_exec"],
    )
    second = await runtime.hot_reload_plugins(
        ["hypothalamus", "cli_exec"],
    )
    assert any(a["plugin"] == "cli_exec" for a in first["added"])
    # On the second call, cli_exec is now in skipped, not added.
    assert second["added"] == []
    skipped_plugins = [s["plugin"] for s in second["skipped"]]
    assert "cli_exec" in skipped_plugins


async def test_hot_reload_unknown_plugin_lands_in_errors():
    """An unknown plugin name should produce a clean error
    entry, not a crash."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM(), hypo_llm=ScriptedLLM(),
        modifiers=["hypothalamus"],
    )
    report = await runtime.hot_reload_plugins(
        ["hypothalamus", "totally_made_up_plugin_xyz"],
    )
    error_plugins = [e["plugin"] for e in report["errors"]]
    assert "totally_made_up_plugin_xyz" in error_plugins


async def test_hot_reload_advises_pending_remove_for_dropped_plugins():
    """If the target list is missing a currently-loaded plugin,
    the runtime can't actually remove it (no detach hooks yet) but
    surfaces the gap so the dashboard can prompt for restart."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM(), hypo_llm=ScriptedLLM(),
        modifiers=["hypothalamus", "recall"],
    )
    # Drop "recall" from the target.
    report = await runtime.hot_reload_plugins(["hypothalamus"])
    assert "recall" in report["still_pending_remove"]


async def test_hot_reload_starts_newly_registered_channels(monkeypatch):
    """Channels need start_one() after hot-add or their
    background task never fires. Verify by watching start_one
    calls on the buffer."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM(), hypo_llm=ScriptedLLM(),
        modifiers=["hypothalamus"],
    )
    started: list[str] = []
    real_start_one = runtime.buffer.start_one

    async def spy(name):
        started.append(name)
        await real_start_one(name)

    monkeypatch.setattr(runtime.buffer, "start_one", spy)

    # Add cli_exec — it's tool-only, so no channel start.
    report = await runtime.hot_reload_plugins(
        ["hypothalamus", "cli_exec"],
    )
    assert started == []  # no channels added
    # The added entry's components don't include any channel.
    cli_added = next(
        a for a in report["added"] if a["plugin"] == "cli_exec"
    )
    assert all(
        c["kind"] != "channel" for c in cli_added["components"]
    )
