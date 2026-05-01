"""Idle runtime: Runtime starts cleanly when self_llm is None.

User-facing requirement (Samuel 2026-04-29): a fresh install with no
chat LLM configured should NOT crash on `krakey run`. The runtime
comes up, channels start (so the dashboard is reachable), but the
heartbeat loop never ticks. The user fixes the LLM via the dashboard
or by re-running `krakey onboard`, then restarts.
"""
from __future__ import annotations

import asyncio

import pytest

from tests._runtime_helpers import (
    NullEmbedder,
    ScriptedLLM,
    build_runtime_with_fakes,
)


async def test_runtime_idle_mode_skips_heartbeat(monkeypatch):
    """When self_llm is None the run loop never invokes _heartbeat."""
    rt = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]),
        hypo_llm=ScriptedLLM([]),
        embedder=NullEmbedder(),
    )
    rt.self_llm = None  # post-construct override

    heartbeat_called = 0
    orig_hb = rt._heartbeat

    async def _track():
        nonlocal heartbeat_called
        heartbeat_called += 1
        await orig_hb()

    rt._heartbeat = _track   # type: ignore[assignment]

    # Make the idle-mode sleep return faster so the test doesn't drag.
    real_sleep = asyncio.sleep

    async def _fast_sleep(_delay):
        rt._stop = True
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    await rt.run()
    assert heartbeat_called == 0


async def test_build_runtime_from_config_allows_no_chat_tag(tmp_path):
    """`build_runtime_from_config` no longer raises when
    `core_purposes.self_thinking` is unbound — it returns a Runtime
    with `self_llm=None` so the CLI can start the idle loop."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "llm:\n"
        "  providers: {}\n"
        "  tags: {}\n"
        "  core_purposes: {}\n"
        "plugins: []\n"
        "graph_memory:\n"
        "  db_path: ':memory:'\n",
        encoding="utf-8",
    )
    from krakey.main import build_runtime_from_config

    rt = build_runtime_from_config(str(cfg))
    assert rt.self_llm is None
