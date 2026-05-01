"""Regression: tests must never write into production workspace paths.

History: the web_chat reply tool appended "Hi there!" / "go" etc.
(from test_main_loop.py fixtures) to the real user-facing
``workspace/data/web_chat.jsonl``. Root cause: FilePluginConfigStore
reads ``workspace/plugin-configs/web_chat.yaml`` first and only falls
back to the test helper's `legacy_plugins` dict if the file is
missing — so the helper's history_path override was silently shadowed
by the production YAML.

These tests pin the fix: the helper now carves a tmpdir for plugin
configs + self-model + chat history, and the Runtime's writers land
there exclusively.
"""
import json
from pathlib import Path

import pytest

from tests._runtime_helpers import (
    NullEmbedder, ScriptedLLM, build_runtime_with_fakes,
)


# Absolute prod paths that a test MUST NEVER touch. Rooted off the
# repo so the check works whatever the cwd happens to be.
_PROD_ROOT = Path(__file__).resolve().parent.parent
_FORBIDDEN = [
    _PROD_ROOT / "workspace" / "data" / "web_chat.jsonl",
    _PROD_ROOT / "workspace" / "self_model.yaml",
    _PROD_ROOT / "workspace" / "data" / "in_mind.json",
]


def _baseline_sizes() -> dict[Path, int]:
    """Snapshot current file sizes so the test can assert they didn't
    grow during the runtime build + web_chat dispatch."""
    out: dict[Path, int] = {}
    for p in _FORBIDDEN:
        out[p] = p.stat().st_size if p.exists() else -1
    return out


def _assert_unchanged(before: dict[Path, int]) -> None:
    for p, size in before.items():
        after = p.stat().st_size if p.exists() else -1
        assert after == size, (
            f"{p} changed during test (before={size}, after={after}); "
            f"a write path is escaping the tmpdir isolation"
        )


async def test_runtime_construction_does_not_touch_prod_paths(tmp_path):
    """Just building a Runtime must not append to web_chat.jsonl or
    rewrite self_model.yaml. Regression for the leak we just fixed."""
    before = _baseline_sizes()
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    _ = runtime  # suppress unused warning
    _assert_unchanged(before)


async def test_web_chat_reply_writes_to_tmpdir_not_prod(tmp_path):
    """Dispatching the web_chat_reply tool must write only to the
    helper's tmpdir chat JSONL, never to workspace/data/web_chat.jsonl.

    Web chat is owned by the dashboard plugin now; we reach into the
    registered tool's history (the same object the dashboard
    channel's chat-WS handler would push to) instead of touching a
    runtime field that no longer exists.
    """
    before = _baseline_sizes()
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    tent = runtime.tools.get("web_chat_reply")
    history = tent._history  # noqa: SLF001 — test reaches into plugin
    await history.append("bot", "this must not leak")
    # Prod files untouched
    _assert_unchanged(before)
    # And the actual target is the tmpdir the helper provisioned
    target = Path(history.path)
    assert target.exists(), "bot message never hit disk"
    assert "krakey_test_chat_" in str(target), (
        f"expected history_path under a test tmpdir; got {target}"
    )
    # Payload round-trips
    lines = target.read_text(encoding="utf-8").splitlines()
    assert any("this must not leak" in ln for ln in lines)


def test_helper_provisions_isolated_plugin_configs_root(tmp_path):
    """The helper must set RuntimeDeps.plugin_configs_root to a
    tmpdir — without that, the runtime + dashboard adapter would read
    the prod YAML and silently shadow the helper's overrides."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    root = runtime._plugin_configs_root
    assert "krakey_test_plugcfg_" in str(root), (
        f"plugin_configs_root not isolated; got {root}"
    )
    # And it's isolated from prod: if the plugin loader materialized
    # per-plugin YAMLs, they went into this tmpdir, not
    # workspace/plugin-configs/. (We don't assert emptiness — plugin
    # discovery legitimately writes scaffold files here.)
    prod_root = _PROD_ROOT / "workspace" / "plugin-configs"
    assert root != prod_root
