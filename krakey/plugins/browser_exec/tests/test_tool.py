"""Unit tests for ``browser_exec`` plugin.

Run from repo root:

    pytest krakey/plugins/browser_exec

Step-1 skeleton: this file pins the public surface (factory exists,
class shape, schema shell, ``execute`` raises ``NotImplementedError``)
so subsequent commits that fill in ``snippets.py`` and the dispatch
body have a stable contract to extend. Real dispatch tests land in
step 3.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from krakey.plugins.browser_exec.tool import (
    ACTIONS,
    BROWSERS,
    DEFAULT_BROWSER,
    DEFAULT_HEADLESS,
    DEFAULT_PYTHON_CMD,
    DEFAULT_TIMEOUT_S,
    OUTPUT_FORMATS,
    BrowserExecTool,
    build_tool,
)


# =====================================================================
# Plugin metadata sanity
# =====================================================================


def test_meta_yaml_parses_and_declares_one_tool():
    """Catches accidental meta.yaml drift (e.g. missing factory_attr,
    wrong kind enum). Uses the same parser the runtime uses so the
    failure mode here matches what an operator would see."""
    import yaml

    meta_path = Path(__file__).resolve().parent.parent / "meta.yaml"
    data = yaml.safe_load(meta_path.read_text(encoding="utf-8"))

    assert data["name"] == "browser_exec"
    assert isinstance(data.get("description"), str) and data["description"]

    components = data["components"]
    assert isinstance(components, list) and len(components) == 1
    comp = components[0]
    assert comp["kind"] == "tool"
    assert comp["factory_module"] == "krakey.plugins.browser_exec.tool"
    assert comp["factory_attr"] == "build_tool"

    # config_schema advertises the four documented fields.
    schema = data["config_schema"]
    field_names = {entry["field"] for entry in schema}
    assert field_names == {
        "python_cmd", "headless",
        "default_browser", "default_timeout_s",
    }


# =====================================================================
# Module-level constants — pinned so other modules / tests can
# import them safely
# =====================================================================


def test_default_constants_have_expected_shapes():
    assert DEFAULT_PYTHON_CMD == "python"
    assert DEFAULT_TIMEOUT_S == 30.0
    assert DEFAULT_BROWSER == "chromium"
    assert DEFAULT_HEADLESS is True
    assert set(BROWSERS) == {"chromium", "firefox", "webkit"}
    assert "a11y" in OUTPUT_FORMATS  # default format
    assert set(OUTPUT_FORMATS) == {"a11y", "text", "html"}
    assert set(ACTIONS) == {
        "navigate", "click", "type", "press",
        "scroll", "wait_for", "screenshot",
    }


# =====================================================================
# Factory + tool shape (skeleton-level, real dispatch in step 3)
# =====================================================================


class _FakeCtx:
    """Minimal duck-typed PluginContext stand-in (matches the pattern
    used in gui_exec's factory tests). The factory only touches
    ``.config`` and ``.environment`` so a small namespace suffices."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def environment(self, _name: str):  # pragma: no cover — stub
        raise AssertionError("env_resolver not exercised in skeleton tests")


def test_build_tool_returns_browser_exec_tool_instance():
    tool = build_tool(_FakeCtx())  # type: ignore[arg-type]
    assert isinstance(tool, BrowserExecTool)


def test_tool_static_metadata():
    tool = build_tool(_FakeCtx())  # type: ignore[arg-type]
    assert tool.name == "browser_exec"

    schema = tool.parameters_schema
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"env", "start_url", "actions"}

    props = schema["properties"]
    for k in (
        "env", "start_url", "actions",
        "timeout_s", "output", "return_screenshot",
        "headless", "browser",
    ):
        assert k in props, f"missing schema property: {k}"

    # Description names the env names + the python_cmd config field +
    # the playwright dep so Self can describe the failure modes.
    desc = tool.description
    assert "local" in desc
    assert "sandbox" in desc
    assert "playwright" in desc.lower()
    assert "python_cmd" in desc
    assert "a11y" in desc.lower()


def test_execute_is_not_yet_implemented():
    """Skeleton commit: dispatch body lands in step 3. Pinning this
    here means the next commit's tests *must* update this file —
    can't silently regress to a no-op stimulus."""
    import asyncio

    tool = build_tool(_FakeCtx())  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        asyncio.run(tool.execute(
            "x",
            {
                "env": "local",
                "start_url": "https://example.com",
                "actions": [],
            },
        ))
