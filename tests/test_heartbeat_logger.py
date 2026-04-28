"""HeartbeatLogger: format/colour layer separated from Runtime."""
import io
import sys

import pytest

from krakey.runtime.console import colors
from krakey.runtime.console.heartbeat_logger import HeartbeatLogger


def _capture(monkeypatch):
    out = io.StringIO()
    err = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    return out, err


def test_set_heartbeat_then_hb_includes_id(monkeypatch):
    out, _ = _capture(monkeypatch)
    monkeypatch.setattr(colors, "_ENABLED", False)
    log = HeartbeatLogger()
    log.set_heartbeat(7)
    log.hb("stimuli=2")
    assert "[HB #7] stimuli=2" in out.getvalue()


def test_hb_thought_uses_cyan_when_colors_enabled(monkeypatch):
    out, _ = _capture(monkeypatch)
    monkeypatch.setattr(colors, "_ENABLED", True)
    log = HeartbeatLogger()
    log.set_heartbeat(3)
    log.hb_thought("decision", "  hello world  ")
    rendered = out.getvalue()
    assert "[HB #3] decision: hello world" in rendered
    assert rendered.startswith("\033[")


def test_hypo_uses_yellow(monkeypatch):
    out, _ = _capture(monkeypatch)
    monkeypatch.setattr(colors, "_ENABLED", True)
    log = HeartbeatLogger()
    log.hypo("tool_calls=2")
    rendered = out.getvalue()
    assert "[hypo] tool_calls=2" in rendered
    assert rendered.startswith("\033[")


def test_internal_uses_magenta(monkeypatch):
    """memory_recall and other internal-only tools render magenta to
    visually distinguish from green outward chat."""
    out, _ = _capture(monkeypatch)
    monkeypatch.setattr(colors, "_ENABLED", True)
    log = HeartbeatLogger()
    log.internal("memory_recall", "Recall result for 'apple'...")
    rendered = out.getvalue()
    assert "[memory_recall] Recall result" in rendered
    assert rendered.startswith("\033[35m")  # magenta


def test_chat_uses_green(monkeypatch):
    out, _ = _capture(monkeypatch)
    monkeypatch.setattr(colors, "_ENABLED", True)
    log = HeartbeatLogger()
    log.chat("action", "Hi there!")
    rendered = out.getvalue()
    assert "[action] Hi there!" in rendered
    assert rendered.startswith("\033[")


def test_runtime_error_goes_plain(monkeypatch):
    out, _ = _capture(monkeypatch)
    monkeypatch.setattr(colors, "_ENABLED", True)
    log = HeartbeatLogger()
    log.runtime_error("auto_ingest error: boom")
    rendered = out.getvalue()
    assert "[runtime] auto_ingest error: boom" in rendered
    # no ANSI wrapping
    assert "\033[" not in rendered


def test_warn_goes_to_stderr(monkeypatch):
    out, err = _capture(monkeypatch)
    monkeypatch.setattr(colors, "_ENABLED", False)
    log = HeartbeatLogger()
    log.set_heartbeat(5)
    log.hb_warn("force-sleep threshold reached (fatigue=130%)")
    assert err.getvalue() and "force-sleep" in err.getvalue()
    assert out.getvalue() == ""


def test_dispatch_uses_yellow(monkeypatch):
    out, _ = _capture(monkeypatch)
    monkeypatch.setattr(colors, "_ENABLED", True)
    log = HeartbeatLogger()
    log.dispatch("action ← 'do thing' (adrenalin)")
    rendered = out.getvalue()
    assert "[dispatch] action" in rendered
    assert rendered.startswith("\033[")
