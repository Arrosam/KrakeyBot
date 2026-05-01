"""ANSI color helpers — wrap when enabled, pass through when disabled."""
from krakey.runtime.console import colors


def test_cyan_wraps_when_enabled(monkeypatch):
    monkeypatch.setattr(colors, "_ENABLED", True)
    out = colors.cyan("hello")
    assert out.startswith("\033[")
    assert "hello" in out
    assert out.endswith("\033[0m")


def test_green_wraps_when_enabled(monkeypatch):
    monkeypatch.setattr(colors, "_ENABLED", True)
    out = colors.green("ok")
    assert "ok" in out
    assert out.endswith("\033[0m")


def test_disabled_returns_plain_text(monkeypatch):
    monkeypatch.setattr(colors, "_ENABLED", False)
    assert colors.cyan("hi") == "hi"
    assert colors.green("hi") == "hi"
    assert colors.yellow("hi") == "hi"


def test_yellow_wraps_when_enabled(monkeypatch):
    monkeypatch.setattr(colors, "_ENABLED", True)
    out = colors.yellow("hypo")
    assert "hypo" in out
    assert out.endswith("\033[0m")


def test_magenta_wraps_when_enabled(monkeypatch):
    monkeypatch.setattr(colors, "_ENABLED", True)
    out = colors.magenta("recall")
    assert "recall" in out
    assert out.endswith("\033[0m")


def test_magenta_disabled_passthrough(monkeypatch):
    monkeypatch.setattr(colors, "_ENABLED", False)
    assert colors.magenta("recall") == "recall"


def test_no_color_env_disables(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert colors._compute_enabled() is False


def test_non_tty_disables(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)

    class FakeStream:
        def isatty(self):
            return False
    monkeypatch.setattr(colors.sys, "stdout", FakeStream())
    assert colors._compute_enabled() is False


def test_tty_and_no_no_color_enables(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)

    class FakeStream:
        def isatty(self):
            return True
    monkeypatch.setattr(colors.sys, "stdout", FakeStream())
    assert colors._compute_enabled() is True
