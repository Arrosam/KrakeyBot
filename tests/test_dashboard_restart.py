"""Dashboard ``_restart_self`` argv construction — three launch modes.

The dashboard's "Apply changes" → restart path re-spawns the process
via subprocess.Popen. The argv shape depends on how Krakey was
launched:

  A. ``python -m krakey``                  → ``python -m krakey ...``
  B. ``krakey`` (console-script entry pt)  → ``<argv[0]>`` (the wrapper)
  C. ``python <path>/krakey/__main__.py``  → ``python <path> ...``

Pre-fix the code lumped B + C together and unconditionally tried
``-m <spec.name>``, which broke mode B because ``__main__.__spec__``
was set to ``None`` by setuptools' generated wrapper, and ``python -m
__main__`` raises ``ValueError: __main__.__spec__ is None`` in the
child process before the runtime can come back up.
"""
from __future__ import annotations

import sys
import types
from typing import Any

import pytest


@pytest.fixture
def _spawn_capture(monkeypatch):
    """Hijack subprocess.Popen + os._exit + time.sleep so
    _restart_self runs to completion in-test without spawning a real
    child or actually exiting the process.
    """
    import krakey.plugins.dashboard as dashboard_mod

    captured: dict[str, Any] = {"args": None, "exited": False}

    class _FakePopen:
        def __init__(self, args, **_kw):
            captured["args"] = list(args)

    def _fake_exit(code):
        captured["exited"] = True
        # Don't actually exit — raise a marker instead so the test
        # stops without killing the pytest process.
        raise SystemExit(code)

    monkeypatch.setattr(
        "subprocess.Popen", _FakePopen,
    )
    monkeypatch.setattr(
        "os._exit", _fake_exit,
    )
    monkeypatch.setattr(
        "time.sleep", lambda _s: None,
    )
    yield captured, dashboard_mod


def _set_main_spec(monkeypatch, *, spec_name: str | None):
    """Plant a fake ``__main__`` module with the given ``__spec__.name``.
    ``spec_name=None`` simulates ``__spec__ is None`` (script launch)."""
    fake_main = types.ModuleType("_fake_main_for_test")
    if spec_name is None:
        fake_main.__spec__ = None
    else:
        # Minimal duck-typed Spec — only ``.name`` is read by the
        # restart code under test.
        fake_main.__spec__ = types.SimpleNamespace(name=spec_name)
    monkeypatch.setitem(sys.modules, "__main__", fake_main)


def _stub_runtime():
    class _Log:
        def hb(self, *_a, **_kw): pass
        def runtime_error(self, *_a, **_kw): pass
    return types.SimpleNamespace(log=_Log())


def test_mode_a_python_m_pkg_respawns_via_dash_m(monkeypatch, _spawn_capture):
    """``python -m krakey`` sets ``__main__.__spec__.name ==
    "krakey.__main__"``. Re-spawn must drop the ``.__main__`` suffix
    and use the package name with ``-m``."""
    captured, dashboard_mod = _spawn_capture
    _set_main_spec(monkeypatch, spec_name="krakey.__main__")
    monkeypatch.setattr(
        sys, "argv",
        ["F:/path/to/krakey/__main__.py", "--flag"],
    )
    with pytest.raises(SystemExit):
        dashboard_mod._restart_self(_stub_runtime())
    assert captured["args"] == [
        sys.executable, "-m", "krakey", "--flag",
    ]
    assert captured["exited"]


def test_mode_b_console_script_respawns_argv0_directly(
    monkeypatch, _spawn_capture,
):
    """``krakey`` (the setuptools entry-point .exe / shebang) sets
    ``__main__.__spec__`` to ``None``. argv[0] is the wrapper path
    and should be re-executed AS-IS (no python prefix) — the wrapper
    is responsible for bootstrapping Python.

    Pre-fix this constructed ``[python, -m, __main__, ...]`` and
    failed in the child with ``ValueError: __main__.__spec__ is None``.
    """
    captured, dashboard_mod = _spawn_capture
    _set_main_spec(monkeypatch, spec_name=None)
    monkeypatch.setattr(
        sys, "argv",
        ["F:/path/.venv/Scripts/krakey.exe", "--flag"],
    )
    with pytest.raises(SystemExit):
        dashboard_mod._restart_self(_stub_runtime())
    # Re-spawn argv as-is, NOT via ``-m __main__``.
    assert captured["args"] == [
        "F:/path/.venv/Scripts/krakey.exe", "--flag",
    ]


def test_mode_b_handles_literal_main_spec_name(
    monkeypatch, _spawn_capture,
):
    """Some launchers leave ``__spec__`` non-None but with
    ``.name == "__main__"``. Treat that the same as B (re-spawn argv
    directly) — never use ``-m __main__``."""
    captured, dashboard_mod = _spawn_capture
    _set_main_spec(monkeypatch, spec_name="__main__")
    monkeypatch.setattr(
        sys, "argv",
        ["F:/path/.venv/Scripts/krakey.exe"],
    )
    with pytest.raises(SystemExit):
        dashboard_mod._restart_self(_stub_runtime())
    args = captured["args"]
    assert "-m" not in args
    assert args == ["F:/path/.venv/Scripts/krakey.exe"]


def test_mode_c_script_path_prepends_interpreter(
    monkeypatch, _spawn_capture,
):
    """``python somescript.py`` — argv[0] is a .py path, __spec__ is
    None. Re-spawn with the interpreter prepended."""
    captured, dashboard_mod = _spawn_capture
    _set_main_spec(monkeypatch, spec_name=None)
    monkeypatch.setattr(
        sys, "argv",
        ["F:/path/krakey/__main__.py", "extra"],
    )
    with pytest.raises(SystemExit):
        dashboard_mod._restart_self(_stub_runtime())
    assert captured["args"] == [
        sys.executable,
        "F:/path/krakey/__main__.py",
        "extra",
    ]
