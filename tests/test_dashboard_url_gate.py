"""Edge tests — KRAKEY_DAEMON_MODE gate on the dashboard URL log line.

Spec: the `url_full` assignment + the KRAKEY_REDACT_TOKEN_LOG if/else inside
`_start_dashboard_server` will be wrapped in
`if not os.environ.get("KRAKEY_DAEMON_MODE"):`.

Tests marked RED (expected to fail against current code) are noted inline.
They are RED because the gate does not exist yet; they turn GREEN once the
wrapping `if` is added.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Harness helpers
# ---------------------------------------------------------------------------

PORT = 18765
HOST = "127.0.0.1"


def _make_runtime():
    """Minimal runtime stub with a log sub-stub whose hb/runtime_error calls
    are MagicMocks so we can assert against call_args_list."""
    events_stub = SimpleNamespace(subscribe=lambda *a, **k: None)
    log_stub = SimpleNamespace(
        hb=MagicMock(),
        runtime_error=MagicMock(),
    )
    return SimpleNamespace(events=events_stub, log=log_stub)


def _ctx(runtime, tmp_path: Path):
    """Build a minimal PluginContext-like namespace."""
    history_path = str(tmp_path / "web_chat.jsonl")
    return SimpleNamespace(
        plugin_name="dashboard",
        config={
            "history_path": history_path,
            "host": HOST,
            "port": PORT,
        },
        services={"runtime": runtime},
        plugin_cache={},
        deps=SimpleNamespace(config_path=None, plugin_configs_root=None),
    )


def _hb_messages(runtime) -> list[str]:
    """Return all first positional args passed to runtime.log.hb."""
    return [
        call.args[0]
        for call in runtime.log.hb.call_args_list
        if call.args
    ]


# ---------------------------------------------------------------------------
# Monkeypatch fixture — replaces every collaborator that would touch the
# filesystem or bind a real socket, while leaving load_or_create_token real
# (it is cheap + file-based and its result is what we want to assert on).
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_collaborators(monkeypatch):
    """Patch the module-local names that `_start_dashboard_server` imports
    inside its body.  All patches target the import paths used by that
    function, not the definition sites, because they are lazy local imports.
    """

    # Fake ThreadedDashboardServer: stores port, no-op start.
    class _FakeServer:
        def __init__(self, app, *, host="127.0.0.1", port=8765,
                     log_level="warning"):
            self.port = port

        def start(self):
            pass

    monkeypatch.setattr(
        "krakey.plugins.dashboard.threaded_server.ThreadedDashboardServer",
        _FakeServer,
    )

    # Fake create_app (aliased as create_dashboard_app inside the function).
    monkeypatch.setattr(
        "krakey.plugins.dashboard.app_factory.create_app",
        lambda **kwargs: object(),
    )

    # Fake EventBroadcaster: accepts one positional arg (the event bus).
    class _FakeBroadcaster:
        def __init__(self, bus):
            pass

    monkeypatch.setattr(
        "krakey.plugins.dashboard.events.EventBroadcaster",
        _FakeBroadcaster,
    )

    # Fake make_stimulus_read_handler: accepts one arg (history), returns
    # a no-op handler.
    monkeypatch.setattr(
        "krakey.plugins.dashboard.web_chat.read_receipts.make_stimulus_read_handler",
        lambda history: (lambda *a, **k: None),
    )

    # Fake LogCapture: no-op install.
    class _FakeLogCapture:
        def install(self):
            pass

        def uninstall(self):
            pass

    monkeypatch.setattr(
        "krakey.plugins.dashboard.log_capture.LogCapture",
        _FakeLogCapture,
    )

    # Suppress the color side-effect (set NO_COLOR so the try-block is
    # skipped entirely rather than importing console.colors).
    monkeypatch.setenv("NO_COLOR", "1")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Guarantee both gate vars are absent at the start of every test so
    tests cannot bleed env state into each other."""
    monkeypatch.delenv("KRAKEY_DAEMON_MODE", raising=False)
    monkeypatch.delenv("KRAKEY_REDACT_TOKEN_LOG", raising=False)


# ---------------------------------------------------------------------------
# Helper: read the token that load_or_create_token wrote for a given
# history_path so assertions can check for its exact value.
# ---------------------------------------------------------------------------

def _token_for(ctx) -> str:
    history_path = ctx.config["history_path"]
    token_path = Path(history_path).parent / "dashboard.token"
    return token_path.read_text().strip()


# ---------------------------------------------------------------------------
# 1. POSITIVE — foreground default
#    Both env vars absent → one-click URL must be logged.
# ---------------------------------------------------------------------------

def test_foreground_oneclick_url_logged(monkeypatch, tmp_path):
    """POSITIVE: KRAKEY_DAEMON_MODE unset, REDACT unset → hb is called with
    the full one-click URL including the token."""
    from krakey.plugins.dashboard import build_channel

    runtime = _make_runtime()
    ctx = _ctx(runtime, tmp_path)

    build_channel(ctx)

    token = _token_for(ctx)
    messages = _hb_messages(runtime)
    expected_fragment = (
        f"dashboard URL (one-click): http://{HOST}:{PORT}/?token={token}"
    )
    assert any(expected_fragment in m for m in messages), (
        f"Expected one-click URL in log messages. Got: {messages}"
    )


# ---------------------------------------------------------------------------
# 2. GATE — daemon mode suppresses the URL block entirely
#    RED against current code (gate not yet implemented).
# ---------------------------------------------------------------------------

def test_daemon_mode_suppresses_oneclick_url(monkeypatch, tmp_path):
    """GATE (RED against current code): KRAKEY_DAEMON_MODE='1' →
    no log message containing 'dashboard URL' must be emitted."""
    monkeypatch.setenv("KRAKEY_DAEMON_MODE", "1")

    from krakey.plugins.dashboard import build_channel

    runtime = _make_runtime()
    ctx = _ctx(runtime, tmp_path)

    build_channel(ctx)

    messages = _hb_messages(runtime)
    url_messages = [m for m in messages if "dashboard URL" in m]
    assert url_messages == [], (
        f"Expected no 'dashboard URL' log in daemon mode. Got: {url_messages}"
    )


# ---------------------------------------------------------------------------
# 3. REDACT preserved in foreground
#    KRAKEY_DAEMON_MODE unset, KRAKEY_REDACT_TOKEN_LOG='1' →
#    redacted URL + redaction note logged; NOT the one-click form.
# ---------------------------------------------------------------------------

def test_foreground_redact_logs_redacted_url_not_oneclick(monkeypatch, tmp_path):
    """POSITIVE/REDACT: foreground + REDACT=1 → redacted form logged, NOT
    the one-click /?token= form."""
    monkeypatch.setenv("KRAKEY_REDACT_TOKEN_LOG", "1")

    from krakey.plugins.dashboard import build_channel

    runtime = _make_runtime()
    ctx = _ctx(runtime, tmp_path)

    build_channel(ctx)

    messages = _hb_messages(runtime)

    # Redacted form must appear (contains <see and the token path hint).
    assert any("dashboard URL:" in m and "<see" in m for m in messages), (
        f"Expected redacted URL message. Got: {messages}"
    )
    # Redaction note must appear.
    assert any("redacted" in m.lower() for m in messages), (
        f"Expected redaction note. Got: {messages}"
    )
    # One-click form must NOT appear.
    assert not any("(one-click)" in m for m in messages), (
        f"One-click URL must not be logged when REDACT=1. Got: {messages}"
    )


# ---------------------------------------------------------------------------
# 4. GATE wins over REDACT
#    KRAKEY_DAEMON_MODE='1', KRAKEY_REDACT_TOKEN_LOG='1' → no URL logged.
#    RED against current code.
# ---------------------------------------------------------------------------

def test_daemon_mode_wins_over_redact(monkeypatch, tmp_path):
    """GATE (RED against current code): daemon mode takes priority — even
    with REDACT=1, no 'dashboard URL' message must appear."""
    monkeypatch.setenv("KRAKEY_DAEMON_MODE", "1")
    monkeypatch.setenv("KRAKEY_REDACT_TOKEN_LOG", "1")

    from krakey.plugins.dashboard import build_channel

    runtime = _make_runtime()
    ctx = _ctx(runtime, tmp_path)

    build_channel(ctx)

    messages = _hb_messages(runtime)
    url_messages = [m for m in messages if "dashboard URL" in m]
    assert url_messages == [], (
        f"Daemon mode must suppress URL even when REDACT=1. Got: {url_messages}"
    )


# ---------------------------------------------------------------------------
# 5a. BOUNDARY — listening line always logged in foreground
# ---------------------------------------------------------------------------

def test_listening_line_logged_in_foreground(monkeypatch, tmp_path):
    """BOUNDARY: the unconditional 'dashboard listening on' line must appear
    regardless of env vars — foreground case."""
    from krakey.plugins.dashboard import build_channel

    runtime = _make_runtime()
    ctx = _ctx(runtime, tmp_path)

    build_channel(ctx)

    messages = _hb_messages(runtime)
    expected = f"dashboard listening on http://{HOST}:{PORT}"
    assert any(expected in m for m in messages), (
        f"Expected listening line in foreground. Got: {messages}"
    )


# ---------------------------------------------------------------------------
# 5b. BOUNDARY — listening line always logged in daemon mode
#     Validates the gate only wraps the URL block, NOT the listening line.
# ---------------------------------------------------------------------------

def test_listening_line_logged_in_daemon_mode(monkeypatch, tmp_path):
    """BOUNDARY: the 'dashboard listening on' line must also appear in daemon
    mode — the gate must not swallow the line that precedes the URL block."""
    monkeypatch.setenv("KRAKEY_DAEMON_MODE", "1")

    from krakey.plugins.dashboard import build_channel

    runtime = _make_runtime()
    ctx = _ctx(runtime, tmp_path)

    build_channel(ctx)

    messages = _hb_messages(runtime)
    expected = f"dashboard listening on http://{HOST}:{PORT}"
    assert any(expected in m for m in messages), (
        f"Expected listening line even in daemon mode. Got: {messages}"
    )


# ---------------------------------------------------------------------------
# 6a. BOUNDARY — empty-string KRAKEY_DAEMON_MODE treated as falsy
#     os.environ.get("KRAKEY_DAEMON_MODE") returns "" → not "" is True →
#     URL block RUNS (one-click logged).
# ---------------------------------------------------------------------------

def test_empty_string_daemon_mode_treated_as_falsy(monkeypatch, tmp_path):
    """BOUNDARY: KRAKEY_DAEMON_MODE='' (set but empty) must behave like
    unset — URL block runs, one-click URL is logged."""
    monkeypatch.setenv("KRAKEY_DAEMON_MODE", "")

    from krakey.plugins.dashboard import build_channel

    runtime = _make_runtime()
    ctx = _ctx(runtime, tmp_path)

    build_channel(ctx)

    token = _token_for(ctx)
    messages = _hb_messages(runtime)
    expected_fragment = (
        f"dashboard URL (one-click): http://{HOST}:{PORT}/?token={token}"
    )
    assert any(expected_fragment in m for m in messages), (
        f"Empty KRAKEY_DAEMON_MODE must not suppress URL. Got: {messages}"
    )


# ---------------------------------------------------------------------------
# 6b. BOUNDARY — non-empty non-"1" value also suppresses
#     Documents that ANY non-empty value suppresses, not just "1".
#     RED against current code.
# ---------------------------------------------------------------------------

def test_nonzero_daemon_mode_value_also_suppresses(monkeypatch, tmp_path):
    """BOUNDARY (RED against current code): KRAKEY_DAEMON_MODE='0' is
    non-empty → os.environ.get returns '0' → truthy string → block suppressed.
    (Contrast with empty string above.)"""
    monkeypatch.setenv("KRAKEY_DAEMON_MODE", "0")

    from krakey.plugins.dashboard import build_channel

    runtime = _make_runtime()
    ctx = _ctx(runtime, tmp_path)

    build_channel(ctx)

    messages = _hb_messages(runtime)
    url_messages = [m for m in messages if "dashboard URL" in m]
    assert url_messages == [], (
        f"Any non-empty KRAKEY_DAEMON_MODE value must suppress URL. "
        f"Got: {url_messages}"
    )


# ---------------------------------------------------------------------------
# 7. NEGATIVE / non-idempotent safety
#    With KRAKEY_DAEMON_MODE='1', build_channel must still return a
#    non-None channel with a server attached and must not raise.
#    RED against current code.
# ---------------------------------------------------------------------------

def test_daemon_mode_does_not_alter_server_lifecycle(monkeypatch, tmp_path):
    """NEGATIVE: the gate is a log-only change. With KRAKEY_DAEMON_MODE='1',
    build_channel must still return a channel with `_server` attached and
    must not raise any exception."""
    monkeypatch.setenv("KRAKEY_DAEMON_MODE", "1")

    from krakey.plugins.dashboard import build_channel

    runtime = _make_runtime()
    ctx = _ctx(runtime, tmp_path)

    channel = build_channel(ctx)

    assert channel is not None, "build_channel must return a channel object"
    assert getattr(channel, "_server", None) is not None, (
        "channel._server must be set even in daemon mode — "
        "the gate must not alter server lifecycle"
    )
    # No exception means runtime.log.runtime_error was never called.
    runtime.log.runtime_error.assert_not_called()
