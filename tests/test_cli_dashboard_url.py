"""Edge tests for lifecycle._print_dashboard_url(repo: Path) -> None.

Spec summary:
  - Best-effort: any exception results in a silent return (no raise, no traceback).
  - Loads cfg via load_config(repo / 'config.yaml'), lazily imported.
  - If cfg.plugins is falsy OR 'dashboard' not in cfg.plugins → silent return, no print.
  - Otherwise reads dashboard plugin config via FilePluginConfigStore.
    Extracts host (default '127.0.0.1'), port (default 8765),
    history_path (default 'workspace/data/web_chat.jsonl').
  - port == 0 → silent return, no print.
  - Derives token file at repo / Path(history_path).parent / 'dashboard.token'.
  - Polls token file up to ~5s (~100 × 0.05s). If non-empty stripped content
    found: prints `dashboard: http://{host}:{port}/?token={token}`.
  - If token file absent / empty after timeout: prints `dashboard: http://{host}:{port}/`.
  - host printed VERBATIM — '0.0.0.0' stays '0.0.0.0', no substitution.
  - Returns None always.

Testing techniques applied:
  1. Positive / equivalence partitioning — happy paths, token present URL.
  2. Boundary value analysis — port 0, empty plugins, None plugins, empty token, missing config.
  3. State transition — token pre-created (first poll wins) vs. absent (timeout fallback).
  4. Error guessing / negative — load_config raises, malformed YAML, unreadable token file,
     missing repo entirely.
"""
from __future__ import annotations

import stat
import sys
import types
from pathlib import Path

import pytest
import yaml

from krakey.plugin_system.config import FilePluginConfigStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(plugins):
    """Return a SimpleNamespace mimicking a Config object with .plugins."""
    return types.SimpleNamespace(plugins=plugins)


def _write_dashboard_config(repo: Path, data: dict) -> None:
    """Write dashboard plugin config at the canonical location."""
    plugin_cfg_path = repo / "workspace" / "plugins" / "dashboard" / "config.yaml"
    plugin_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    plugin_cfg_path.write_text(yaml.dump(data), encoding="utf-8")


def _write_token_file(repo: Path, history_path: str, token: str) -> Path:
    """Create the token file derived from history_path and return its path."""
    token_file = repo / Path(history_path).parent / "dashboard.token"
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(token, encoding="utf-8")
    return token_file


def _derived_token_path(repo: Path, history_path: str) -> Path:
    """Mirror the spec's derivation of the token file path."""
    return repo / Path(history_path).parent / "dashboard.token"


def _call(repo: Path) -> None:
    """Import and call _print_dashboard_url; deferred to avoid early import."""
    from krakey.cli.lifecycle import _print_dashboard_url
    _print_dashboard_url(repo)


# ===========================================================================
# 1. Positive tests — equivalence partitioning / happy paths
# ===========================================================================

class TestPositiveDashboardEnabled:
    """dashboard in plugins + token present → token URL printed."""

    def test_token_url_printed_when_token_file_present(
        self, monkeypatch, tmp_path, capsys
    ):
        """POSITIVE: dashboard enabled, token file pre-created with a valid token.
        Expected output: 'dashboard: http://127.0.0.1:8765/?token=abc123'"""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        _write_token_file(tmp_path, "workspace/data/web_chat.jsonl", "abc123")
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        assert out == "dashboard: http://127.0.0.1:8765/?token=abc123"

    def test_token_url_uses_correct_default_host_and_port(
        self, monkeypatch, tmp_path, capsys
    ):
        """POSITIVE: no plugin config file → all defaults used (127.0.0.1:8765)."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        _write_token_file(tmp_path, "workspace/data/web_chat.jsonl", "mytoken")
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        assert out.startswith("dashboard: http://127.0.0.1:8765/")
        assert "?token=mytoken" in out

    def test_custom_host_and_port_in_token_url(
        self, monkeypatch, tmp_path, capsys
    ):
        """POSITIVE: custom host + port from plugin config → reflected in URL."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        _write_dashboard_config(tmp_path, {
            "host": "192.168.1.10",
            "port": 9090,
            "history_path": "workspace/data/web_chat.jsonl",
        })
        _write_token_file(tmp_path, "workspace/data/web_chat.jsonl", "tok42")
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        assert out == "dashboard: http://192.168.1.10:9090/?token=tok42"

    def test_custom_history_path_derives_correct_token_location(
        self, monkeypatch, tmp_path, capsys
    ):
        """POSITIVE: custom history_path → token file derived from its parent dir."""
        custom_history = "workspace/logs/history.jsonl"
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        _write_dashboard_config(tmp_path, {
            "history_path": custom_history,
        })
        _write_token_file(tmp_path, custom_history, "tokenXYZ")
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        assert "?token=tokenXYZ" in out

    def test_multiple_plugins_dashboard_present_prints_url(
        self, monkeypatch, tmp_path, capsys
    ):
        """POSITIVE: plugins list has multiple entries including 'dashboard'."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["search", "dashboard", "telegram"]),
        )
        _write_token_file(tmp_path, "workspace/data/web_chat.jsonl", "multitok")
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        assert "dashboard: http://" in out
        assert "?token=multitok" in out

    def test_returns_none_on_success(
        self, monkeypatch, tmp_path
    ):
        """POSITIVE: function always returns None (not a status code)."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        _write_token_file(tmp_path, "workspace/data/web_chat.jsonl", "t")
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        result = _call(tmp_path)
        assert result is None

    def test_token_with_whitespace_is_stripped(
        self, monkeypatch, tmp_path, capsys
    ):
        """POSITIVE: token file content with surrounding whitespace/newline.
        The stripped token must appear in the URL."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        token_file = _derived_token_path(tmp_path, "workspace/data/web_chat.jsonl")
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("  mytoken\n", encoding="utf-8")
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        assert "?token=mytoken" in out


# ===========================================================================
# 2. Boundary value analysis
# ===========================================================================

class TestBVAPortZero:
    """BVA: port == 0 → silent return, nothing printed."""

    def test_port_zero_prints_nothing(
        self, monkeypatch, tmp_path, capsys
    ):
        """BVA: port=0 is the explicit 'no-print' boundary."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        _write_dashboard_config(tmp_path, {"port": 0})
        _write_token_file(tmp_path, "workspace/data/web_chat.jsonl", "tok")
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out
        assert out == ""

    def test_port_zero_returns_none(
        self, monkeypatch, tmp_path
    ):
        """BVA: port=0 still returns None (not an exception)."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        _write_dashboard_config(tmp_path, {"port": 0})
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        result = _call(tmp_path)
        assert result is None

    def test_port_one_does_not_trigger_zero_guard(
        self, monkeypatch, tmp_path, capsys
    ):
        """BVA: port=1 (min+1 above boundary) → should print URL, not be silenced."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        _write_dashboard_config(tmp_path, {"port": 1})
        _write_token_file(tmp_path, "workspace/data/web_chat.jsonl", "tok")
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        assert "dashboard: http://" in out


class TestBVAPluginsList:
    """BVA: falsy / missing / None plugins list → no print."""

    def test_plugins_none_prints_nothing(
        self, monkeypatch, tmp_path, capsys
    ):
        """BVA: cfg.plugins is None → early return, nothing printed."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(None),
        )

        _call(tmp_path)

        out = capsys.readouterr().out
        assert out == ""

    def test_plugins_empty_list_prints_nothing(
        self, monkeypatch, tmp_path, capsys
    ):
        """BVA: cfg.plugins is [] → early return, nothing printed."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg([]),
        )

        _call(tmp_path)

        out = capsys.readouterr().out
        assert out == ""

    def test_plugins_without_dashboard_prints_nothing(
        self, monkeypatch, tmp_path, capsys
    ):
        """BVA: plugins has entries but 'dashboard' is absent → no print."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["search", "telegram"]),
        )

        _call(tmp_path)

        out = capsys.readouterr().out
        assert out == ""

    def test_plugins_single_entry_not_dashboard_prints_nothing(
        self, monkeypatch, tmp_path, capsys
    ):
        """BVA: single non-dashboard plugin → no print."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["telegram"]),
        )

        _call(tmp_path)

        out = capsys.readouterr().out
        assert out == ""

    def test_plugins_single_entry_dashboard_prints_url(
        self, monkeypatch, tmp_path, capsys
    ):
        """BVA: single element list ['dashboard'] → crosses the boundary, prints."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        assert out.startswith("dashboard: http://")


class TestBVATokenFileEmpty:
    """BVA: token file exists but is empty → fallback URL (no token param)."""

    def test_empty_token_file_prints_base_url(
        self, monkeypatch, tmp_path, capsys
    ):
        """BVA: token file has only whitespace/empty → treated as absent token."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        token_file = _derived_token_path(tmp_path, "workspace/data/web_chat.jsonl")
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("", encoding="utf-8")
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        # Base URL must be printed; no ?token= param.
        assert out.startswith("dashboard: http://127.0.0.1:8765/")
        assert "?token=" not in out

    def test_whitespace_only_token_file_prints_base_url(
        self, monkeypatch, tmp_path, capsys
    ):
        """BVA: token file contains only spaces/newlines → strip → empty → fallback."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        token_file = _derived_token_path(tmp_path, "workspace/data/web_chat.jsonl")
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("   \n  \t  \n", encoding="utf-8")
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        assert "?token=" not in out
        assert out.startswith("dashboard: http://")


class TestBVAHostVerbatim:
    """BVA: host is printed verbatim — '0.0.0.0' must not be substituted."""

    def test_host_zero_zero_zero_zero_printed_verbatim(
        self, monkeypatch, tmp_path, capsys
    ):
        """BVA: host='0.0.0.0' must appear literally in the URL."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        _write_dashboard_config(tmp_path, {"host": "0.0.0.0"})
        _write_token_file(tmp_path, "workspace/data/web_chat.jsonl", "tok")
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        assert "0.0.0.0" in out
        assert "127.0.0.1" not in out

    def test_host_zero_zero_zero_zero_in_fallback_url(
        self, monkeypatch, tmp_path, capsys
    ):
        """BVA: 0.0.0.0 verbatim in fallback (no token) URL too."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        _write_dashboard_config(tmp_path, {"host": "0.0.0.0"})
        # No token file → fallback URL
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        assert "0.0.0.0" in out
        assert "127.0.0.1" not in out


class TestBVAMissingPluginConfig:
    """BVA: plugin config file missing → FilePluginConfigStore returns {} → all defaults."""

    def test_missing_plugin_config_uses_default_host(
        self, monkeypatch, tmp_path, capsys
    ):
        """BVA: no dashboard/config.yaml → defaults host=127.0.0.1, port=8765."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        # No plugin config file written; store.read('dashboard') returns {}.
        _write_token_file(tmp_path, "workspace/data/web_chat.jsonl", "tok")
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        assert "127.0.0.1" in out
        assert "8765" in out

    def test_missing_plugin_config_uses_default_history_path(
        self, monkeypatch, tmp_path, capsys
    ):
        """BVA: default history_path → token file expected at workspace/data/dashboard.token."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        # Write token at the default-derived location.
        _write_token_file(tmp_path, "workspace/data/web_chat.jsonl", "default_tok")
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        assert "?token=default_tok" in out


# ===========================================================================
# 3. State transition tests
# ===========================================================================

class TestStateTransitionTokenWait:
    """State transitions: token present at poll-time vs. never present."""

    def test_token_present_from_start_prints_token_url_fast(
        self, monkeypatch, tmp_path, capsys
    ):
        """STATE: token file exists BEFORE call → first poll succeeds immediately.
        The token URL (with ?token=) must be printed and time.sleep need not be called
        many times (may be called 0 or 1 times before success)."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        _write_token_file(tmp_path, "workspace/data/web_chat.jsonl", "fasttoken")

        sleep_calls: list[float] = []
        monkeypatch.setattr(
            "krakey.cli.lifecycle.time.sleep",
            lambda s: sleep_calls.append(s),
        )

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        assert "?token=fasttoken" in out
        # Sleep should have been called far fewer than ~100 times
        assert len(sleep_calls) < 10, (
            f"too many sleep calls ({len(sleep_calls)}) when token was present immediately"
        )

    def test_token_absent_produces_fallback_url(
        self, monkeypatch, tmp_path, capsys
    ):
        """STATE: token file never created → poll exhausts → fallback base URL printed.
        Uses no-op sleep so the loop spins fast without real ~5s wait."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        # No token file created.
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        # Must still print something — the base URL.
        assert out.startswith("dashboard: http://127.0.0.1:8765/")
        assert "?token=" not in out

    def test_fallback_url_still_prints_when_token_file_absent(
        self, monkeypatch, tmp_path, capsys
    ):
        """STATE: fallback must NEVER be silent — some URL is always printed
        when dashboard is enabled and port != 0."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        assert out != "", "fallback must print something, got empty output"
        assert "dashboard: http://" in out

    def test_safe_to_call_twice_in_succession(
        self, monkeypatch, tmp_path, capsys
    ):
        """STATE: calling _print_dashboard_url twice must not raise on either call."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        _write_token_file(tmp_path, "workspace/data/web_chat.jsonl", "tok")
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)
        _call(tmp_path)  # second call must also not raise

        # Both calls should have produced output; just ensure no exception.

    def test_token_appears_only_after_some_polls_is_captured(
        self, monkeypatch, tmp_path, capsys
    ):
        """STATE: simulate token file appearing partway through polling.
        We create the token file on the 3rd sleep call, then the poll should
        eventually capture it and print the token URL."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        token_file = _derived_token_path(tmp_path, "workspace/data/web_chat.jsonl")

        call_count = [0]

        def _deferred_sleep(_s):
            call_count[0] += 1
            if call_count[0] == 3:
                # Token file appears on the 3rd sleep call.
                token_file.parent.mkdir(parents=True, exist_ok=True)
                token_file.write_text("deferred_token", encoding="utf-8")

        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", _deferred_sleep)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        assert "?token=deferred_token" in out


# ===========================================================================
# 4. Negative tests — error guessing
# ===========================================================================

class TestNegativeLoadConfigRaises:
    """Negative: load_config raises → best-effort guard → silent return, no print."""

    def test_load_config_raises_runtime_error_no_print(
        self, monkeypatch, tmp_path, capsys
    ):
        """NEGATIVE: load_config throws RuntimeError → silent, nothing printed."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: (_ for _ in ()).throw(RuntimeError("config parse error")),
        )

        _call(tmp_path)

        out, err = capsys.readouterr().out, capsys.readouterr().err
        assert out == ""

    def test_load_config_raises_does_not_propagate(
        self, monkeypatch, tmp_path
    ):
        """NEGATIVE: load_config throws → function must not raise."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        # Must not raise.
        result = _call(tmp_path)
        assert result is None

    def test_load_config_raises_file_not_found_no_exception(
        self, monkeypatch, tmp_path
    ):
        """NEGATIVE: load_config raises FileNotFoundError → swallowed silently."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: (_ for _ in ()).throw(FileNotFoundError("no config.yaml")),
        )

        _call(tmp_path)  # must not raise

    def test_load_config_raises_value_error_no_exception(
        self, monkeypatch, tmp_path
    ):
        """NEGATIVE: load_config raises ValueError (bad YAML) → swallowed."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: (_ for _ in ()).throw(ValueError("invalid YAML")),
        )

        _call(tmp_path)  # must not raise


class TestNegativeMalformedPluginConfig:
    """Negative: dashboard plugin config.yaml malformed / not a mapping."""

    def test_non_mapping_plugin_config_does_not_raise(
        self, monkeypatch, tmp_path, capsys
    ):
        """NEGATIVE: plugin config.yaml is a list — FilePluginConfigStore returns {}.
        Function must not raise and must still print something (using defaults)."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        plugin_cfg_path = (
            tmp_path / "workspace" / "plugins" / "dashboard" / "config.yaml"
        )
        plugin_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        plugin_cfg_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)  # must not raise

        out = capsys.readouterr().out.strip()
        # FilePluginConfigStore returns {} for non-mapping → defaults → should print.
        assert out.startswith("dashboard: http://")

    def test_empty_plugin_config_yaml_does_not_raise(
        self, monkeypatch, tmp_path, capsys
    ):
        """NEGATIVE: dashboard config.yaml is completely empty.
        store.read returns {} → defaults used → prints base URL."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        plugin_cfg_path = (
            tmp_path / "workspace" / "plugins" / "dashboard" / "config.yaml"
        )
        plugin_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        plugin_cfg_path.write_text("", encoding="utf-8")
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)  # must not raise

    def test_plugin_config_with_string_port_does_not_crash(
        self, monkeypatch, tmp_path
    ):
        """NEGATIVE: port is a string in the YAML → may cause issues, must not raise."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        _write_dashboard_config(tmp_path, {
            "host": "127.0.0.1",
            "port": "not_a_number",
            "history_path": "workspace/data/web_chat.jsonl",
        })
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)  # must not raise


class TestNegativeMissingRepo:
    """Negative: repo path has no files whatsoever → never raises."""

    def test_completely_empty_repo_does_not_raise(
        self, monkeypatch, tmp_path
    ):
        """NEGATIVE: tmp_path has no config.yaml, no workspace, no plugins.
        load_config is NOT patched — it will raise because there is no config.yaml.
        The best-effort wrapper must catch and silently return."""
        # Do NOT monkeypatch load_config — let it fail naturally.
        # We only ensure the function itself does not propagate.
        _call(tmp_path)  # must not raise

    def test_completely_empty_repo_prints_nothing(
        self, monkeypatch, tmp_path, capsys
    ):
        """NEGATIVE: with a real failing load_config, nothing should be printed."""
        _call(tmp_path)

        out = capsys.readouterr().out
        assert out == ""

    def test_nonexistent_repo_path_does_not_raise(
        self, monkeypatch, tmp_path
    ):
        """NEGATIVE: repo path that does not exist at all → must not raise."""
        fake_repo = tmp_path / "no_such_dir"
        assert not fake_repo.exists()

        _call(fake_repo)  # must not raise

    def test_nonexistent_repo_path_prints_nothing(
        self, monkeypatch, tmp_path, capsys
    ):
        """NEGATIVE: nonexistent repo → swallowed → no output."""
        fake_repo = tmp_path / "no_such_dir"
        _call(fake_repo)

        out = capsys.readouterr().out
        assert out == ""


class TestNegativeTokenFileUnreadable:
    """Negative: token file exists but cannot be read (permissions) → swallowed."""

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="chmod 000 permission tests are not reliable on Windows",
    )
    def test_unreadable_token_file_does_not_raise(
        self, monkeypatch, tmp_path, capsys
    ):
        """NEGATIVE: token file exists with no read permission → best-effort swallows."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        token_file = _derived_token_path(tmp_path, "workspace/data/web_chat.jsonl")
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("secret", encoding="utf-8")
        token_file.chmod(0o000)
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        try:
            _call(tmp_path)  # must not raise
        finally:
            # Restore so tmp_path cleanup works.
            token_file.chmod(0o644)

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="chmod 000 permission tests are not reliable on Windows",
    )
    def test_unreadable_token_file_prints_something(
        self, monkeypatch, tmp_path, capsys
    ):
        """NEGATIVE: unreadable token → error swallowed → either fallback URL or silent.
        The spec says best-effort: any exception → silent. So either a base URL is printed
        (if the error is caught at poll level and loop falls through to fallback)
        or nothing is printed (if the exception causes early return). Either is acceptable
        as long as no exception propagates."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        token_file = _derived_token_path(tmp_path, "workspace/data/web_chat.jsonl")
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("secret", encoding="utf-8")
        token_file.chmod(0o000)
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        try:
            _call(tmp_path)
            # No assertion on output — spec says "any exception → silent return".
            # The test verifies only that no exception escaped.
        finally:
            token_file.chmod(0o644)


class TestNegativeExceptionNeverEscapes:
    """Negative: various exception types from arbitrary internal failures → silent."""

    def test_cfg_plugins_attribute_raises_on_access_no_exception(
        self, monkeypatch, tmp_path
    ):
        """NEGATIVE: cfg.plugins access raises AttributeError → swallowed."""
        class _BadCfg:
            @property
            def plugins(self):
                raise AttributeError("plugins gone")

        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _BadCfg(),
        )

        _call(tmp_path)  # must not raise

    def test_cfg_plugins_attribute_raises_prints_nothing(
        self, monkeypatch, tmp_path, capsys
    ):
        """NEGATIVE: AttributeError accessing .plugins → silent, no output."""
        class _BadCfg:
            @property
            def plugins(self):
                raise AttributeError("plugins gone")

        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _BadCfg(),
        )

        _call(tmp_path)

        out = capsys.readouterr().out
        assert out == ""

    def test_load_config_returns_none_no_exception(
        self, monkeypatch, tmp_path
    ):
        """NEGATIVE: load_config returns None (unexpected) → AttributeError on .plugins
        must be swallowed by the best-effort guard."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: None,
        )

        _call(tmp_path)  # must not raise

    def test_load_config_returns_none_prints_nothing(
        self, monkeypatch, tmp_path, capsys
    ):
        """NEGATIVE: load_config returns None → .plugins access fails → no output."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: None,
        )

        _call(tmp_path)

        out = capsys.readouterr().out
        assert out == ""

    def test_exception_does_not_print_traceback(
        self, monkeypatch, tmp_path, capsys
    ):
        """NEGATIVE: when an exception is swallowed, no traceback must leak to stdout
        or stderr."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: (_ for _ in ()).throw(RuntimeError("internal boom")),
        )

        _call(tmp_path)

        out, err = capsys.readouterr()
        # No 'Traceback' header and no 'RuntimeError' in either stream.
        assert "Traceback" not in out
        assert "Traceback" not in err
        assert "RuntimeError" not in out
        assert "RuntimeError" not in err


# ===========================================================================
# 5. Output format precision tests
# ===========================================================================

class TestOutputFormatPrecision:
    """Verify exact output line structure from the spec."""

    def test_token_url_format_is_exact(
        self, monkeypatch, tmp_path, capsys
    ):
        """FORMAT: output must be exactly 'dashboard: http://{host}:{port}/?token={token}'."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        _write_token_file(tmp_path, "workspace/data/web_chat.jsonl", "abc")
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        assert out == "dashboard: http://127.0.0.1:8765/?token=abc"

    def test_fallback_url_format_is_exact(
        self, monkeypatch, tmp_path, capsys
    ):
        """FORMAT: fallback output must be exactly 'dashboard: http://{host}:{port}/'."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        assert out == "dashboard: http://127.0.0.1:8765/"

    def test_token_url_uses_query_param_named_token(
        self, monkeypatch, tmp_path, capsys
    ):
        """FORMAT: query param must be literally '?token=' (not '?t=' or '#token=')."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        _write_token_file(tmp_path, "workspace/data/web_chat.jsonl", "xyz")
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        assert "?token=xyz" in out

    def test_output_is_single_line(
        self, monkeypatch, tmp_path, capsys
    ):
        """FORMAT: output must be a single line — no extra blank lines or headers."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        _write_token_file(tmp_path, "workspace/data/web_chat.jsonl", "t")
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out
        # Strip trailing newline then check no internal newlines remain.
        lines = [l for l in out.splitlines() if l.strip()]
        assert len(lines) == 1, f"expected 1 non-empty line, got: {lines!r}"

    def test_token_is_appended_raw_no_encoding(
        self, monkeypatch, tmp_path, capsys
    ):
        """FORMAT: token value appended raw — special chars like '=' in token
        must NOT be percent-encoded."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        _write_token_file(tmp_path, "workspace/data/web_chat.jsonl", "a=b&c")
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        # Token must appear literally as 'a=b&c', not URL-encoded.
        assert "?token=a=b&c" in out

    def test_custom_port_in_url(
        self, monkeypatch, tmp_path, capsys
    ):
        """FORMAT: custom port appears correctly in printed URL."""
        monkeypatch.setattr(
            "krakey.models.config.load_config",
            lambda _path: _make_cfg(["dashboard"]),
        )
        _write_dashboard_config(tmp_path, {"port": 12345})
        monkeypatch.setattr("krakey.cli.lifecycle.time.sleep", lambda _s: None)

        _call(tmp_path)

        out = capsys.readouterr().out.strip()
        assert ":12345/" in out
