"""Onboarding wizard — happy path, plugin recommendations, idempotent re-runs.

The wizard is fully injectable (input_fn / output_fn / list_plugins_fn);
these tests drive it with stubbed I/O so they're hermetic.
"""
from __future__ import annotations

from typing import Iterable

import pytest

from src.models.config import load_config
from src.onboarding import run_wizard
from src.plugin_system.loader import PluginMetadata


def _stub_inputs(answers: Iterable[str]):
    it = iter(answers)

    def _input(prompt: str) -> str:
        try:
            return next(it)
        except StopIteration as e:
            raise AssertionError(
                f"wizard asked an extra prompt {prompt!r}; ran out of stubs"
            ) from e

    return _input


def _capture_output():
    lines: list[str] = []

    def _print(msg: str) -> None:
        lines.append(str(msg))

    return lines, _print


def _fake_catalogue(*names: str) -> dict[str, PluginMetadata]:
    return {n: PluginMetadata(name=n, description=f"{n} plugin") for n in names}


def test_wizard_writes_minimal_config(tmp_path):
    """Happy path: chat provider + skip embedding + accept defaults
    in plugin picker. Resulting file round-trips through load_config."""
    cfg_path = tmp_path / "config.yaml"
    answers = [
        # Step 1: chat provider
        "MyOpenAI",                      # provider label
        "https://api.example.com",       # base url
        "sk-test-key",                   # api key
        "gpt-4o-mini",                   # model
        # Step 2: skip embedding
        "n",
        # Step 3: accept default plugin selection (dashboard preselected)
        "done",
        # Confirm save
        "y",
    ]
    lines, out = _capture_output()
    catalogue = _fake_catalogue("dashboard", "memory", "telegram")
    written = run_wizard(
        config_path=cfg_path,
        backup_dir=str(tmp_path / "backups"),
        input_fn=_stub_inputs(answers),
        output_fn=out,
        list_plugins_fn=lambda: catalogue,
    )
    assert written == cfg_path
    assert cfg_path.exists()
    cfg = load_config(cfg_path)
    assert "MyOpenAI" in cfg.llm.providers
    assert cfg.llm.providers["MyOpenAI"].api_key == "sk-test-key"
    assert "self_main" in cfg.llm.tags
    assert cfg.llm.core_purposes["self_thinking"] == "self_main"
    assert cfg.plugins == ["dashboard"]


def test_wizard_dashboard_default_recommended_and_first(tmp_path):
    """Dashboard sorts to the top of the picker AND is preselected."""
    cfg_path = tmp_path / "config.yaml"
    answers = [
        "P", "http://x", "k", "m",   # provider
        "n",                          # skip embedding
        "done",                       # accept default
        "y",                          # save
    ]
    lines, out = _capture_output()
    catalogue = _fake_catalogue(
        "aaa_first_alpha", "dashboard", "zzz_last",
    )
    run_wizard(
        config_path=cfg_path,
        backup_dir=str(tmp_path / "backups"),
        input_fn=_stub_inputs(answers),
        output_fn=out,
        list_plugins_fn=lambda: catalogue,
    )
    cfg = load_config(cfg_path)
    assert cfg.plugins == ["dashboard"]
    block = "\n".join(lines)
    assert block.index("dashboard") < block.index("aaa_first_alpha")
    # The dashboard listing line must carry the recommended star and
    # the [x] preselect mark.
    dashboard_line = next(
        l for l in lines if "dashboard - " in l and l.lstrip().startswith("1.")
    )
    assert "*" in dashboard_line
    assert "[x]" in dashboard_line


def test_wizard_toggle_plugin_selection(tmp_path):
    """User can toggle dashboard off and other plugins on by index."""
    cfg_path = tmp_path / "config.yaml"
    catalogue = _fake_catalogue("dashboard", "memory", "telegram")
    # Recommended-first sort: 1=dashboard, 2=memory, 3=telegram
    answers = [
        "P", "http://x", "k", "m",   # provider
        "n",                          # skip embedding
        "1",                          # toggle dashboard OFF
        "3",                          # toggle telegram ON
        "done",
        "y",
    ]
    _, out = _capture_output()
    run_wizard(
        config_path=cfg_path,
        backup_dir=str(tmp_path / "backups"),
        input_fn=_stub_inputs(answers),
        output_fn=out,
        list_plugins_fn=lambda: catalogue,
    )
    cfg = load_config(cfg_path)
    assert cfg.plugins is not None
    assert "dashboard" not in cfg.plugins
    assert "telegram" in cfg.plugins


def test_wizard_embedding_same_provider_as_chat(tmp_path):
    """Reusing the chat provider for embeddings does not duplicate it."""
    cfg_path = tmp_path / "config.yaml"
    catalogue = _fake_catalogue("dashboard")
    answers = [
        "ChatCo", "http://x", "k", "chat-model",
        "y", "y", "embed-model",   # embed: yes, same provider, model
        "done", "y",
    ]
    _, out = _capture_output()
    run_wizard(
        config_path=cfg_path,
        backup_dir=str(tmp_path / "backups"),
        input_fn=_stub_inputs(answers),
        output_fn=out,
        list_plugins_fn=lambda: catalogue,
    )
    cfg = load_config(cfg_path)
    assert list(cfg.llm.providers.keys()) == ["ChatCo"]
    assert cfg.llm.embedding == "embed"
    assert cfg.llm.tags["embed"].provider == "ChatCo/embed-model"


def test_wizard_backs_up_existing_config(tmp_path):
    """Re-running over an existing config.yaml backs it up first."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("existing: 1\n", encoding="utf-8")
    backup_dir = tmp_path / "backups"
    answers = [
        "P", "http://x", "k", "m", "n", "done", "y",
    ]
    catalogue = _fake_catalogue("dashboard")
    lines, out = _capture_output()
    run_wizard(
        config_path=cfg_path,
        backup_dir=str(backup_dir),
        input_fn=_stub_inputs(answers),
        output_fn=out,
        list_plugins_fn=lambda: catalogue,
    )
    backups = list(backup_dir.iterdir())
    assert backups, "expected a backup file in backup_dir"
    assert any("backed up" in line for line in lines)


def test_wizard_abort_at_confirm_does_not_write(tmp_path):
    """Saying 'n' at the final confirm leaves no file on disk."""
    cfg_path = tmp_path / "config.yaml"
    catalogue = _fake_catalogue("dashboard")
    answers = [
        "P", "http://x", "k", "m", "n", "done",
        "n",  # confirm: no
    ]
    lines, out = _capture_output()
    run_wizard(
        config_path=cfg_path,
        backup_dir=str(tmp_path / "backups"),
        input_fn=_stub_inputs(answers),
        output_fn=out,
        list_plugins_fn=lambda: catalogue,
    )
    assert not cfg_path.exists()
    assert any("aborted" in line for line in lines)


def test_module_exports_run_wizard():
    """`from src.onboarding import run_wizard` works (entry point relies on it)."""
    from src.onboarding import run_wizard as imported
    assert callable(imported)


def test_wizard_handles_unknown_command(tmp_path):
    """Garbage input at the plugin picker is rejected without crashing."""
    cfg_path = tmp_path / "config.yaml"
    catalogue = _fake_catalogue("dashboard")
    answers = [
        "P", "http://x", "k", "m", "n",
        "??",       # unknown command
        "999",      # out of range
        "done",
        "y",
    ]
    lines, out = _capture_output()
    run_wizard(
        config_path=cfg_path,
        backup_dir=str(tmp_path / "backups"),
        input_fn=_stub_inputs(answers),
        output_fn=out,
        list_plugins_fn=lambda: catalogue,
    )
    assert any("unknown command" in line for line in lines)
    assert any("out of range" in line for line in lines)
    # Dashboard pre-select stayed in place across the noisy commands.
    cfg = load_config(cfg_path)
    assert cfg.plugins == ["dashboard"]
