"""Onboarding wizard — happy path, plugin recommendations, idempotent re-runs.

The wizard is fully injectable (input_fn / output_fn / list_plugins_fn);
these tests drive it with stubbed I/O so they're hermetic.
"""
from __future__ import annotations

from typing import Iterable

import pytest

from krakey.models.config import load_config
from krakey.onboarding import run_wizard
from krakey.plugin_system.loader import PluginMetadata


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


def _skip_verify(kind, provider, model):
    """Tests don't hit the network — stub the connectivity check
    so the wizard treats every endpoint as reachable."""
    return True, "stubbed"


def _no_models(provider):
    """Stub model listing as unavailable so tests fall through to
    plain text model entry. Tests that exercise the picker pass
    their own stub."""
    return None


def test_wizard_writes_minimal_config(tmp_path):
    """Happy path: chat provider + skip embedding + accept defaults
    in plugin picker. Resulting file round-trips through load_config."""
    cfg_path = tmp_path / "config.yaml"
    answers = [
        # Step 1: chat provider — choose openai_compatible (1)
        "1",                             # provider type
        "MyOpenAI",                      # provider label
        "https://api.example.com",       # base url
        "sk-test-key",                   # api key
        "gpt-4o-mini",                   # model
        # Step 2: skip embedding
        "n",
        # Step 3: skip reranker
        "n",
        # Step 4: accept default plugin selection (dashboard preselected)
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
        verify_fn=_skip_verify,
        list_models_fn=_no_models,
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
        "1", "P", "http://x", "k", "m",   # provider
        "n",                          # skip embedding
        "n",                          # skip reranker
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
        verify_fn=_skip_verify,
        list_models_fn=_no_models,
    )
    cfg = load_config(cfg_path)
    assert cfg.plugins == ["dashboard"]
    block = "\n".join(lines)
    assert block.index("dashboard") < block.index("aaa_first_alpha")
    # The dashboard listing line must carry the recommended star and
    # the [x] preselect mark. Format: "  1. [x] * dashboard"
    dashboard_line = next(
        l for l in lines
        if "dashboard" in l and "1." in l and ("[x]" in l or "[ ]" in l)
    )
    assert "*" in dashboard_line
    assert "[x]" in dashboard_line


def test_wizard_toggle_plugin_selection(tmp_path):
    """User can toggle dashboard off and other plugins on by index."""
    cfg_path = tmp_path / "config.yaml"
    catalogue = _fake_catalogue("dashboard", "memory", "telegram")
    # Recommended-first sort: 1=dashboard, 2=memory, 3=telegram
    answers = [
        "1", "P", "http://x", "k", "m",   # provider
        "n",                          # skip embedding
        "n",                          # skip reranker
        "1",                          # toggle dashboard OFF
        "3",                          # toggle telegram ON
        "done",
        "y",                          # confirm: yes continue without dashboard
        "y",                          # save config
    ]
    _, out = _capture_output()
    run_wizard(
        config_path=cfg_path,
        backup_dir=str(tmp_path / "backups"),
        input_fn=_stub_inputs(answers),
        output_fn=out,
        list_plugins_fn=lambda: catalogue,
        verify_fn=_skip_verify,
        list_models_fn=_no_models,
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
        "1",                              # chat type: openai_compatible
        "ChatCo", "http://x", "k", "chat-model",
        "y", "y", "embed-model",   # embed: yes, same provider, model
        "n",                       # skip reranker
        "done", "y",
    ]
    _, out = _capture_output()
    run_wizard(
        config_path=cfg_path,
        backup_dir=str(tmp_path / "backups"),
        input_fn=_stub_inputs(answers),
        output_fn=out,
        list_plugins_fn=lambda: catalogue,
        verify_fn=_skip_verify,
        list_models_fn=_no_models,
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
        "1", "P", "http://x", "k", "m", "n", "n", "done", "y",
    ]
    catalogue = _fake_catalogue("dashboard")
    lines, out = _capture_output()
    run_wizard(
        config_path=cfg_path,
        backup_dir=str(backup_dir),
        input_fn=_stub_inputs(answers),
        output_fn=out,
        list_plugins_fn=lambda: catalogue,
        verify_fn=_skip_verify,
        list_models_fn=_no_models,
    )
    backups = list(backup_dir.iterdir())
    assert backups, "expected a backup file in backup_dir"
    assert any("backed up" in line for line in lines)


def test_wizard_reranker_reuses_embedding_provider(tmp_path):
    """Configuring a reranker after embedding reuses the embedding
    provider by default — rerankers commonly co-locate with embedders."""
    cfg_path = tmp_path / "config.yaml"
    catalogue = _fake_catalogue("dashboard")
    answers = [
        "1",                              # chat type: openai_compatible
        "ChatCo", "http://x", "k", "chat-model",
        # embed: yes, separate provider, type 1, then fields, model
        "y", "n", "1", "EmbedCo", "http://e", "ek", "embed-model",
        # reranker: yes, reuse embedding provider, model
        "y", "y", "rerank-model",
        "done", "y",
    ]
    _, out = _capture_output()
    run_wizard(
        config_path=cfg_path,
        backup_dir=str(tmp_path / "backups"),
        input_fn=_stub_inputs(answers),
        output_fn=out,
        list_plugins_fn=lambda: catalogue,
        verify_fn=_skip_verify,
        list_models_fn=_no_models,
    )
    cfg = load_config(cfg_path)
    assert cfg.llm.reranker == "rerank"
    assert cfg.llm.tags["rerank"].provider == "EmbedCo/rerank-model"
    # No duplicate provider entry — EmbedCo is reused.
    assert sorted(cfg.llm.providers.keys()) == ["ChatCo", "EmbedCo"]


def test_wizard_skip_reranker_leaves_field_unset(tmp_path):
    """When the user declines reranker config, llm.reranker stays None."""
    cfg_path = tmp_path / "config.yaml"
    catalogue = _fake_catalogue("dashboard")
    answers = [
        "1", "P", "http://x", "k", "m",
        "n",         # skip embedding
        "n",         # skip reranker
        "done", "y",
    ]
    _, out = _capture_output()
    run_wizard(
        config_path=cfg_path,
        backup_dir=str(tmp_path / "backups"),
        input_fn=_stub_inputs(answers),
        output_fn=out,
        list_plugins_fn=lambda: catalogue,
        verify_fn=_skip_verify,
        list_models_fn=_no_models,
    )
    cfg = load_config(cfg_path)
    assert cfg.llm.reranker is None
    assert "rerank" not in cfg.llm.tags


def test_wizard_verify_called_for_each_endpoint(tmp_path):
    """verify_fn fires once per provider step the user actually enables."""
    cfg_path = tmp_path / "config.yaml"
    catalogue = _fake_catalogue("dashboard")
    calls: list[tuple[str, str, str]] = []

    def _record(kind, provider, model):
        calls.append((kind, provider.base_url, model))
        return True, "ok"

    answers = [
        "1", "ChatCo", "http://chat", "k1", "chat-model",
        "y", "n", "1", "EmbedCo", "http://embed", "k2", "embed-model",
        "y", "n", "1", "RerankCo", "http://rerank", "k3", "rerank-model",
        "done", "y",
    ]
    _, out = _capture_output()
    run_wizard(
        config_path=cfg_path,
        backup_dir=str(tmp_path / "backups"),
        input_fn=_stub_inputs(answers),
        output_fn=out,
        list_plugins_fn=lambda: catalogue,
        verify_fn=_record,
        list_models_fn=_no_models,
    )
    kinds = [c[0] for c in calls]
    assert kinds == ["chat", "embedding", "reranker"]
    assert calls[0] == ("chat", "http://chat", "chat-model")
    assert calls[1] == ("embedding", "http://embed", "embed-model")
    assert calls[2] == ("reranker", "http://rerank", "rerank-model")


def test_wizard_verify_failure_warns_but_does_not_abort(tmp_path):
    """A failing verify is reported as a warning; the wizard still
    writes the config so the user can fix the typo offline."""
    cfg_path = tmp_path / "config.yaml"
    catalogue = _fake_catalogue("dashboard")

    def _failing(kind, provider, model):
        return False, "HTTP 401 Unauthorized"

    answers = [
        "1", "P", "http://x", "k", "m",
        "n",         # skip embedding
        "n",         # skip reranker
        "done", "y",
    ]
    lines, out = _capture_output()
    run_wizard(
        config_path=cfg_path,
        backup_dir=str(tmp_path / "backups"),
        input_fn=_stub_inputs(answers),
        output_fn=out,
        list_plugins_fn=lambda: catalogue,
        verify_fn=_failing,
        list_models_fn=_no_models,
    )
    # Config still written.
    assert cfg_path.exists()
    block = "\n".join(lines)
    assert "warn" in block.lower()
    assert "401" in block


def test_wizard_anthropic_provider_type(tmp_path):
    """Picking 'anthropic' (option 2) sets provider.type accordingly."""
    cfg_path = tmp_path / "config.yaml"
    catalogue = _fake_catalogue("dashboard")
    answers = [
        "2",                              # chat type: anthropic
        "Claude",                         # label
        "https://api.anthropic.com/v1",   # base
        "sk-ant-test",                    # api key
        "claude-haiku-4-5-20251001",      # model
        "n", "n", "done", "y",
    ]
    _, out = _capture_output()
    run_wizard(
        config_path=cfg_path,
        backup_dir=str(tmp_path / "backups"),
        input_fn=_stub_inputs(answers),
        output_fn=out,
        list_plugins_fn=lambda: catalogue,
        verify_fn=_skip_verify,
        list_models_fn=_no_models,
    )
    cfg = load_config(cfg_path)
    assert cfg.llm.providers["Claude"].type == "anthropic"


def test_wizard_skip_chat_force_enables_dashboard(tmp_path):
    """If user picks 'skip for now' on chat, the wizard force-adds the
    dashboard plugin so the user has a way to fill in providers later
    even if they unchecked it in step 4."""
    cfg_path = tmp_path / "config.yaml"
    catalogue = _fake_catalogue("dashboard", "memory")
    answers = [
        "3",                  # chat type: skip
        "n",                  # skip embedding
        "n",                  # skip reranker
        "1",                  # toggle dashboard OFF (was preselected)
        "done",
        "y",                  # confirm: yes continue without dashboard
        "y",                  # save config
    ]
    lines, out = _capture_output()
    run_wizard(
        config_path=cfg_path,
        backup_dir=str(tmp_path / "backups"),
        input_fn=_stub_inputs(answers),
        output_fn=out,
        list_plugins_fn=lambda: catalogue,
        verify_fn=_skip_verify,
        list_models_fn=_no_models,
    )
    cfg = load_config(cfg_path)
    # Dashboard force-enabled despite being toggled off, because the
    # user has no providers configured.
    assert cfg.plugins is not None
    assert "dashboard" in cfg.plugins
    block = "\n".join(lines)
    assert "auto-enabling dashboard" in block.lower() \
        or "auto-enabling" in block.lower()


def test_wizard_model_picker_uses_listed_models(tmp_path):
    """When list_models_fn returns IDs, the wizard shows a numbered
    picker; answering with the index uses that model name."""
    cfg_path = tmp_path / "config.yaml"
    catalogue = _fake_catalogue("dashboard")

    listed = ["gpt-4o-mini", "gpt-4o", "o1-mini"]

    def _models(provider):
        return listed

    answers = [
        "1",                              # chat type
        "OpenAI", "http://x", "k",       # label, base, key
        "2",                              # pick model #2 → gpt-4o
        "n", "n", "done", "y",
    ]
    lines, out = _capture_output()
    run_wizard(
        config_path=cfg_path,
        backup_dir=str(tmp_path / "backups"),
        input_fn=_stub_inputs(answers),
        output_fn=out,
        list_plugins_fn=lambda: catalogue,
        verify_fn=_skip_verify,
        list_models_fn=_models,
    )
    cfg = load_config(cfg_path)
    assert cfg.llm.tags["self_main"].provider == "OpenAI/gpt-4o"
    block = "\n".join(lines)
    for m in listed:
        assert m in block


def test_wizard_model_picker_falls_back_to_text_when_listing_fails(tmp_path):
    """When list_models_fn returns None (network error / unsupported),
    the wizard quietly falls back to plain-text model name entry."""
    cfg_path = tmp_path / "config.yaml"
    catalogue = _fake_catalogue("dashboard")
    answers = [
        "1", "OpenAI", "http://x", "k",
        "my-custom-model",                # plain entry, no picker
        "n", "n", "done", "y",
    ]
    _, out = _capture_output()
    run_wizard(
        config_path=cfg_path,
        backup_dir=str(tmp_path / "backups"),
        input_fn=_stub_inputs(answers),
        output_fn=out,
        list_plugins_fn=lambda: catalogue,
        verify_fn=_skip_verify,
        list_models_fn=_no_models,        # returns None
    )
    cfg = load_config(cfg_path)
    assert cfg.llm.tags["self_main"].provider == "OpenAI/my-custom-model"


def test_wizard_dashboard_nudge_re_enables_on_no(tmp_path):
    """Toggling dashboard off and answering 'no' to the 'continue
    without dashboard?' prompt re-adds dashboard to the selection."""
    cfg_path = tmp_path / "config.yaml"
    catalogue = _fake_catalogue("dashboard", "telegram")
    answers = [
        "1", "P", "http://x", "k", "m",
        "n",                          # skip embedding
        "n",                          # skip reranker
        "1",                          # toggle dashboard OFF
        "done",
        "n",                          # nudge: NO, don't continue without dashboard
        "done",                       # back to plugin loop, accept current
        "y",                          # save (no second nudge — dashboard now selected)
    ]
    lines, out = _capture_output()
    run_wizard(
        config_path=cfg_path,
        backup_dir=str(tmp_path / "backups"),
        input_fn=_stub_inputs(answers),
        output_fn=out,
        list_plugins_fn=lambda: catalogue,
        verify_fn=_skip_verify,
        list_models_fn=_no_models,
    )
    cfg = load_config(cfg_path)
    assert "dashboard" in (cfg.plugins or [])
    assert any("re-enabled dashboard" in l for l in lines)


def test_wizard_skip_embedding_warns(tmp_path):
    """Declining embedding config prints a [warn] line about
    recall + KB indexing being inert."""
    cfg_path = tmp_path / "config.yaml"
    catalogue = _fake_catalogue("dashboard")
    answers = [
        "1", "P", "http://x", "k", "m",
        "n",                  # skip embedding
        "n",                  # skip reranker
        "done", "y",
    ]
    lines, out = _capture_output()
    run_wizard(
        config_path=cfg_path,
        backup_dir=str(tmp_path / "backups"),
        input_fn=_stub_inputs(answers),
        output_fn=out,
        list_plugins_fn=lambda: catalogue,
        verify_fn=_skip_verify,
        list_models_fn=_no_models,
    )
    block = "\n".join(lines).lower()
    assert "warn" in block
    assert ("recall" in block or "kb" in block)


def test_arrow_picker_toggles_with_space_and_confirms_with_enter(
    monkeypatch, capsys,
):
    """The arrow-key picker drives plugin selection without input_fn:
    Down moves cursor, Space toggles, Enter confirms. Test by injecting
    a scripted key sequence and a fake `is_interactive=True`."""
    from krakey.onboarding import wizard as wiz
    from krakey.onboarding import _ui

    # Force the picker into the interactive branch.
    monkeypatch.setattr(_ui, "is_interactive", lambda: True)

    # Catalogue: dashboard preselected, memory + telegram start unchecked.
    catalogue = _fake_catalogue("dashboard", "memory", "telegram")
    names = sorted(
        catalogue.keys(),
        key=lambda n: (n not in wiz.RECOMMENDED_PLUGINS, n),
    )
    initial = {"dashboard"}

    # Cursor starts at index 0 = dashboard.
    # Sequence: Down (→ memory) → Space (→ select memory) → Enter.
    keys = iter([_ui.KEY_DOWN, _ui.KEY_SPACE, _ui.KEY_ENTER])
    monkeypatch.setattr(_ui, "read_key", lambda: next(keys))

    selected = wiz._ask_plugins_arrow(names, catalogue, initial)
    assert selected == {"dashboard", "memory"}


def test_arrow_picker_esc_returns_current_selection(monkeypatch):
    from krakey.onboarding import wizard as wiz
    from krakey.onboarding import _ui

    monkeypatch.setattr(_ui, "is_interactive", lambda: True)
    catalogue = _fake_catalogue("dashboard", "memory")
    names = sorted(
        catalogue.keys(),
        key=lambda n: (n not in wiz.RECOMMENDED_PLUGINS, n),
    )
    initial = {"dashboard"}

    keys = iter([_ui.KEY_ESC])
    monkeypatch.setattr(_ui, "read_key", lambda: next(keys))

    selected = wiz._ask_plugins_arrow(names, catalogue, initial)
    # Esc keeps the initial selection unchanged.
    assert selected == {"dashboard"}


def test_module_exports_run_wizard():
    """`from krakey.onboarding import run_wizard` works (entry point relies on it)."""
    from krakey.onboarding import run_wizard as imported
    assert callable(imported)


def test_wizard_handles_unknown_command(tmp_path):
    """Garbage input at the plugin picker is rejected without crashing."""
    cfg_path = tmp_path / "config.yaml"
    catalogue = _fake_catalogue("dashboard")
    answers = [
        "1", "P", "http://x", "k", "m", "n", "n",
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
        verify_fn=_skip_verify,
        list_models_fn=_no_models,
    )
    assert any("unknown command" in line for line in lines)
    assert any("out of range" in line for line in lines)
    # Dashboard pre-select stayed in place across the noisy commands.
    cfg = load_config(cfg_path)
    assert cfg.plugins == ["dashboard"]
