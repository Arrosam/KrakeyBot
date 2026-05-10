"""Config loader tests — tag-based LLM shape (post 2026-04-26 refactor).

The old `llm.roles:` shape is removed; loader detects it and exits.
New shape: `llm.tags` + `llm.core_purposes` + `llm.embedding` +
`llm.reranker`.
"""
import os
import textwrap

import pytest
import yaml

from krakey.models.config import (
    Config,
    LLMParams,
    LLMSection,
    Provider,
    TagBinding,
    dump_config,
    ensure_config,
    llm_params_schema,
    load_config,
)
from krakey.models.config import _ConfigBootstrapExit


def _write(tmp_path, body):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


# ---- TagBinding parsing ----------------------------------------------


def test_tag_split_provider_simple():
    t = TagBinding(provider="One API/qwen3.6-9b")
    assert t.split_provider() == ("One API", "qwen3.6-9b")


def test_tag_split_provider_model_with_slash():
    """Model names containing '/' must survive (e.g. BAAI/bge-m3).
    Split on the FIRST slash; the rest is the model name."""
    t = TagBinding(provider="SiliconFlow/BAAI/bge-m3")
    assert t.split_provider() == ("SiliconFlow", "BAAI/bge-m3")


def test_tag_split_provider_missing_slash_raises():
    t = TagBinding(provider="just-a-model")
    with pytest.raises(ValueError, match="provider.*model"):
        t.split_provider()


# ---- Loader: tag + core_purposes shape -------------------------------


def test_loader_parses_tags_and_core_purposes(tmp_path):
    p = _write(tmp_path, """
        llm:
          providers:
            "One API":
              type: openai_compatible
              base_url: "http://x"
              api_key: "k"
              models: []
          tags:
            fast:
              provider: "One API/qwen-9b"
              params: {max_output_tokens: 512}
            heavy:
              provider: "One API/astron"
              params: {max_output_tokens: 8192}
          core_purposes:
            self_thinking: heavy
            compact: fast
          embedding: fast
        idle: {min_interval: 2, max_interval: 300, default_interval: 30}
        fatigue: {gm_node_soft_limit: 200, force_sleep_threshold: 120, thresholds: {}}
        graph_memory: {db_path: "x", auto_ingest_similarity_threshold: 0.9, recall_per_stimulus_k: 5, neighbor_expand_depth: 1}
        knowledge_base: {dir: "kb"}
        sleep: {max_duration_seconds: 7200}
        safety: {gm_node_hard_limit: 500, max_consecutive_no_action: 50}
    """)
    cfg = load_config(p)
    assert "fast" in cfg.llm.tags
    assert "heavy" in cfg.llm.tags
    assert cfg.llm.tags["fast"].provider == "One API/qwen-9b"
    assert cfg.llm.tags["fast"].params.max_output_tokens == 512
    assert cfg.llm.core_purposes == {
        "self_thinking": "heavy",
        "compact": "fast",
    }
    assert cfg.llm.embedding == "fast"
    assert cfg.llm.reranker is None  # not set


def test_loader_resolves_max_input_tokens_from_model_lookup(tmp_path):
    """Tag's params.max_input_tokens auto-resolved from the model
    name (split out of the provider/model field) if user didn't set it."""
    p = _write(tmp_path, """
        llm:
          providers:
            "Anthropic":
              type: anthropic
              base_url: "http://x"
              api_key: "k"
              models: []
          tags:
            heavy:
              provider: "Anthropic/claude-sonnet-4-5"
        idle: {min_interval: 1, max_interval: 60, default_interval: 1}
        fatigue: {gm_node_soft_limit: 100, force_sleep_threshold: 60, thresholds: {}}
        graph_memory: {db_path: "x", auto_ingest_similarity_threshold: 0.9, recall_per_stimulus_k: 5, neighbor_expand_depth: 1}
        knowledge_base: {dir: "kb"}
        sleep: {max_duration_seconds: 7200}
        safety: {gm_node_hard_limit: 500, max_consecutive_no_action: 50}
    """)
    cfg = load_config(p)
    # claude-sonnet-4-5 → 200_000 from krakey.utils.model_context
    assert cfg.llm.tags["heavy"].params.max_input_tokens == 200_000


def test_loader_old_roles_shape_exits_with_migration_message(
    tmp_path, capsys,
):
    """Detect deprecated `llm.roles:` shape and exit loud rather than
    silently mis-parsing."""
    p = _write(tmp_path, """
        llm:
          providers: {}
          roles:
            self: {provider: "X", model: "Y", params: {}}
        idle: {min_interval: 1, max_interval: 60, default_interval: 1}
        fatigue: {gm_node_soft_limit: 100, force_sleep_threshold: 60, thresholds: {}}
        graph_memory: {db_path: "x", auto_ingest_similarity_threshold: 0.9, recall_per_stimulus_k: 5, neighbor_expand_depth: 1}
        knowledge_base: {dir: "kb"}
        sleep: {max_duration_seconds: 7200}
        safety: {gm_node_hard_limit: 500, max_consecutive_no_action: 50}
    """)
    with pytest.raises(_ConfigBootstrapExit):
        load_config(p)
    err = capsys.readouterr().err
    assert "deprecated" in err.lower()
    assert "tags" in err
    assert "core_purposes" in err


def test_core_params_returns_tag_params_for_known_purpose(tmp_path):
    """LLMSection.core_params(purpose) returns the LLMParams of the
    tag that purpose is bound to."""
    cfg = Config(
        llm=LLMSection(
            providers={"P": Provider(type="openai_compatible",
                                       base_url="http://x", api_key="k")},
            tags={
                "fast": TagBinding(provider="P/m",
                                   params=LLMParams(max_output_tokens=42)),
            },
            core_purposes={"self_thinking": "fast"},
        ),
    )
    p = cfg.llm.core_params("self_thinking")
    assert p is not None
    assert p.max_output_tokens == 42
    # Unknown purpose: None (caller falls back to LLMParams() defaults)
    assert cfg.llm.core_params("does_not_exist") is None


# ---- LLMParams + schema ---------------------------------------------


def test_llm_params_universal_defaults():
    p = LLMParams()
    assert p.max_output_tokens == 4096
    assert p.max_input_tokens is None  # resolved by loader, not LLMParams
    assert p.temperature == 0.7
    assert p.reasoning_mode == "off"


def test_llm_params_schema_lists_known_fields():
    schema = llm_params_schema()
    names = {e["field"] for e in schema}
    assert {"max_output_tokens", "max_input_tokens", "temperature",
            "reasoning_mode"}.issubset(names)
    # legacy `max_tokens` (alias only) NOT in schema
    assert "max_tokens" not in names


# ---- dump / round-trip -----------------------------------------------


def test_dump_config_roundtrips_through_load(tmp_path):
    cfg = Config(
        llm=LLMSection(
            providers={"P": Provider(type="openai_compatible",
                                       base_url="http://x", api_key="k")},
            tags={"t": TagBinding(provider="P/m",
                                   params=LLMParams(max_output_tokens=999))},
            core_purposes={"self_thinking": "t"},
            embedding="t",
        ),
    )
    path = tmp_path / "roundtrip.yaml"
    path.write_text(dump_config(cfg), encoding="utf-8")
    loaded = load_config(path)
    assert "t" in loaded.llm.tags
    assert loaded.llm.tags["t"].provider == "P/m"
    assert loaded.llm.tags["t"].params.max_output_tokens == 999
    assert loaded.llm.core_purposes == {"self_thinking": "t"}
    assert loaded.llm.embedding == "t"


def test_ensure_config_creates_default_file(tmp_path):
    path = tmp_path / "fresh.yaml"
    created = ensure_config(path)
    assert created is True
    assert path.exists()
    # Calling again is a no-op
    assert ensure_config(path) is False


def test_load_config_missing_file_points_at_onboarding(tmp_path):
    """No more silent auto-generation: load_config raises with a hint
    pointing at the onboarding wizard via `krakey onboard`."""
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError) as excinfo:
        load_config(missing)
    assert "krakey onboard" in str(excinfo.value)
    # And we must NOT have created the file as a side effect.
    assert not missing.exists()


def test_default_config_is_loadable():
    """Config() with no args plus dump→load must round-trip cleanly."""
    text = dump_config(Config())
    parsed = yaml.safe_load(text)
    assert "llm" in parsed
    assert "tags" in parsed["llm"]


# ---------------- sliding_window section (2026-05-07) ----------------


def test_sliding_window_default_state_path_is_workspace_data():
    """Default Config exposes the persistence path so users see
    where the file lives without having to dig into source."""
    cfg = Config()
    assert cfg.sliding_window.state_path == "workspace/data/sliding_window.json"


def test_sliding_window_dumped_then_reloaded(tmp_path):
    """Dashboard edits write YAML, runtime restart reloads. The
    section must round-trip through dump → load."""
    cfg = Config()
    cfg.sliding_window.state_path = "/tmp/custom/sw.json"
    p = tmp_path / "config.yaml"
    p.write_text(dump_config(cfg), encoding="utf-8")
    reloaded = load_config(p)
    assert reloaded.sliding_window.state_path == "/tmp/custom/sw.json"


def test_sliding_window_empty_string_means_in_memory_only(tmp_path):
    """Setting state_path to "" in YAML opts out of persistence —
    matches the RuntimeDeps sentinel ("" → in-memory only)."""
    p = _write(tmp_path, """
        llm:
          providers: {}
          tags: {}
          core_purposes: {}
        plugins: []
        graph_memory:
          db_path: ":memory:"
        sliding_window:
          state_path: ""
    """)
    cfg = load_config(p)
    assert cfg.sliding_window.state_path == ""


def test_sliding_window_section_absent_falls_back_to_default(tmp_path):
    """Old config.yaml files without a sliding_window section keep
    working — the default path is auto-applied."""
    p = _write(tmp_path, """
        llm:
          providers: {}
          tags: {}
          core_purposes: {}
        plugins: []
        graph_memory:
          db_path: ":memory:"
    """)
    cfg = load_config(p)
    assert cfg.sliding_window.state_path == "workspace/data/sliding_window.json"


def _minimal_config_body(extra: str) -> str:
    """Return a syntactically valid config.yaml body with the
    plugin-enable section provided by the test."""
    return f"""
        llm:
          providers: {{}}
          tags: {{}}
          core_purposes: {{}}
        graph_memory:
          db_path: ":memory:"
{extra}
    """


def test_plugins_field_alone_loads_as_before(tmp_path):
    """Pure ``plugins:`` config — no legacy field — loads verbatim."""
    p = _write(tmp_path, _minimal_config_body(
        "        plugins:\n"
        "          - dashboard\n"
        "          - cli_exec\n"
    ))
    cfg = load_config(p)
    assert cfg.plugins == ["dashboard", "cli_exec"]


def test_modifiers_alone_treated_as_plugin_list_with_deprecation(
    tmp_path, capsys,
):
    """Pre-2026-04-26 layout: only ``modifiers:`` present. Loads,
    but emits a deprecation note pointing to the new field."""
    p = _write(tmp_path, _minimal_config_body(
        "        modifiers:\n"
        "          - hypothalamus\n"
    ))
    cfg = load_config(p)
    err = capsys.readouterr().err
    assert cfg.plugins == ["hypothalamus"]
    assert "modifiers" in err
    assert "pre-2026" in err.lower() or "deprecated" in err.lower() \
        or "silence" in err.lower()


def test_both_lists_merged_modifier_only_plugin_loads(tmp_path):
    """Regression: dashboard's split UX writes modifier-only plugins
    (e.g. ``hypothalamus``) ONLY to ``modifiers:`` and tool/channel
    plugins to ``plugins:``. The loader must merge both or
    modifier-only plugins silently never load — symptom: hypothalamus
    in dashboard ticked, but the [ACTION FORMAT] layer never gets
    swapped to its NL flavor because no modifier was ever registered."""
    p = _write(tmp_path, _minimal_config_body(
        "        plugins:\n"
        "          - dashboard\n"
        "          - cli_exec\n"
        "        modifiers:\n"
        "          - hypothalamus\n"
    ))
    cfg = load_config(p)
    assert "hypothalamus" in cfg.plugins
    assert "dashboard" in cfg.plugins
    assert "cli_exec" in cfg.plugins


def test_both_lists_merged_modifier_chain_order_first(tmp_path):
    """The modifier list is order-sensitive (heartbeat chain order).
    Merge must put modifier entries BEFORE plugins-list entries so
    chain order is preserved."""
    p = _write(tmp_path, _minimal_config_body(
        "        plugins:\n"
        "          - cli_exec\n"
        "        modifiers:\n"
        "          - hypothalamus\n"
        "          - recall\n"
    ))
    cfg = load_config(p)
    # Modifier-list entries land first, in their declared order.
    assert cfg.plugins.index("hypothalamus") < cfg.plugins.index("cli_exec")
    assert cfg.plugins.index("recall") < cfg.plugins.index("cli_exec")
    assert cfg.plugins.index("hypothalamus") < cfg.plugins.index("recall")


def test_both_lists_merged_dedupes_plugins_in_both(tmp_path):
    """A plugin with both modifier + tool components (e.g. ``recall``,
    ``in_mind_note``) appears in both lists. Merge must de-dup,
    keeping the modifier-list slot so chain order is honored."""
    p = _write(tmp_path, _minimal_config_body(
        "        plugins:\n"
        "          - cli_exec\n"
        "          - recall\n"
        "        modifiers:\n"
        "          - hypothalamus\n"
        "          - recall\n"
    ))
    cfg = load_config(p)
    # ``recall`` should appear once and at the modifier-list slot
    # (before ``cli_exec``), not duplicated.
    assert cfg.plugins.count("recall") == 1
    assert cfg.plugins.index("recall") < cfg.plugins.index("cli_exec")


def test_both_lists_empty_returns_empty_list(tmp_path):
    """``plugins: []`` + ``modifiers: []`` is the canonical
    "explicitly zero" shape. Result is empty list, not None."""
    p = _write(tmp_path, _minimal_config_body(
        "        plugins: []\n"
        "        modifiers: []\n"
    ))
    cfg = load_config(p)
    assert cfg.plugins == []


def test_neither_field_yields_none(tmp_path):
    """When both fields are absent the loader returns None so the
    runtime prints its no-plugins nudge."""
    p = _write(tmp_path, _minimal_config_body(""))
    cfg = load_config(p)
    assert cfg.plugins is None


def test_sliding_window_max_tokens_still_warns(tmp_path, capsys):
    """The deprecated `sliding_window.max_tokens` field has been
    removed but users may still have it from old configs. Warn
    explicitly so they remove it; honor `state_path` alongside."""
    p = _write(tmp_path, """
        llm:
          providers: {}
          tags: {}
          core_purposes: {}
        plugins: []
        graph_memory:
          db_path: ":memory:"
        sliding_window:
          max_tokens: 4096
          state_path: /tmp/sw.json
    """)
    cfg = load_config(p)
    err = capsys.readouterr().err
    assert "max_tokens" in err
    assert "deprecated" in err.lower()
    # state_path still applied even when max_tokens triggers the warning.
    assert cfg.sliding_window.state_path == "/tmp/sw.json"
