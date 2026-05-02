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
