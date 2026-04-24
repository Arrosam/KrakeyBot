import os
import textwrap

import pytest
import yaml

from src.models.config import (
    Config,
    LLMParams,
    dump_config,
    ensure_config,
    llm_params_schema,
    load_config,
    role_default_params,
)


def _write(tmp_path, body):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_loads_minimal_config(tmp_path):
    path = _write(tmp_path, """
        llm:
          providers: {}
          roles: {}
        hibernate:
          min_interval: 2
          max_interval: 300
          default_interval: 30
        fatigue:
          gm_node_soft_limit: 200
          force_sleep_threshold: 120
          thresholds: {50: "a", 75: "b", 100: "c"}
        sliding_window: {max_tokens: 4096}
        graph_memory:
          db_path: "x.sqlite"
          auto_ingest_similarity_threshold: 0.9
          recall_per_stimulus_k: 5
          max_recall_nodes: 20
          neighbor_expand_depth: 1
        knowledge_base: {dir: "kb"}
        sensory: {}
        tentacle: {}
        sleep: {max_duration_seconds: 7200}
        safety: {gm_node_hard_limit: 500, max_consecutive_no_action: 50}
    """)
    cfg = load_config(path)
    assert cfg.hibernate.default_interval == 30
    assert cfg.fatigue.force_sleep_threshold == 120
    assert cfg.graph_memory.max_recall_nodes == 20


def test_env_substitution(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret123")
    path = _write(tmp_path, """
        llm:
          providers:
            p: {type: "openai_compatible", base_url: "http://x", api_key: "${MY_KEY}", models: []}
          roles: {}
        hibernate: {min_interval: 2, max_interval: 300, default_interval: 30}
        fatigue:
          gm_node_soft_limit: 200
          force_sleep_threshold: 120
          thresholds: {50: "a"}
        sliding_window: {max_tokens: 4096}
        graph_memory:
          db_path: "x"
          auto_ingest_similarity_threshold: 0.9
          recall_per_stimulus_k: 5
          max_recall_nodes: 20
          neighbor_expand_depth: 1
        knowledge_base: {dir: "kb"}
        sensory: {}
        tentacle: {}
        sleep: {max_duration_seconds: 7200}
        safety: {gm_node_hard_limit: 500, max_consecutive_no_action: 50}
    """)
    cfg = load_config(path)
    assert cfg.llm.providers["p"].api_key == "secret123"


def test_warn_when_fatigue_threshold_ge_force(tmp_path, capsys):
    path = _write(tmp_path, """
        llm: {providers: {}, roles: {}}
        hibernate: {min_interval: 2, max_interval: 300, default_interval: 30}
        fatigue:
          gm_node_soft_limit: 200
          force_sleep_threshold: 100
          thresholds: {50: "a", 120: "too-high"}
        sliding_window: {max_tokens: 4096}
        graph_memory:
          db_path: "x"
          auto_ingest_similarity_threshold: 0.9
          recall_per_stimulus_k: 5
          max_recall_nodes: 20
          neighbor_expand_depth: 1
        knowledge_base: {dir: "kb"}
        sensory: {}
        tentacle: {}
        sleep: {max_duration_seconds: 7200}
        safety: {gm_node_hard_limit: 500, max_consecutive_no_action: 50}
    """)
    load_config(path)
    captured = capsys.readouterr()
    assert "warning" in (captured.out + captured.err).lower()


def test_env_substitution_missing_var_keeps_placeholder(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    path = _write(tmp_path, """
        llm:
          providers:
            p: {type: "openai_compatible", base_url: "http://x", api_key: "${MISSING_VAR}", models: []}
          roles: {}
        hibernate: {min_interval: 2, max_interval: 300, default_interval: 30}
        fatigue:
          gm_node_soft_limit: 200
          force_sleep_threshold: 120
          thresholds: {}
        sliding_window: {max_tokens: 4096}
        graph_memory:
          db_path: "x"
          auto_ingest_similarity_threshold: 0.9
          recall_per_stimulus_k: 5
          max_recall_nodes: 20
          neighbor_expand_depth: 1
        knowledge_base: {dir: "kb"}
        sensory: {}
        tentacle: {}
        sleep: {max_duration_seconds: 7200}
        safety: {gm_node_hard_limit: 500, max_consecutive_no_action: 50}
    """)
    cfg = load_config(path)
    assert cfg.llm.providers["p"].api_key is None


def test_real_config_file_loads():
    cfg = load_config("config.yaml")
    assert cfg.hibernate.default_interval > 0
    assert cfg.hibernate.min_interval > 0
    assert cfg.graph_memory.max_recall_nodes > 0
    assert "self" in cfg.llm.roles


# ---------------- Phase 1: defaults + bootstrap ----------------


def test_default_config_has_usable_sections():
    """Config() with no args should be fully populated from defaults —
    no KeyError / None in required subsections."""
    cfg = Config()
    # Scaffolding defaults are empty so the user MUST fill them in,
    # but the sections themselves must exist and be the right type.
    assert cfg.llm.providers == {}
    assert cfg.llm.roles == {}
    # All numeric knobs have sensible non-zero defaults.
    assert cfg.hibernate.default_interval > 0
    assert cfg.fatigue.force_sleep_threshold > 0
    assert cfg.graph_memory.max_recall_nodes > 0
    assert cfg.sleep.max_duration_seconds > 0
    assert cfg.safety.gm_node_hard_limit > 0
    # Dashboard defaults on so first-run users get the UI.
    assert cfg.dashboard.enabled is True


def test_dump_config_roundtrips_through_load(tmp_path):
    """Serialize defaults → write → load. Structure must survive intact."""
    original = Config()
    path = tmp_path / "roundtrip.yaml"
    path.write_text(dump_config(original), encoding="utf-8")
    loaded = load_config(path)
    # Spot-check a mix of sections. Full equality is fragile because
    # e.g. default dicts may differ by key iteration; identity of
    # values is what matters.
    assert loaded.hibernate.default_interval == original.hibernate.default_interval
    assert loaded.fatigue.force_sleep_threshold == original.fatigue.force_sleep_threshold
    assert loaded.fatigue.thresholds == original.fatigue.thresholds
    assert loaded.sleep.max_duration_seconds == original.sleep.max_duration_seconds
    assert loaded.safety.gm_node_hard_limit == original.safety.gm_node_hard_limit
    assert loaded.dashboard.enabled == original.dashboard.enabled


def test_dump_config_produces_valid_yaml(tmp_path):
    """The dumped text must parse as YAML cleanly."""
    text = dump_config(Config())
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict)
    assert "llm" in parsed
    assert "fatigue" in parsed
    # int keys normalized to strings for portability.
    ft = parsed["fatigue"]["thresholds"]
    assert all(isinstance(k, str) for k in ft.keys())


def test_ensure_config_writes_when_missing(tmp_path):
    target = tmp_path / "nested" / "config.yaml"
    created = ensure_config(target)
    assert created is True
    assert target.exists()
    # And what it wrote must load back as a Config.
    cfg = load_config(target)
    assert cfg.hibernate.default_interval > 0


def test_ensure_config_no_op_when_present(tmp_path):
    target = tmp_path / "config.yaml"
    target.write_text("hibernate:\n  default_interval: 42\n",
                       encoding="utf-8")
    created = ensure_config(target)
    assert created is False
    # Must NOT be overwritten.
    assert "42" in target.read_text(encoding="utf-8")


def test_load_config_bootstraps_and_exits_when_missing(tmp_path, capsys):
    """First-run behavior: generate defaults at path, print guidance,
    exit. No more FileNotFoundError, no copying config.yaml.example."""
    target = tmp_path / "nope" / "config.yaml"
    assert not target.exists()
    with pytest.raises(SystemExit):
        load_config(target)
    # File now exists and contains defaults.
    assert target.exists()
    # Guidance message went to stderr.
    captured = capsys.readouterr()
    assert "Generated default config" in captured.err
    assert "api_key" in captured.err.lower() or "api key" in captured.err.lower()


def test_load_config_accepts_sparse_yaml(tmp_path):
    """A config file that only specifies some sections should load fine —
    missing sections fall back to defaults."""
    path = tmp_path / "sparse.yaml"
    path.write_text("hibernate:\n  default_interval: 99\n",
                      encoding="utf-8")
    cfg = load_config(path)
    assert cfg.hibernate.default_interval == 99
    # Unspecified sections use defaults.
    assert cfg.safety.gm_node_hard_limit == 500
    assert cfg.dashboard.enabled is True


# ---------------- Phase 2: LLMParams + role defaults ----------------


def test_llm_params_universal_defaults():
    """LLMParams() should be usable as-is — the fallback for any role
    that isn't in _ROLE_DEFAULTS."""
    p = LLMParams()
    assert p.max_output_tokens == 4096
    assert p.max_input_tokens is None
    assert p.temperature == 0.7
    assert p.reasoning_mode == "off"
    assert p.timeout_seconds == 120.0
    assert p.max_retries == 3
    assert 429 in p.retry_on_status
    # Mutable defaults must not be shared across instances.
    q = LLMParams()
    q.stop_sequences.append("X")
    assert p.stop_sequences == []


def test_role_default_params_includes_known_roles():
    """Known roles (self, hypothalamus, compact, classifier, embedding,
    reranker) all carry a defaults dict."""
    for rname in ("self", "hypothalamus", "compact", "classifier",
                  "embedding", "reranker"):
        d = role_default_params(rname)
        assert isinstance(d, dict)
        assert "timeout_seconds" in d, f"{rname} missing timeout default"


def test_role_default_params_returns_fresh_copy():
    """Callers must be able to mutate the return without poisoning the
    next call."""
    d = role_default_params("self")
    d["max_output_tokens"] = 999999
    d2 = role_default_params("self")
    assert d2.get("max_output_tokens") != 999999


def test_role_defaults_applied_when_yaml_omits_params(tmp_path):
    """When a role is declared in YAML with no params block, the role
    defaults (not the universal dataclass defaults) should land."""
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""
        llm:
          providers:
            p: {type: "openai_compatible", base_url: "http://x", api_key: "k", models: []}
          roles:
            self: {provider: "p", model: "m"}
            hypothalamus: {provider: "p", model: "h"}
    """), encoding="utf-8")
    cfg = load_config(p)
    self_params = cfg.llm.roles["self"].params
    assert self_params.max_output_tokens == 8192
    assert self_params.reasoning_mode == "medium"
    assert self_params.reasoning_budget_tokens == 4096
    assert self_params.timeout_seconds == 180.0

    hypo_params = cfg.llm.roles["hypothalamus"].params
    # Bumped from 512 → 2048: Chinese JSON with multiple tentacle_calls
    # truncates at 512 and some providers surface that as a 500
    # ("Unexpected EOF" from xunfei engine).
    assert hypo_params.max_output_tokens == 2048
    assert hypo_params.temperature == 0.0
    # response_format intentionally NOT defaulted — many OpenAI-compat
    # providers choke on it. Users opt in per-role in YAML.
    assert hypo_params.response_format is None
    assert hypo_params.timeout_seconds == 20.0


def test_yaml_params_override_role_defaults(tmp_path):
    """User-supplied params in YAML win over role defaults."""
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""
        llm:
          providers:
            p: {type: "openai_compatible", base_url: "http://x", api_key: "k", models: []}
          roles:
            self:
              provider: "p"
              model: "m"
              params:
                max_output_tokens: 2000
                reasoning_mode: "off"
                max_input_tokens: 200000
    """), encoding="utf-8")
    cfg = load_config(p)
    sp = cfg.llm.roles["self"].params
    assert sp.max_output_tokens == 2000
    assert sp.max_input_tokens == 200000
    assert sp.reasoning_mode == "off"
    # Unspecified fields still come from role defaults.
    assert sp.timeout_seconds == 180.0


def test_yaml_max_tokens_legacy_alias_accepted(tmp_path):
    """Older configs use `max_tokens` (pre-rename to max_output_tokens).
    The loader must accept it silently so existing setups don't break
    on upgrade."""
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""
        llm:
          providers:
            p: {type: "openai_compatible", base_url: "http://x", api_key: "k", models: []}
          roles:
            self:
              provider: "p"
              model: "m"
              params:
                max_tokens: 3000
    """), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.llm.roles["self"].params.max_output_tokens == 3000


def test_yaml_new_name_wins_over_legacy_alias(tmp_path):
    """If both `max_tokens` and `max_output_tokens` appear, the explicit
    new name wins — the alias is a read-only bridge."""
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""
        llm:
          providers:
            p: {type: "openai_compatible", base_url: "http://x", api_key: "k", models: []}
          roles:
            self:
              provider: "p"
              model: "m"
              params:
                max_tokens: 1111
                max_output_tokens: 2222
    """), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.llm.roles["self"].params.max_output_tokens == 2222


def test_unknown_role_gets_universal_defaults(tmp_path):
    """A role name with no entry in _ROLE_DEFAULTS should still load
    cleanly, falling back to the universal LLMParams defaults."""
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""
        llm:
          providers:
            p: {type: "openai_compatible", base_url: "http://x", api_key: "k", models: []}
          roles:
            custom_role: {provider: "p", model: "m"}
    """), encoding="utf-8")
    cfg = load_config(p)
    cp = cfg.llm.roles["custom_role"].params
    assert cp.max_output_tokens == 4096
    assert cp.temperature == 0.7
    assert cp.timeout_seconds == 120.0


def test_llm_params_schema_has_expected_fields():
    """The schema endpoint's payload must cover every LLMParams field,
    with enum choices for the two enum-shaped ones."""
    schema = llm_params_schema()
    names = {e["field"] for e in schema}
    assert {"max_output_tokens", "max_input_tokens", "temperature",
            "top_p", "stop_sequences", "response_format", "seed",
            "reasoning_mode", "reasoning_budget_tokens",
            "timeout_seconds", "max_retries", "retry_on_status"} <= names
    # The old ambiguous name must NOT be re-introduced in the schema —
    # it stays only as a YAML-input alias, not a first-class field.
    assert "max_tokens" not in names

    by_name = {e["field"]: e for e in schema}
    assert by_name["reasoning_mode"]["type"] == "enum"
    assert set(by_name["reasoning_mode"]["choices"]) == {
        "off", "low", "medium", "high"
    }
    assert by_name["response_format"]["type"] == "enum"
    assert by_name["max_output_tokens"]["type"] == "number"
    assert by_name["max_input_tokens"]["type"] == "number"
    assert by_name["temperature"]["type"] == "number_float"
    assert by_name["stop_sequences"]["type"] == "list"


def test_llm_params_schema_defaults_match_dataclass():
    """The `default` field in the schema must match LLMParams() so the
    UI pre-fills with the same value the Python loader would."""
    schema = llm_params_schema()
    d = LLMParams()
    for entry in schema:
        assert entry["default"] == getattr(d, entry["field"]), (
            f"schema default for {entry['field']} drifted from dataclass"
        )


def test_llm_params_unknown_key_in_yaml_ignored(tmp_path):
    """Forward-compat: unknown params field shouldn't blow up the
    loader (lets us roll back to an older Krakey without hand-editing
    configs)."""
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""
        llm:
          providers:
            p: {type: "openai_compatible", base_url: "http://x", api_key: "k", models: []}
          roles:
            self:
              provider: "p"
              model: "m"
              params:
                max_output_tokens: 1234
                this_field_does_not_exist: "xyz"
    """), encoding="utf-8")
    cfg = load_config(p)  # must not raise
    assert cfg.llm.roles["self"].params.max_output_tokens == 1234


def test_llm_params_roundtrip_through_dump(tmp_path):
    """A Config with custom role params must round-trip."""
    from src.models.config import LLMSection, Provider, RoleBinding

    cfg = Config()
    cfg.llm = LLMSection(
        providers={"p": Provider(type="openai_compatible",
                                  base_url="http://x", api_key="k")},
        roles={
            "self": RoleBinding(
                provider="p", model="m",
                params=LLMParams(max_output_tokens=9999,
                                   max_input_tokens=200000,
                                   reasoning_mode="high"),
            ),
        },
    )
    path = tmp_path / "roundtrip.yaml"
    path.write_text(dump_config(cfg), encoding="utf-8")
    loaded = load_config(path)
    assert loaded.llm.roles["self"].params.max_output_tokens == 9999
    assert loaded.llm.roles["self"].params.max_input_tokens == 200000
    assert loaded.llm.roles["self"].params.reasoning_mode == "high"
