import os
import textwrap

import pytest
import yaml

from src.models.config import (
    Config,
    dump_config,
    ensure_config,
    load_config,
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
