import os
import textwrap

import pytest

from src.models.config import load_config


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
    assert cfg.hibernate.default_interval == 30
    assert cfg.graph_memory.max_recall_nodes == 20
    assert "self" in cfg.llm.roles
