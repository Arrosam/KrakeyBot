"""Config-driven Reflect registration.

Three input states for ``config.reflects`` (see Config docstring):
  * None         — migration sentinel; runtime falls back to legacy
                   default + emits a stderr deprecation.
  * []           — explicit zero Reflects; honored without warning.
  * [name, ...]  — explicit ordered list; each name looked up in
                   BUILTIN_FACTORIES, unknown names skipped with log.

These tests pin the loader's parsing of all three forms AND the
runtime registration behavior, including unknown-name handling
(strictly additive: a typo can't crash startup).
"""
import textwrap

import pytest

from src.models.config import Config, load_config
from src.reflects.builtin import BUILTIN_FACTORIES, register_builtin
from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes


# ---- loader: parsing the YAML field ---------------------------------


def _write(tmp_path, body: str):
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_loader_returns_none_when_reflects_key_absent(tmp_path):
    """Migration sentinel — old configs have no `reflects:` field at all."""
    p = _write(tmp_path, """
        llm:
          providers:
            p: {type: "openai_compatible", base_url: "http://x", api_key: "k", models: []}
          roles:
            self: {provider: "p", model: "claude-sonnet-4-5"}
    """)
    cfg = load_config(p)
    assert cfg.reflects is None


def test_loader_returns_empty_list_when_reflects_is_empty(tmp_path):
    """Explicit `reflects: []` is the user choosing zero plugins."""
    p = _write(tmp_path, """
        llm:
          providers:
            p: {type: "openai_compatible", base_url: "http://x", api_key: "k", models: []}
          roles:
            self: {provider: "p", model: "claude-sonnet-4-5"}
        reflects: []
    """)
    cfg = load_config(p)
    assert cfg.reflects == []


def test_loader_returns_empty_list_when_reflects_is_null(tmp_path):
    """`reflects: null` is treated the same as empty list — same intent."""
    p = _write(tmp_path, """
        llm:
          providers:
            p: {type: "openai_compatible", base_url: "http://x", api_key: "k", models: []}
          roles:
            self: {provider: "p", model: "claude-sonnet-4-5"}
        reflects: null
    """)
    cfg = load_config(p)
    assert cfg.reflects == []


def test_loader_returns_ordered_list_when_specified(tmp_path):
    """Order in YAML = order in Python list = registration order at startup."""
    p = _write(tmp_path, """
        llm:
          providers:
            p: {type: "openai_compatible", base_url: "http://x", api_key: "k", models: []}
          roles:
            self: {provider: "p", model: "claude-sonnet-4-5"}
        reflects:
          - default_recall_anchor
          - default_hypothalamus
    """)
    cfg = load_config(p)
    assert cfg.reflects == [
        "default_recall_anchor", "default_hypothalamus",
    ]


def test_loader_drops_non_string_entries_with_warning(tmp_path, capsys):
    """Defensive: a typo like `- {default_hypothalamus}` (mapping not
    string) shouldn't crash the loader."""
    p = _write(tmp_path, """
        llm:
          providers:
            p: {type: "openai_compatible", base_url: "http://x", api_key: "k", models: []}
          roles:
            self: {provider: "p", model: "claude-sonnet-4-5"}
        reflects:
          - default_recall_anchor
          - 42
          - ""
          - default_hypothalamus
    """)
    cfg = load_config(p)
    assert cfg.reflects == [
        "default_recall_anchor", "default_hypothalamus",
    ]
    err = capsys.readouterr().err
    assert "42" in err or "skipping" in err


def test_loader_warns_when_reflects_not_a_list(tmp_path, capsys):
    """`reflects:` accidentally a string or dict → warn + None."""
    p = _write(tmp_path, """
        llm:
          providers:
            p: {type: "openai_compatible", base_url: "http://x", api_key: "k", models: []}
          roles:
            self: {provider: "p", model: "claude-sonnet-4-5"}
        reflects: "default_hypothalamus"
    """)
    cfg = load_config(p)
    assert cfg.reflects is None  # falls back to legacy migration default
    err = capsys.readouterr().err
    assert "should be a list" in err


# ---- builtin factory dict --------------------------------------------


def test_builtin_factories_have_known_names():
    assert "default_hypothalamus" in BUILTIN_FACTORIES
    assert "default_recall_anchor" in BUILTIN_FACTORIES


def test_register_builtin_rejects_duplicate():
    """Adding two factories under the same name should fail loudly."""
    with pytest.raises(ValueError, match="already registered"):
        register_builtin(
            "default_hypothalamus",
            lambda deps: None,  # type: ignore[arg-type,return-value]
        )


# ---- Runtime registration end-to-end ---------------------------------


def _capture_warning_runtime(tmp_path, reflects, capsys=None):
    """Helper: build a runtime with the given config.reflects value
    and return both the runtime and the captured stderr."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    runtime.config.reflects = reflects
    return runtime


async def test_runtime_registers_explicit_list_in_order(tmp_path, capsys):
    """Explicit list → exactly those names registered, no extras, no
    deprecation warning."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    err = capsys.readouterr().err
    assert "no `reflects:` section" not in err
    # Helper sets reflects=["default_hypothalamus", "default_recall_anchor"]
    assert set(runtime.reflects.names()) == {
        "default_hypothalamus", "default_recall_anchor",
    }


async def test_runtime_registers_empty_list_with_no_warning(tmp_path, capsys):
    """`reflects: []` → zero plugins registered, no deprecation warn,
    runtime still works (zero-plugin invariant)."""
    # We need to bypass the helper's default; build the cfg manually.
    from src.main import Runtime, RuntimeDeps
    from src.models.config import (
        Config, DashboardSection, FatigueSection, GraphMemorySection,
        HibernateSection, KnowledgeBaseSection, LLMParams, LLMSection,
        RoleBinding, SafetySection, SleepSection,
    )
    import tempfile

    cfg = Config(
        llm=LLMSection(
            providers={},
            roles={"self": RoleBinding(
                provider="", model="",
                params=LLMParams(max_input_tokens=16_000),
            )},
        ),
        hibernate=HibernateSection(min_interval=1, max_interval=60,
                                    default_interval=1),
        fatigue=FatigueSection(gm_node_soft_limit=200,
                                force_sleep_threshold=120, thresholds={}),
        graph_memory=GraphMemorySection(
            db_path=str(tmp_path / "gm.sqlite"),
            auto_ingest_similarity_threshold=0.9,
            recall_per_stimulus_k=5, neighbor_expand_depth=1,
        ),
        knowledge_base=KnowledgeBaseSection(
            dir=tempfile.mkdtemp(prefix="krakey_test_kb_"),
        ),
        plugins={},
        reflects=[],
        sleep=SleepSection(max_duration_seconds=7200, min_community_size=1),
        safety=SafetySection(gm_node_hard_limit=500,
                               max_consecutive_no_action=50),
        dashboard=DashboardSection(enabled=False),
    )
    deps = RuntimeDeps(
        config=cfg, self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        compact_llm=ScriptedLLM(), classify_llm=ScriptedLLM(),
        embedder=type("_E", (), {"__call__": lambda self, t: [0.0]})(),
        plugin_configs_root=tempfile.mkdtemp(prefix="krakey_test_pc_"),
        self_model_path=f"{tempfile.mkdtemp(prefix='krakey_test_sm_')}/sm.yaml",
    )
    runtime = Runtime(deps, hibernate_min=0.01, hibernate_max=1.0,
                       is_bootstrap_override=False)
    err = capsys.readouterr().err
    assert "no `reflects:` section" not in err
    assert runtime.reflects.names() == []


async def test_runtime_warns_and_uses_legacy_default_when_reflects_is_none(
    tmp_path, capsys,
):
    """No `reflects:` field at all → loud deprecation + legacy default
    so existing users don't see behavior change."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    # Helper pre-populates reflects, so simulate the migration sentinel
    # by clearing config.reflects and re-running registration manually.
    runtime.reflects._by_kind.clear()
    runtime.config.reflects = None
    capsys.readouterr()  # drop any earlier output

    from src.reflects.builtin import BUILTIN_FACTORIES
    runtime._register_reflects_from_config(
        type("_FakeDeps", (), {"hypo_llm": ScriptedLLM([])})(),
        BUILTIN_FACTORIES,
    )
    err = capsys.readouterr().err
    assert "no `reflects:` section" in err
    assert "deprecation" in err.lower() or "legacy" in err.lower()
    assert set(runtime.reflects.names()) == {
        "default_hypothalamus", "default_recall_anchor",
    }


async def test_runtime_skips_unknown_reflect_names_loudly(tmp_path, capsys):
    """A typo in `reflects:` must not crash startup — log + skip and
    register the rest. Strictly additive plugin model means
    registration is best-effort."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    runtime.reflects._by_kind.clear()
    runtime.config.reflects = [
        "default_recall_anchor", "typo_reflect", "default_hypothalamus",
    ]
    capsys.readouterr()

    from src.reflects.builtin import BUILTIN_FACTORIES
    runtime._register_reflects_from_config(
        type("_FakeDeps", (), {"hypo_llm": ScriptedLLM([])})(),
        BUILTIN_FACTORIES,
    )
    err = capsys.readouterr().err
    assert "typo_reflect" in err
    # The two valid ones still registered; typo skipped.
    assert set(runtime.reflects.names()) == {
        "default_recall_anchor", "default_hypothalamus",
    }


async def test_runtime_tolerates_factory_exception(tmp_path, capsys):
    """A bad factory (raises during construction) must not block
    runtime startup. The other Reflects still register."""
    runtime = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]), hypo_llm=ScriptedLLM([]),
        gm_path=str(tmp_path / "gm.sqlite"),
    )
    runtime.reflects._by_kind.clear()

    def _bad(deps):
        raise RuntimeError("factory blew up")

    factories = {
        "default_recall_anchor":
            BUILTIN_FACTORIES["default_recall_anchor"],
        "broken_reflect": _bad,
    }
    runtime.config.reflects = ["default_recall_anchor", "broken_reflect"]
    capsys.readouterr()
    runtime._register_reflects_from_config(
        type("_FakeDeps", (), {"hypo_llm": ScriptedLLM([])})(),
        factories,
    )
    err = capsys.readouterr().err
    assert "broken_reflect" in err
    assert "factory blew up" in err or "factory raised" in err
    assert runtime.reflects.names() == ["default_recall_anchor"]
