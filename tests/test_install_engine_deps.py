"""Edge tests for the engine-dependency pipeline added to
``krakey/install/service.py``.

New surface under test
----------------------
  collect_engine_dependencies(cfg) -> dict[str, list[str]]
  collect_engine_post_install(cfg) -> dict[str, list[dict]]
  _load_config_for_install()       -> Config | None
  deps_hash(plugin_deps, post_install=None,
            engine_deps=None, engine_post=None) -> str   (extended sig)
  has_pending_deps()               (now folds engine deps)
  install(args)                    (writes engine_installed to state)
  DefaultInstallService().deps_status()  (new "engines" key)

Tests monkeypatch ``krakey.install.service.load_slot_meta`` so no
real engine meta.yaml files are touched, and stub
``subprocess.call`` so pip never runs.

Conventions match ``tests/test_install_command.py``:
  - ``isolated_workspace`` fixture (monkeypatch.chdir to tmp_path)
  - ``monkeypatch.setattr(install_mod.subprocess, "call", ...)``
  - YAML-based Config objects via load_config on a written fixture,
    OR SimpleNamespace duck-types for the smallest cases.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from krakey.engine_system.catalog import EngineImpl
from krakey.models.config import Config, load_config
from krakey.models.config.core_impls import CoreImplementations
from krakey.install import service as install_mod


# =====================================================================
# Constants — the 11 canonical slot names
# =====================================================================

_ALL_SLOTS = [
    "embedder", "reranker", "memory", "context", "explicit_history",
    "decision", "recall", "heartbeat", "dispatch",
    "llm_factory", "llm_client_factory",
]


# =====================================================================
# Test-seam helpers
# =====================================================================


def _make_engine_impl(
    *,
    dependencies: list[str] | None = None,
    post_install: list[dict[str, Any]] | None = None,
) -> EngineImpl:
    """Build a minimal EngineImpl with controllable deps fields."""
    return EngineImpl(
        cls=object,
        description="test engine",
        config_schema=[],
        dependencies=list(dependencies or []),
        post_install=list(post_install or []),
    )


def _fake_load_slot_meta(catalog_by_slot: dict[str, Any]):
    """Return a callable that mimics load_slot_meta from a fixture dict.

    ``catalog_by_slot`` maps slot-name -> ``(catalog_dict, default_name)``
    tuples. Slots absent from the dict raise FileNotFoundError, matching
    the real meta_loader's behaviour on a missing meta.yaml.
    """
    def _impl(slot: str, *, engines_root=None):
        if slot not in catalog_by_slot:
            raise FileNotFoundError(f"no meta for {slot!r}")
        return catalog_by_slot[slot]
    return _impl


def _stub_noop_subprocess(monkeypatch) -> list[list[str]]:
    """Stub subprocess.call to succeed (rc=0) and record every invocation."""
    captured: list[list[str]] = []

    def fake_call(cmd, *a, **kw):
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(install_mod.subprocess, "call", fake_call)
    return captured


# =====================================================================
# Config construction helpers
# =====================================================================


def _make_cfg(slot_overrides: dict[str, str]) -> Config:
    """Build a Config whose core_implementations slots are set from
    ``slot_overrides``.  Any slot not listed is left as ''."""
    return Config(
        core_implementations=CoreImplementations(**{
            k: v for k, v in slot_overrides.items()
            if k in CoreImplementations.__dataclass_fields__
        })
    )


def _minimal_yaml_config(tmp_path: Path, *, memory: str = "") -> Path:
    """Write a minimal valid config.yaml to tmp_path and return its Path."""
    body = f"""\
        core_implementations:
          memory: "{memory}"
        idle:
          min_interval: 1
          max_interval: 60
          default_interval: 1
    """
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def _write_config_yaml(tmp_path: Path, body: str) -> Path:
    """Write arbitrary YAML to config.yaml in tmp_path."""
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def isolated_workspace(monkeypatch, tmp_path: Path):
    """chdir to tmp_path so workspace/ + INSTALL_STATE_PATH resolve
    relative to a clean directory — matches test_install_command.py."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(tmp_path)
    return workspace


@pytest.fixture
def simple_catalog(monkeypatch):
    """Patch load_slot_meta with a small catalog covering memory + recall.

    memory:
      graph_memory  (default, no deps)
      memos         (dependencies=["MemoryOS>=2.0.17,<3.0"],
                     post_install=[{args:["{python}","-m","memos","setup"],
                                    description:"memos daemon",
                                    optional:false}])
    recall:
      default_recall (default, no deps)
    """
    memos_post = [{"args": ["{python}", "-m", "memos", "setup"],
                   "description": "memos daemon", "optional": False}]
    catalog = {
        "memory": (
            {
                "graph_memory": _make_engine_impl(),
                "memos": _make_engine_impl(
                    dependencies=["MemoryOS>=2.0.17,<3.0"],
                    post_install=memos_post,
                ),
            },
            "graph_memory",   # default
        ),
        "recall": (
            {"default_recall": _make_engine_impl()},
            "default_recall",
        ),
    }
    monkeypatch.setattr(install_mod, "load_slot_meta",
                        _fake_load_slot_meta(catalog))
    return catalog


# =====================================================================
# 1. collect_engine_dependencies — positive / equivalence
# =====================================================================


class TestCollectEngineDependenciesPositive:
    """Happy-path behaviour of collect_engine_dependencies."""

    def test_explicit_selection_returns_selected_engine_deps(
        self, simple_catalog, monkeypatch,
    ):
        """core_implementations.memory='memos' → result contains the
        memos dependency spec under the namespaced key."""
        cfg = _make_cfg({"memory": "memos"})
        result = install_mod.collect_engine_dependencies(cfg)
        key = "engine:memory:memos"
        assert key in result
        assert "MemoryOS>=2.0.17,<3.0" in result[key]

    def test_blank_slot_resolves_to_default_engine(
        self, simple_catalog, monkeypatch,
    ):
        """core_implementations.memory='' → default 'graph_memory'
        is resolved; its deps (empty) are returned (or the key is
        absent — both acceptable per spec)."""
        cfg = _make_cfg({"memory": ""})
        result = install_mod.collect_engine_dependencies(cfg)
        # graph_memory has no deps; key may be absent or present with [].
        default_key = "engine:memory:graph_memory"
        if default_key in result:
            assert result[default_key] == []

    def test_slot_with_deps_only_that_slot_key_carries_non_empty_list(
        self, simple_catalog, monkeypatch,
    ):
        """Only slots whose selected engine actually declares dependencies
        contribute non-empty entries; the dict values are pip spec strings."""
        cfg = _make_cfg({"memory": "memos", "recall": ""})
        result = install_mod.collect_engine_dependencies(cfg)
        # memos has deps
        assert result["engine:memory:memos"] == ["MemoryOS>=2.0.17,<3.0"]
        # default_recall has no deps — may be absent or [] but NOT wrong pkg
        recall_key = "engine:recall:default_recall"
        if recall_key in result:
            assert result[recall_key] == []

    def test_multiple_slots_with_deps_each_keyed_independently(
        self, monkeypatch,
    ):
        """Two slots both selecting engines with deps produce two
        independent entries with their respective specs."""
        catalog = {
            "embedder": (
                {
                    "fast_embed": _make_engine_impl(
                        dependencies=["fastembed>=0.3"]
                    ),
                },
                "fast_embed",
            ),
            "reranker": (
                {
                    "bge_reranker": _make_engine_impl(
                        dependencies=["FlagEmbedding>=1.2"]
                    ),
                },
                "bge_reranker",
            ),
        }
        monkeypatch.setattr(install_mod, "load_slot_meta",
                            _fake_load_slot_meta(catalog))
        cfg = _make_cfg({"embedder": "fast_embed", "reranker": "bge_reranker"})
        result = install_mod.collect_engine_dependencies(cfg)
        assert "fastembed>=0.3" in result["engine:embedder:fast_embed"]
        assert "FlagEmbedding>=1.2" in result["engine:reranker:bge_reranker"]

    def test_result_keys_use_engine_slot_shortname_namespace_format(
        self, simple_catalog, monkeypatch,
    ):
        """Key format is strictly 'engine:<slot>:<short_name>'."""
        cfg = _make_cfg({"memory": "memos"})
        result = install_mod.collect_engine_dependencies(cfg)
        for key in result:
            parts = key.split(":")
            assert parts[0] == "engine", f"key {key!r} does not start with 'engine:'"
            assert len(parts) == 3, f"key {key!r} does not have 3 colon-separated parts"


# =====================================================================
# 2. collect_engine_dependencies — boundary value analysis
# =====================================================================


class TestCollectEngineDependenciesBVA:
    """Boundary conditions for collect_engine_dependencies."""

    def test_dotted_path_override_skips_slot(self, simple_catalog, monkeypatch):
        """A value containing ':' is a dotted-path override → SKIP; no
        entry appears in result for that slot."""
        cfg = _make_cfg({"memory": "mymod.path:DottedClass"})
        result = install_mod.collect_engine_dependencies(cfg)
        # No key for memory slot — dotted-path form is not catalogued
        memory_keys = [k for k in result if k.startswith("engine:memory:")]
        assert memory_keys == []

    def test_whitespace_only_slot_treated_as_blank_uses_default(
        self, simple_catalog, monkeypatch,
    ):
        """A value of '  ' (whitespace) is treated the same as blank —
        resolved to the meta's default engine."""
        cfg = _make_cfg({"memory": "   "})
        result = install_mod.collect_engine_dependencies(cfg)
        # Should resolve to graph_memory default (no deps → absent or [])
        dotted_keys = [k for k in result if "memory" in k and ":" in k[7:]]
        for k in dotted_keys:
            # None of these should be a whitespace-value key
            assert "   " not in k

    def test_all_slots_blank_returns_only_default_engine_entries(
        self, monkeypatch,
    ):
        """When every slot is blank, we get at most the defaults for
        slots that have meta. Slots with missing meta are silently
        skipped. No crash."""
        # Provide a catalog for every canonical slot, all with no deps.
        catalog = {
            slot: (
                {f"{slot}_default": _make_engine_impl()},
                f"{slot}_default",
            )
            for slot in _ALL_SLOTS
        }
        monkeypatch.setattr(install_mod, "load_slot_meta",
                            _fake_load_slot_meta(catalog))
        cfg = _make_cfg({})  # all blank
        result = install_mod.collect_engine_dependencies(cfg)
        # All engines have no deps — result is either empty or all-empty values
        for deps in result.values():
            assert deps == []

    def test_single_slot_with_single_dep_appears_exactly_once(
        self, monkeypatch,
    ):
        """Minimal case: exactly one slot selected, one dep string."""
        catalog = {
            "memory": (
                {"single": _make_engine_impl(dependencies=["pkg==1.0"])},
                "single",
            ),
        }
        monkeypatch.setattr(install_mod, "load_slot_meta",
                            _fake_load_slot_meta(catalog))
        cfg = _make_cfg({"memory": "single"})
        result = install_mod.collect_engine_dependencies(cfg)
        assert result["engine:memory:single"] == ["pkg==1.0"]

    def test_engine_with_empty_deps_list_does_not_pollute_result(
        self, monkeypatch,
    ):
        """A slot explicitly declaring dependencies: [] should not produce
        a non-empty entry in the result."""
        catalog = {
            "memory": (
                {"no_deps_engine": _make_engine_impl(dependencies=[])},
                "no_deps_engine",
            ),
        }
        monkeypatch.setattr(install_mod, "load_slot_meta",
                            _fake_load_slot_meta(catalog))
        cfg = _make_cfg({"memory": "no_deps_engine"})
        result = install_mod.collect_engine_dependencies(cfg)
        key = "engine:memory:no_deps_engine"
        if key in result:
            assert result[key] == []


# =====================================================================
# 3. collect_engine_dependencies — negative / error-guessing
# =====================================================================


class TestCollectEngineDependenciesNegative:
    """Error paths and stale-config cases."""

    def test_stale_config_unknown_short_name_skips_slot(
        self, simple_catalog, monkeypatch,
    ):
        """core_implementations.memory='doesnotexist' — the name is not
        in the catalog → slot SKIPPED, no abort, no entry in result."""
        cfg = _make_cfg({"memory": "doesnotexist"})
        result = install_mod.collect_engine_dependencies(cfg)
        memory_keys = [k for k in result if "memory" in k]
        assert memory_keys == []

    def test_missing_meta_yaml_skips_slot(self, monkeypatch):
        """When load_slot_meta raises FileNotFoundError for a slot,
        that slot is silently skipped — no crash, no entry."""
        # Provide catalog only for 'recall'; 'memory' raises FileNotFoundError
        catalog = {
            "recall": (
                {"rc": _make_engine_impl(dependencies=["recall-pkg>=1"])},
                "rc",
            ),
        }
        monkeypatch.setattr(install_mod, "load_slot_meta",
                            _fake_load_slot_meta(catalog))
        cfg = _make_cfg({"memory": "something", "recall": "rc"})
        result = install_mod.collect_engine_dependencies(cfg)
        memory_keys = [k for k in result if "memory" in k]
        assert memory_keys == []
        # recall still present
        assert "engine:recall:rc" in result

    def test_meta_parse_error_skips_slot(self, monkeypatch):
        """When load_slot_meta raises MetaParseError, that slot is skipped."""
        from krakey.engine_system.meta_loader import MetaParseError

        def _raising_loader(slot: str, *, engines_root=None):
            if slot == "memory":
                raise MetaParseError("simulated parse error")
            raise FileNotFoundError(f"no meta for {slot}")

        monkeypatch.setattr(install_mod, "load_slot_meta", _raising_loader)
        cfg = _make_cfg({"memory": "anything"})
        result = install_mod.collect_engine_dependencies(cfg)
        memory_keys = [k for k in result if "memory" in k]
        assert memory_keys == []

    def test_does_not_abort_on_multiple_bad_slots(self, monkeypatch):
        """Even if most slots raise FileNotFoundError, the one good slot
        is still collected and no exception propagates out."""
        catalog = {
            "dispatch": (
                {"fast_dispatch": _make_engine_impl(dependencies=["fdp>=0.1"])},
                "fast_dispatch",
            ),
        }
        monkeypatch.setattr(install_mod, "load_slot_meta",
                            _fake_load_slot_meta(catalog))
        # Set every slot; most will have no meta → skipped
        overrides = {s: "some_engine" for s in _ALL_SLOTS}
        overrides["dispatch"] = "fast_dispatch"
        cfg = _make_cfg(overrides)
        result = install_mod.collect_engine_dependencies(cfg)
        assert "engine:dispatch:fast_dispatch" in result

    def test_returns_empty_dict_when_all_slots_blank_and_no_meta(
        self, monkeypatch,
    ):
        """All slots blank + no meta for any slot → returns {} (or dict
        with only empty-list values). Nothing raises."""
        monkeypatch.setattr(install_mod, "load_slot_meta",
                            _fake_load_slot_meta({}))
        cfg = _make_cfg({})
        result = install_mod.collect_engine_dependencies(cfg)
        for deps in result.values():
            assert deps == []


# =====================================================================
# 4. collect_engine_post_install — positive + boundary
# =====================================================================


class TestCollectEnginePostInstall:
    """collect_engine_post_install mirrors collect_engine_dependencies
    but returns post_install entries instead of dep strings."""

    def test_selected_engine_with_post_install_returned(
        self, simple_catalog, monkeypatch,
    ):
        """memos declares a post_install entry; it appears under the
        correct namespaced key."""
        cfg = _make_cfg({"memory": "memos"})
        result = install_mod.collect_engine_post_install(cfg)
        key = "engine:memory:memos"
        assert key in result
        entries = result[key]
        assert isinstance(entries, list)
        assert len(entries) >= 1
        args = entries[0]["args"]
        assert "-m" in args
        assert "memos" in args

    def test_engine_with_no_post_install_absent_or_empty(
        self, simple_catalog, monkeypatch,
    ):
        """graph_memory has no post_install; the key is absent or maps
        to an empty list — NOT a non-empty list."""
        cfg = _make_cfg({"memory": "graph_memory"})
        result = install_mod.collect_engine_post_install(cfg)
        key = "engine:memory:graph_memory"
        if key in result:
            assert result[key] == []

    def test_dotted_path_slot_skipped_in_post_install_too(
        self, simple_catalog, monkeypatch,
    ):
        """Dotted-path override must be skipped in post_install collection
        the same way it is in dep collection."""
        cfg = _make_cfg({"memory": "some.module:SomeClass"})
        result = install_mod.collect_engine_post_install(cfg)
        memory_keys = [k for k in result if "memory" in k]
        assert memory_keys == []

    def test_stale_config_unknown_name_skips_post_install_too(
        self, simple_catalog, monkeypatch,
    ):
        """Unknown short-name → no post_install entry (same skip rule)."""
        cfg = _make_cfg({"memory": "nonexistent_engine"})
        result = install_mod.collect_engine_post_install(cfg)
        memory_keys = [k for k in result if "memory" in k]
        assert memory_keys == []

    def test_post_install_entry_has_expected_shape(
        self, simple_catalog, monkeypatch,
    ):
        """Each post_install entry must have 'args' (list) key at minimum."""
        cfg = _make_cfg({"memory": "memos"})
        result = install_mod.collect_engine_post_install(cfg)
        for entry in result["engine:memory:memos"]:
            assert "args" in entry
            assert isinstance(entry["args"], list)

    def test_python_token_preserved_in_collected_entries(
        self, monkeypatch,
    ):
        """The raw '{python}' token is NOT expanded by collect_engine_post_install
        — expansion happens at subprocess dispatch time (expand_python_token)."""
        post = [{"args": ["{python}", "-m", "something"], "optional": False}]
        catalog = {
            "memory": (
                {"eng": _make_engine_impl(post_install=post)},
                "eng",
            ),
        }
        monkeypatch.setattr(install_mod, "load_slot_meta",
                            _fake_load_slot_meta(catalog))
        cfg = _make_cfg({"memory": "eng"})
        result = install_mod.collect_engine_post_install(cfg)
        entry_args = result["engine:memory:eng"][0]["args"]
        assert entry_args[0] == "{python}"


# =====================================================================
# 5. _load_config_for_install — all paths
# =====================================================================


class TestLoadConfigForInstall:
    """_load_config_for_install() reads config.yaml relative to cwd."""

    def test_returns_config_when_file_exists(self, monkeypatch, tmp_path):
        """A well-formed config.yaml in cwd is parsed and returned."""
        p = _minimal_yaml_config(tmp_path)
        monkeypatch.chdir(tmp_path)
        cfg = install_mod._load_config_for_install()
        assert cfg is not None
        assert isinstance(cfg, Config)

    def test_returns_none_when_config_yaml_missing(
        self, monkeypatch, tmp_path, capsys,
    ):
        """No config.yaml → returns None (not raises)."""
        monkeypatch.chdir(tmp_path)
        result = install_mod._load_config_for_install()
        assert result is None

    def test_stderr_note_when_config_yaml_missing(
        self, monkeypatch, tmp_path, capsys,
    ):
        """A missing config.yaml produces a diagnostic to stderr."""
        monkeypatch.chdir(tmp_path)
        install_mod._load_config_for_install()
        err = capsys.readouterr().err
        assert err  # something was printed to stderr

    def test_returns_none_on_parse_error(
        self, monkeypatch, tmp_path, capsys,
    ):
        """A structurally invalid config.yaml returns None, not a raised
        exception. The install pipeline must not crash."""
        p = tmp_path / "config.yaml"
        p.write_text(
            "this is: [not: valid yaml: {for: our schema",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        result = install_mod._load_config_for_install()
        assert result is None

    def test_stderr_warning_on_parse_error(
        self, monkeypatch, tmp_path, capsys,
    ):
        """A parse failure emits a warning to stderr."""
        p = tmp_path / "config.yaml"
        p.write_text("{malformed: yaml: oops", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        install_mod._load_config_for_install()
        err = capsys.readouterr().err
        assert err

    def test_parsed_core_implementations_reflects_yaml(
        self, monkeypatch, tmp_path,
    ):
        """The returned Config's core_implementations mirrors the file."""
        _write_config_yaml(tmp_path, """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        monkeypatch.chdir(tmp_path)
        cfg = install_mod._load_config_for_install()
        assert cfg is not None
        assert cfg.core_implementations.memory == "memos"


# =====================================================================
# 6. deps_hash — extended signature backward compat + new params
# =====================================================================


class TestDepsHashExtended:
    """deps_hash(plugin_deps, post_install, engine_deps, engine_post)."""

    def test_old_two_arg_call_still_works(self):
        """Existing callers that pass only (plugin_deps, post_install)
        must not break — new params default gracefully."""
        h = install_mod.deps_hash({"a": ["pkg>=1"]}, {"a": []})
        assert isinstance(h, str)
        assert len(h) == 64  # sha256 hexdigest

    def test_old_one_arg_call_still_works(self):
        """Callers passing just plugin_deps must still get a hash."""
        h = install_mod.deps_hash({"a": ["pkg>=1"]})
        assert isinstance(h, str)
        assert len(h) == 64

    def test_all_defaults_returns_valid_hash(self):
        """deps_hash({}) with all optional params absent must not crash."""
        h = install_mod.deps_hash({})
        assert isinstance(h, str)
        assert len(h) == 64

    def test_engine_deps_change_changes_hash(self):
        """Adding engine_deps with non-empty specs changes the hash."""
        base = install_mod.deps_hash({})
        with_engine = install_mod.deps_hash(
            {}, engine_deps={"engine:memory:memos": ["MemoryOS>=2.0.17"]}
        )
        assert base != with_engine

    def test_engine_post_change_changes_hash(self):
        """Adding engine_post entries changes the hash."""
        base = install_mod.deps_hash({})
        with_post = install_mod.deps_hash(
            {}, engine_post={
                "engine:memory:memos": [
                    {"args": ["{python}", "-m", "memos", "setup"],
                     "optional": False}
                ]
            }
        )
        assert base != with_post

    def test_engine_deps_hash_stable_across_key_order(self):
        """The engine_deps hash is order-independent (sorted union)."""
        h1 = install_mod.deps_hash(
            {},
            engine_deps={
                "engine:memory:memos": ["MemoryOS>=2.0.17"],
                "engine:embedder:fast_embed": ["fastembed>=0.3"],
            },
        )
        h2 = install_mod.deps_hash(
            {},
            engine_deps={
                "engine:embedder:fast_embed": ["fastembed>=0.3"],
                "engine:memory:memos": ["MemoryOS>=2.0.17"],
            },
        )
        assert h1 == h2

    def test_engine_deps_combined_with_plugin_deps_produces_stable_hash(self):
        """Mixing plugin_deps + engine_deps: result must be stable and
        distinct from either alone."""
        plugin_only = install_mod.deps_hash({"alpha": ["x>=1"]})
        engine_only = install_mod.deps_hash(
            {}, engine_deps={"engine:memory:memos": ["y>=2"]}
        )
        combined = install_mod.deps_hash(
            {"alpha": ["x>=1"]},
            engine_deps={"engine:memory:memos": ["y>=2"]},
        )
        assert combined != plugin_only
        assert combined != engine_only

    def test_engine_deps_with_duplicate_spec_same_as_single(self):
        """Same spec appearing in two engine entries folds to a single
        occurrence in the hash (set union), matching plugin behaviour."""
        h_once = install_mod.deps_hash(
            {}, engine_deps={"engine:memory:memos": ["pkg>=1"]}
        )
        h_dup = install_mod.deps_hash(
            {},
            engine_deps={
                "engine:memory:memos": ["pkg>=1"],
                "engine:recall:rc": ["pkg>=1"],
            },
        )
        assert h_once == h_dup

    def test_engine_none_same_as_empty_dict(self):
        """engine_deps=None and engine_deps={} must produce the same hash."""
        h_none = install_mod.deps_hash({}, engine_deps=None)
        h_empty = install_mod.deps_hash({}, engine_deps={})
        assert h_none == h_empty

    def test_engine_post_none_same_as_empty_dict(self):
        """engine_post=None and engine_post={} must produce the same hash."""
        h_none = install_mod.deps_hash({}, engine_post=None)
        h_empty = install_mod.deps_hash({}, engine_post={})
        assert h_none == h_empty

    def test_positional_post_install_still_accepted(self):
        """Callers using deps_hash(deps, post_install) positionally
        must still work after the signature extension."""
        h = install_mod.deps_hash(
            {"a": ["pkg>=1"]},
            {"a": [{"args": ["echo", "hi"], "optional": False}]},
        )
        assert isinstance(h, str)
        assert len(h) == 64


# =====================================================================
# 7. has_pending_deps — engine deps folded into hash
# =====================================================================


class TestHasPendingDepsWithEngineDeps:
    """has_pending_deps() must fold engine deps into the hash comparison."""

    def test_pending_true_before_any_install(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """Without any state file, pending=True regardless of engine config."""
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        pending, _ = install_mod.has_pending_deps()
        assert pending is True

    def test_pending_false_when_state_hash_includes_engine_deps(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """After writing a state whose hash was computed WITH engine deps,
        has_pending_deps() returns False."""
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        # Simulate what install() will write: compute a matching hash
        plugin_deps = install_mod.collect_plugin_dependencies()
        plugin_post = install_mod.collect_plugin_post_install()
        cfg = install_mod._load_config_for_install()
        if cfg is not None:
            engine_deps = install_mod.collect_engine_dependencies(cfg)
            engine_post = install_mod.collect_engine_post_install(cfg)
        else:
            engine_deps = {}
            engine_post = {}
        h = install_mod.deps_hash(
            plugin_deps, plugin_post,
            engine_deps=engine_deps, engine_post=engine_post,
        )
        install_mod.write_install_state({"deps_hash": h, "installed": [],
                                         "engine_installed": [], "installed_at": "x"})
        pending, _ = install_mod.has_pending_deps()
        assert pending is False

    def test_pending_true_when_engine_selection_changes_after_install(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """Installing with memory='' then changing to memory='memos'
        makes has_pending_deps() return True again (hash changed because
        the selected engine's deps changed)."""
        _stub_noop_subprocess(monkeypatch)

        # Install with default (no engine deps)
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: ""
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        install_mod.install(argparse.Namespace(dry_run=False, upgrade=False))
        assert install_mod.has_pending_deps()[0] is False

        # Now switch to memos (which has deps) — config changes on disk
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        pending, _ = install_mod.has_pending_deps()
        assert pending is True

    def test_no_config_yaml_engine_deps_treated_as_empty(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """When _load_config_for_install() returns None (no config.yaml),
        engine_deps is {} and the hash is computed without engine entries.
        has_pending_deps() still returns (True/False, dict) without crashing."""
        # No config.yaml written — just check it doesn't raise
        result = install_mod.has_pending_deps()
        assert isinstance(result, tuple)
        assert isinstance(result[0], bool)


# =====================================================================
# 8. install(args) — engine_installed written to state
# =====================================================================


class TestInstallEngineInstalled:
    """install() must write engine_installed to install_state.json."""

    def test_install_writes_engine_installed_field(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """After a successful install with memos selected, state has
        'engine_installed' containing the namespaced key."""
        _stub_noop_subprocess(monkeypatch)
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        rc = install_mod.install(
            argparse.Namespace(dry_run=False, upgrade=False)
        )
        assert rc == 0
        state = json.loads(
            install_mod.INSTALL_STATE_PATH.read_text(encoding="utf-8")
        )
        assert "engine_installed" in state
        assert "engine:memory:memos" in state["engine_installed"]

    def test_install_engine_installed_is_sorted(
        self, isolated_workspace, monkeypatch,
    ):
        """engine_installed must be a sorted list (stable + diffable)."""
        catalog = {
            "embedder": (
                {"fast_embed": _make_engine_impl(
                    dependencies=["fastembed>=0.3"])},
                "fast_embed",
            ),
            "reranker": (
                {"bge": _make_engine_impl(
                    dependencies=["FlagEmbedding>=1.2"])},
                "bge",
            ),
        }
        monkeypatch.setattr(install_mod, "load_slot_meta",
                            _fake_load_slot_meta(catalog))
        _stub_noop_subprocess(monkeypatch)
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              embedder: "fast_embed"
              reranker: "bge"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        install_mod.install(argparse.Namespace(dry_run=False, upgrade=False))
        state = json.loads(
            install_mod.INSTALL_STATE_PATH.read_text(encoding="utf-8")
        )
        installed = state["engine_installed"]
        assert installed == sorted(installed)

    def test_existing_installed_field_preserved_alongside_engine_installed(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """The existing 'installed' (plugin names) field must still be
        present after the engine_installed field is added."""
        _stub_noop_subprocess(monkeypatch)
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        install_mod.install(argparse.Namespace(dry_run=False, upgrade=False))
        state = json.loads(
            install_mod.INSTALL_STATE_PATH.read_text(encoding="utf-8")
        )
        assert "installed" in state      # plugins, unchanged
        assert "engine_installed" in state
        assert "deps_hash" in state

    def test_install_without_config_yaml_still_writes_state_with_empty_engine_installed(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """When no config.yaml exists, engine_deps is {}. install() must
        still succeed and write engine_installed: [] (or absent — not crash)."""
        _stub_noop_subprocess(monkeypatch)
        # No config.yaml written in isolated_workspace
        rc = install_mod.install(
            argparse.Namespace(dry_run=False, upgrade=False)
        )
        assert rc == 0
        state = json.loads(
            install_mod.INSTALL_STATE_PATH.read_text(encoding="utf-8")
        )
        # engine_installed may be [] or absent — but the state file must exist
        # and not contain unexpected non-list values for it
        if "engine_installed" in state:
            assert isinstance(state["engine_installed"], list)

    def test_install_engine_deps_included_in_pip_union(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """The pip call must include engine dependencies (e.g. MemoryOS)
        in the union passed to pip install."""
        captured: list[list[str]] = []
        monkeypatch.setattr(
            install_mod.subprocess, "call",
            lambda cmd, *a, **kw: (captured.append(list(cmd)) or 0),
        )
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        install_mod.install(argparse.Namespace(dry_run=False, upgrade=False))
        pip_calls = [c for c in captured if "pip" in c]
        assert len(pip_calls) == 1
        assert "MemoryOS>=2.0.17,<3.0" in pip_calls[0]

    def test_install_dispatches_engine_post_install(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """Engine post_install entries must be dispatched via the existing
        run_post_install_for_plugin helper after pip."""
        captured: list[list[str]] = []
        monkeypatch.setattr(
            install_mod.subprocess, "call",
            lambda cmd, *a, **kw: (captured.append(list(cmd)) or 0),
        )
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        rc = install_mod.install(
            argparse.Namespace(dry_run=False, upgrade=False)
        )
        assert rc == 0
        # memos has a post_install: [python -m memos setup]
        memos_calls = [
            c for c in captured
            if any("memos" in str(a) for a in c) and "pip" not in c
        ]
        assert len(memos_calls) >= 1

    def test_dry_run_does_not_write_engine_installed(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """--dry-run must not write install_state.json (same contract as
        for plugins)."""
        called = []
        monkeypatch.setattr(
            install_mod.subprocess, "call",
            lambda cmd, *a, **kw: (called.append(cmd) or 0),
        )
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        install_mod.install(argparse.Namespace(dry_run=True, upgrade=False))
        assert called == []
        assert not install_mod.INSTALL_STATE_PATH.exists()

    def test_pip_failure_prevents_engine_installed_state_write(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """If pip exits non-zero, install_state.json must NOT be written
        (including the new engine_installed field)."""
        monkeypatch.setattr(
            install_mod.subprocess, "call",
            lambda cmd, *a, **kw: 1,
        )
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        rc = install_mod.install(
            argparse.Namespace(dry_run=False, upgrade=False)
        )
        assert rc != 0
        assert not install_mod.INSTALL_STATE_PATH.exists()

    def test_engine_post_install_label_uses_namespaced_key(
        self, isolated_workspace, monkeypatch, capsys,
    ):
        """The label passed to run_post_install_for_plugin for an engine
        entry must be the full 'engine:<slot>:<name>' key (not just the
        slot or the short name alone)."""
        post = [{"args": ["echo", "setup"], "optional": False}]
        catalog = {
            "memory": (
                {"memos": _make_engine_impl(post_install=post)},
                "memos",
            ),
        }
        monkeypatch.setattr(install_mod, "load_slot_meta",
                            _fake_load_slot_meta(catalog))
        monkeypatch.setattr(
            install_mod.subprocess, "call",
            lambda cmd, *a, **kw: 0,
        )
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        install_mod.install(argparse.Namespace(dry_run=False, upgrade=False))
        out = capsys.readouterr().out
        # The output from run_post_install_for_plugin includes the label
        assert "engine:memory:memos" in out


# =====================================================================
# 9. State transitions — install → has_pending_deps idempotency
# =====================================================================


class TestInstallStateTransitions:
    """Non-idempotent operations: install, then check, then change."""

    def test_install_then_has_pending_false(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """After successful install, has_pending_deps() returns False
        (even with engine deps in play)."""
        _stub_noop_subprocess(monkeypatch)
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        assert install_mod.has_pending_deps()[0] is True
        install_mod.install(argparse.Namespace(dry_run=False, upgrade=False))
        assert install_mod.has_pending_deps()[0] is False

    def test_second_install_still_pending_false(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """Two back-to-back install() calls: both succeed and
        has_pending_deps() is False after each."""
        _stub_noop_subprocess(monkeypatch)
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        install_mod.install(argparse.Namespace(dry_run=False, upgrade=False))
        install_mod.install(argparse.Namespace(dry_run=False, upgrade=False))
        assert install_mod.has_pending_deps()[0] is False

    def test_engine_installed_list_updated_on_second_install_after_slot_change(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """If the user switches engines between installs, the new
        engine_installed list in state reflects the post-switch selection."""
        _stub_noop_subprocess(monkeypatch)
        # First install with default (no dep engine)
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: ""
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        install_mod.install(argparse.Namespace(dry_run=False, upgrade=False))
        state_before = json.loads(
            install_mod.INSTALL_STATE_PATH.read_text(encoding="utf-8")
        )

        # Switch to memos and re-install
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        install_mod.install(argparse.Namespace(dry_run=False, upgrade=False))
        state_after = json.loads(
            install_mod.INSTALL_STATE_PATH.read_text(encoding="utf-8")
        )
        # After switch, memos key should appear
        if "engine_installed" in state_after:
            assert "engine:memory:memos" in state_after["engine_installed"]


# =====================================================================
# 10. DefaultInstallService.deps_status() — engines key
# =====================================================================


class TestDepsStatusEngines:
    """deps_status() must return an 'engines' key alongside 'plugins'."""

    def test_deps_status_contains_engines_key(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """The returned dict must have an 'engines' key."""
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        svc = install_mod.DefaultInstallService()
        status = svc.deps_status()
        assert "engines" in status

    def test_deps_status_contains_plugins_key(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """The existing 'plugins' key must still be present."""
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        svc = install_mod.DefaultInstallService()
        status = svc.deps_status()
        assert "plugins" in status

    def test_engine_entry_shape_has_required_keys(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """Each entry in status['engines'] must have dependencies,
        post_install, installed, and satisfied keys."""
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        svc = install_mod.DefaultInstallService()
        status = svc.deps_status()
        engines = status.get("engines", {})
        for key, entry in engines.items():
            assert "dependencies" in entry, f"{key} missing 'dependencies'"
            assert "post_install" in entry, f"{key} missing 'post_install'"
            assert "installed" in entry,    f"{key} missing 'installed'"
            assert "satisfied" in entry,    f"{key} missing 'satisfied'"

    def test_engine_installed_false_before_install(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """Before any install, all engine entries have installed=False."""
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        svc = install_mod.DefaultInstallService()
        status = svc.deps_status()
        engines = status.get("engines", {})
        for key, entry in engines.items():
            assert entry["installed"] is False, \
                f"{key} should not be installed before any install"

    def test_engine_installed_true_after_install(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """After install, engine entries that were collected appear as
        installed=True."""
        _stub_noop_subprocess(monkeypatch)
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        install_mod.install(argparse.Namespace(dry_run=False, upgrade=False))
        svc = install_mod.DefaultInstallService()
        status = svc.deps_status()
        engines = status.get("engines", {})
        assert "engine:memory:memos" in engines
        assert engines["engine:memory:memos"]["installed"] is True

    def test_engine_satisfied_true_after_install_hash_matches(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """After install, memos entry is satisfied=True (installed AND
        recorded_hash == live_hash)."""
        _stub_noop_subprocess(monkeypatch)
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        install_mod.install(argparse.Namespace(dry_run=False, upgrade=False))
        svc = install_mod.DefaultInstallService()
        status = svc.deps_status()
        engines = status.get("engines", {})
        if "engine:memory:memos" in engines:
            assert engines["engine:memory:memos"]["satisfied"] is True

    def test_engine_satisfied_false_after_slot_change(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """If the user swaps engine selection after install (without
        re-installing), satisfied=False for all entries (hash mismatch)."""
        _stub_noop_subprocess(monkeypatch)
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: ""
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        install_mod.install(argparse.Namespace(dry_run=False, upgrade=False))

        # Change config without re-installing
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        svc = install_mod.DefaultInstallService()
        status = svc.deps_status()
        # pending should be True due to hash mismatch
        assert status["pending"] is True

    def test_pending_true_when_any_engine_unsatisfied(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """The top-level 'pending' flag is True when ANY engine is
        unsatisfied (broadened semantic from plugin-only)."""
        # No install run — so engine entries will be unsatisfied
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        svc = install_mod.DefaultInstallService()
        status = svc.deps_status()
        assert status["pending"] is True

    def test_deps_status_engines_key_present_even_without_config_yaml(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """Even when there is no config.yaml (engine_deps={}), the
        'engines' key must exist in the returned dict (may be empty {})."""
        # No config.yaml written
        svc = install_mod.DefaultInstallService()
        status = svc.deps_status()
        assert "engines" in status
        assert isinstance(status["engines"], dict)

    def test_engine_entry_dependencies_list_matches_catalog(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """The 'dependencies' list in an engine entry matches what
        collect_engine_dependencies returns for that slot."""
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        svc = install_mod.DefaultInstallService()
        status = svc.deps_status()
        engines = status.get("engines", {})
        if "engine:memory:memos" in engines:
            assert "MemoryOS>=2.0.17,<3.0" in \
                engines["engine:memory:memos"]["dependencies"]

    def test_engine_entry_post_install_list_matches_catalog(
        self, isolated_workspace, simple_catalog, monkeypatch,
    ):
        """The 'post_install' list in an engine entry mirrors what
        collect_engine_post_install returns."""
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "memos"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        svc = install_mod.DefaultInstallService()
        status = svc.deps_status()
        engines = status.get("engines", {})
        if "engine:memory:memos" in engines:
            post = engines["engine:memory:memos"]["post_install"]
            assert isinstance(post, list)
            assert any("memos" in str(e) for e in post)

    def test_installed_set_uses_engine_installed_from_state(
        self, isolated_workspace, monkeypatch,
    ):
        """deps_status() consults state['engine_installed'] for the
        installed flag — a hand-crafted state file should reflect correctly."""
        catalog = {
            "memory": (
                {"manual_engine": _make_engine_impl(dependencies=["pkg>=1"])},
                "manual_engine",
            ),
        }
        monkeypatch.setattr(install_mod, "load_slot_meta",
                            _fake_load_slot_meta(catalog))
        _write_config_yaml(Path.cwd(), """\
            core_implementations:
              memory: "manual_engine"
            idle:
              min_interval: 1
              max_interval: 60
              default_interval: 1
        """)
        # Manually craft a state that marks this engine as installed
        plugin_deps = install_mod.collect_plugin_dependencies()
        plugin_post = install_mod.collect_plugin_post_install()
        engine_deps = {"engine:memory:manual_engine": ["pkg>=1"]}
        engine_post = {}
        h = install_mod.deps_hash(plugin_deps, plugin_post,
                                   engine_deps=engine_deps,
                                   engine_post=engine_post)
        install_mod.write_install_state({
            "deps_hash": h,
            "installed": [],
            "engine_installed": ["engine:memory:manual_engine"],
            "installed_at": "2026-01-01T00:00:00",
        })
        svc = install_mod.DefaultInstallService()
        status = svc.deps_status()
        engines = status.get("engines", {})
        assert "engine:memory:manual_engine" in engines
        assert engines["engine:memory:manual_engine"]["installed"] is True
        assert engines["engine:memory:manual_engine"]["satisfied"] is True
