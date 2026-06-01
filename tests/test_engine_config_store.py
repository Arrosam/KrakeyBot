"""Edge tests for the engine-system "engine config autonomy" change.

Contract source: ``contracts/config-model/definition.md`` §"Engine
configuration autonomy".

Three surfaces under test:
  1. ``FileEngineConfigStore`` (krakey/engine_system/config_store.py) — new
     file, modelled on ``FilePluginConfigStore``.
  2. ``meta_loader.load_slot_meta`` config_path field parsing.
  3. ``EngineImpl.config_path`` dataclass field.
  4. ``EngineRegistry._engine_config`` / ``resolve`` wiring.

All tests target the FINAL state of the feature — they are RED until
the implementation lands.

Testing techniques applied (matching the 4-technique mandate):
  - Positive / equivalence-partition
  - BVA / boundary values
  - State transitions
  - Negative / error-guessing

Assumptions are documented in-line with ``# ASSUMPTION:`` tags and
summarised at the bottom of this module.
"""
from __future__ import annotations

import textwrap
import types
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Guard imports — the module under test does not yet exist; any
# ImportError causes the entire module-level import to fail, which is
# fine because pytest will mark these tests as ERRORs (collection
# failure), not silent passes.  We choose a lazy import inside each
# test class so that *partial* implementations (e.g. only
# FileEngineConfigStore exists but not EngineRegistry.workspace_root)
# still allow unrelated tests to run and produce useful RED signals.
# ---------------------------------------------------------------------------


def _import_store():
    """Lazy import for the new module — fails loudly in red state."""
    from krakey.engine_system.config_store import FileEngineConfigStore  # type: ignore[import]
    return FileEngineConfigStore


def _import_load_slot_meta():
    from krakey.engine_system.meta_loader import load_slot_meta
    return load_slot_meta


def _import_engine_impl():
    from krakey.engine_system.catalog import EngineImpl
    return EngineImpl


def _import_registry():
    from krakey.engine_system.registry import EngineRegistry
    return EngineRegistry


# ---------------------------------------------------------------------------
# Helpers shared across sections
# ---------------------------------------------------------------------------


def _write_meta(tmp_path: Path, slot: str, content: str) -> Path:
    """Write ``content`` to ``<tmp_path>/<slot>/meta.yaml``."""
    slot_dir = tmp_path / slot
    slot_dir.mkdir(parents=True, exist_ok=True)
    meta = slot_dir / "meta.yaml"
    meta.write_text(textwrap.dedent(content), encoding="utf-8")
    return meta


def _load_meta(tmp_path: Path, slot: str):
    """Thin wrapper: ``load_slot_meta(slot, engines_root=tmp_path)``."""
    load_slot_meta = _import_load_slot_meta()
    return load_slot_meta(slot, engines_root=tmp_path)


# ---------------------------------------------------------------------------
# ============================================================
# SECTION 1 — FileEngineConfigStore
# ============================================================
# ---------------------------------------------------------------------------


class TestFileEngineConfigStorePositive:
    """Valid inputs from each equivalence class return correct results."""

    def test_read_existing_yaml_returns_dict(self, tmp_path):
        """read() of an existing YAML file returns its contents as a dict."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        # Write the settings file at a workspace-relative path
        settings_dir = tmp_path / "data" / "memory"
        settings_dir.mkdir(parents=True)
        (settings_dir / "settings.yaml").write_text(
            "cache_size_mb: 128\nmax_nodes: 5000\n", encoding="utf-8"
        )
        result = store.read("data/memory/settings.yaml")
        assert result == {"cache_size_mb": 128, "max_nodes": 5000}

    def test_write_creates_file_and_returns_path(self, tmp_path):
        """write() creates the file (with parent dirs) and returns the path."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        returned = store.write(
            "data/recall/settings.yaml",
            {"top_k": 10, "threshold": 0.75},
        )
        assert isinstance(returned, Path)
        assert returned.exists()
        assert returned == tmp_path / "data" / "recall" / "settings.yaml"

    def test_write_then_read_roundtrip(self, tmp_path):
        """write() then read() returns the same dict (happy-path round-trip)."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        original = {"temperature": 0.7, "max_tokens": 512}
        store.write("engines/decision/settings.yaml", original)
        result = store.read("engines/decision/settings.yaml")
        assert result == original

    def test_read_returns_dict_type(self, tmp_path):
        """Return type of read() is always dict."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        (tmp_path / "s.yaml").write_text("key: val\n", encoding="utf-8")
        result = store.read("s.yaml")
        assert isinstance(result, dict)

    def test_workspace_root_accepts_str(self, tmp_path):
        """Constructor accepts workspace_root as a plain str."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(str(tmp_path))
        # Should be able to write without error
        returned = store.write("cfg.yaml", {"x": 1})
        assert returned.exists()

    def test_workspace_root_accepts_path(self, tmp_path):
        """Constructor accepts workspace_root as a Path object."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        returned = store.write("cfg.yaml", {"x": 1})
        assert returned.exists()

    def test_write_creates_nested_parent_dirs(self, tmp_path):
        """write() creates all intermediate parent directories."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        deep_path = "a/b/c/d/settings.yaml"
        store.write(deep_path, {"deep": True})
        assert (tmp_path / deep_path).exists()

    def test_write_returns_absolute_path(self, tmp_path):
        """write() returns the absolute path to the written file."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        result = store.write("sub/cfg.yaml", {})
        assert result.is_absolute()


class TestFileEngineConfigStoreBVA:
    """Boundary value analysis — empty strings, missing files, edge dicts."""

    def test_read_nonexistent_file_returns_empty_dict(self, tmp_path):
        """read() with a config_path pointing at a non-existent file → {}."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        result = store.read("does/not/exist.yaml")
        assert result == {}

    def test_read_empty_config_path_returns_empty_dict(self, tmp_path):
        """read() with empty string config_path="" → {} (no crash)."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        result = store.read("")
        assert result == {}

    def test_read_falsy_yaml_bare_list_returns_empty_dict(self, tmp_path):
        """read() when the YAML file contains a bare list (not a dict) → {}."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        (tmp_path / "bad.yaml").write_text("- item1\n- item2\n", encoding="utf-8")
        result = store.read("bad.yaml")
        assert result == {}

    def test_read_yaml_scalar_string_returns_empty_dict(self, tmp_path):
        """read() when the YAML file is a bare scalar string → {}."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        (tmp_path / "scalar.yaml").write_text("just a string\n", encoding="utf-8")
        result = store.read("scalar.yaml")
        assert result == {}

    def test_read_yaml_scalar_integer_returns_empty_dict(self, tmp_path):
        """read() when the YAML file is a bare integer → {}."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        (tmp_path / "int.yaml").write_text("42\n", encoding="utf-8")
        result = store.read("int.yaml")
        assert result == {}

    def test_read_yaml_null_returns_empty_dict(self, tmp_path):
        """read() when the YAML file contains only null → {}."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        (tmp_path / "null.yaml").write_text("~\n", encoding="utf-8")
        result = store.read("null.yaml")
        assert result == {}

    def test_read_empty_yaml_file_returns_empty_dict(self, tmp_path):
        """read() of a zero-byte YAML file → {}."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
        result = store.read("empty.yaml")
        assert result == {}

    def test_read_empty_config_path_no_file_io(self, tmp_path):
        """read("") must return {} without touching the filesystem at all
        (no exception, no side effects)."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        # No files exist under tmp_path; the call must still succeed
        result = store.read("")
        assert result == {}
        assert result is not None

    def test_write_empty_dict_creates_valid_yaml(self, tmp_path):
        """write() with an empty dict creates a file and read() returns {}."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        store.write("cfg.yaml", {})
        result = store.read("cfg.yaml")
        assert result == {}

    def test_write_with_empty_config_path_raises_value_error(self, tmp_path):
        """write() with empty config_path raises ValueError."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        with pytest.raises(ValueError):
            store.write("", {"key": "value"})

    def test_read_returns_empty_dict_not_none_for_missing_file(self, tmp_path):
        """Return value for a missing file is specifically {}, not None."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        result = store.read("nonexistent.yaml")
        assert result is not None
        assert result == {}

    def test_write_single_key_value_roundtrips(self, tmp_path):
        """BVA: exactly one key-value pair round-trips correctly."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        store.write("settings.yaml", {"only_key": "only_val"})
        assert store.read("settings.yaml") == {"only_key": "only_val"}


class TestFileEngineConfigStoreStateTransition:
    """State transitions: write → read → overwrite → read again."""

    def test_write_then_read_then_overwrite_reflects_new_values(self, tmp_path):
        """write → read → write (new content) → read returns new content."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        store.write("settings.yaml", {"v": 1})
        assert store.read("settings.yaml") == {"v": 1}
        # Overwrite with different content
        store.write("settings.yaml", {"v": 2, "extra": "added"})
        result = store.read("settings.yaml")
        assert result == {"v": 2, "extra": "added"}

    def test_no_caching_fresh_read_reflects_external_write(self, tmp_path):
        """read() reads from disk each time — no in-memory cache.

        Writing the file externally between two read() calls must be
        reflected in the second call.
        """
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        target = tmp_path / "settings.yaml"
        target.write_text("x: 10\n", encoding="utf-8")
        assert store.read("settings.yaml") == {"x": 10}
        # Mutate the file externally
        target.write_text("x: 99\n", encoding="utf-8")
        assert store.read("settings.yaml") == {"x": 99}

    def test_delete_file_between_reads_returns_empty_dict(self, tmp_path):
        """After write(), deleting the file externally → next read() → {}."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        store.write("settings.yaml", {"k": "v"})
        assert store.read("settings.yaml") == {"k": "v"}
        (tmp_path / "settings.yaml").unlink()
        assert store.read("settings.yaml") == {}

    def test_multiple_writes_to_different_paths_are_independent(self, tmp_path):
        """Writing to path A does not affect reading from path B."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        store.write("a.yaml", {"slot": "a"})
        store.write("b.yaml", {"slot": "b"})
        assert store.read("a.yaml") == {"slot": "a"}
        assert store.read("b.yaml") == {"slot": "b"}

    def test_two_store_instances_same_root_see_shared_disk_state(self, tmp_path):
        """Two FileEngineConfigStore instances on the same root share the same
        filesystem — write via one, read via the other."""
        FileEngineConfigStore = _import_store()
        writer = FileEngineConfigStore(tmp_path)
        reader = FileEngineConfigStore(tmp_path)
        writer.write("shared.yaml", {"shared": True})
        assert reader.read("shared.yaml") == {"shared": True}


class TestFileEngineConfigStoreNegative:
    """Negative / error-guessing tests."""

    def test_write_empty_config_path_raises(self, tmp_path):
        """write("", ...) must raise ValueError."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        with pytest.raises(ValueError):
            store.write("", {"a": 1})

    def test_read_does_not_raise_for_any_absent_path(self, tmp_path):
        """read() must NEVER raise for missing files, only return {}."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        # Deep nonexistent path
        result = store.read("deeply/nested/path/that/doesnt/exist.yaml")
        assert result == {}

    def test_read_yaml_list_not_treated_as_dict(self, tmp_path):
        """A YAML file whose root is a list is not a dict — must return {}."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        (tmp_path / "list.yaml").write_text(
            "- one\n- two\n- three\n", encoding="utf-8"
        )
        result = store.read("list.yaml")
        assert result == {}
        assert not isinstance(result, list)

    def test_read_does_not_raise_for_empty_config_path_even_with_no_files(
        self, tmp_path
    ):
        """read("") must be safe even if the workspace_root is empty."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        assert store.read("") == {}

    def test_write_raises_value_error_not_other_exception_for_empty_path(
        self, tmp_path
    ):
        """The exception raised by write("", ...) is specifically ValueError."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        with pytest.raises(ValueError):
            store.write("", {})


class TestFileEngineConfigStoreRoundTrip:
    """Round-trip tests including nested structures and unicode."""

    def test_roundtrip_nested_dict(self, tmp_path):
        """Nested dict structure survives a write/read round-trip."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        data: dict[str, Any] = {"a": {"b": 1}, "c": {"d": {"e": 2}}}
        store.write("nested.yaml", data)
        assert store.read("nested.yaml") == data

    def test_roundtrip_unicode_values(self, tmp_path):
        """Unicode values survive write/read (YAML allow_unicode=True)."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        data = {"a": {"b": 1}, "u": "中文"}
        store.write("unicode.yaml", data)
        result = store.read("unicode.yaml")
        assert result == data
        assert result["u"] == "中文"

    def test_roundtrip_mixed_types(self, tmp_path):
        """Mixed value types (int, float, bool, list) survive round-trip."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        data = {
            "count": 42,
            "ratio": 0.5,
            "enabled": True,
            "tags": ["a", "b"],
        }
        store.write("mixed.yaml", data)
        result = store.read("mixed.yaml")
        assert result == data

    def test_roundtrip_deeply_nested_unicode(self, tmp_path):
        """THE PIVOTAL ROUND-TRIP TEST: nested dict + unicode values."""
        FileEngineConfigStore = _import_store()
        store = FileEngineConfigStore(tmp_path)
        data = {"a": {"b": 1}, "u": "中文"}
        store.write("deep_unicode.yaml", data)
        assert store.read("deep_unicode.yaml") == {"a": {"b": 1}, "u": "中文"}


# ---------------------------------------------------------------------------
# ============================================================
# SECTION 2 — meta_loader config_path parsing
# ============================================================
# ---------------------------------------------------------------------------


class TestMetaLoaderConfigPathPositive:
    """Positive tests: config_path field parsed correctly from meta.yaml."""

    def test_entry_with_config_path_sets_engine_impl_config_path(self, tmp_path):
        """A builtin_engines entry WITH config_path → EngineImpl.config_path
        equals the declared value."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: myengine
                factory_module: some.pkg
                factory_attr: MyEngine
                default: true
                config_path: data/foo/settings.yaml
        """)
        catalog, _ = _load_meta(tmp_path, "demo")
        assert catalog["myengine"].config_path == "data/foo/settings.yaml"

    def test_entry_without_config_path_defaults_to_empty_string(self, tmp_path):
        """An entry WITHOUT config_path → EngineImpl.config_path == ""."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: nopath
                factory_module: some.pkg
                factory_attr: NoCfg
                default: true
        """)
        catalog, _ = _load_meta(tmp_path, "demo")
        assert catalog["nopath"].config_path == ""

    def test_two_entries_each_gets_own_config_path(self, tmp_path):
        """Multi-entry slot: each engine receives ONLY its own config_path."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: alpha
                factory_module: pkg.a
                factory_attr: Alpha
                default: true
                config_path: data/alpha/settings.yaml
              - name: beta
                factory_module: pkg.b
                factory_attr: Beta
                config_path: data/beta/settings.yaml
        """)
        catalog, _ = _load_meta(tmp_path, "demo")
        assert catalog["alpha"].config_path == "data/alpha/settings.yaml"
        assert catalog["beta"].config_path == "data/beta/settings.yaml"

    def test_entry_with_config_path_coerced_to_str(self, tmp_path):
        """config_path parsed from meta.yaml is always a str (str coercion)."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                config_path: data/memory/settings.yaml
        """)
        catalog, _ = _load_meta(tmp_path, "demo")
        assert isinstance(catalog["eng"].config_path, str)

    def test_entry_has_config_path_attribute(self, tmp_path):
        """EngineImpl returned by load_slot_meta must expose config_path."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
        """)
        catalog, _ = _load_meta(tmp_path, "demo")
        assert hasattr(catalog["eng"], "config_path")


class TestMetaLoaderConfigPathBVA:
    """Boundary values for config_path parsing."""

    def test_entry_explicit_null_config_path_coerced_to_empty_string(self, tmp_path):
        """``config_path:`` (YAML null) → coerced to ""."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                config_path:
        """)  # bare key → null
        catalog, _ = _load_meta(tmp_path, "demo")
        assert catalog["eng"].config_path == ""

    def test_entry_config_path_single_filename(self, tmp_path):
        """config_path with just a filename (no subdirectory) is valid."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                config_path: settings.yaml
        """)
        catalog, _ = _load_meta(tmp_path, "demo")
        assert catalog["eng"].config_path == "settings.yaml"

    def test_entry_config_path_deeply_nested(self, tmp_path):
        """config_path with many path segments is preserved verbatim."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                config_path: data/memory/graph/settings.yaml
        """)
        catalog, _ = _load_meta(tmp_path, "demo")
        assert catalog["eng"].config_path == "data/memory/graph/settings.yaml"

    def test_two_loads_same_slot_config_path_independent(self, tmp_path):
        """Two calls to load_slot_meta on the same slot return independent
        EngineImpl objects (no shared state on config_path)."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                config_path: data/settings.yaml
        """)
        catalog1, _ = _load_meta(tmp_path, "demo")
        catalog2, _ = _load_meta(tmp_path, "demo")
        assert catalog1["eng"] is not catalog2["eng"]

    def test_entry_without_config_path_does_not_contaminate_entry_with(
        self, tmp_path
    ):
        """Entry A (no config_path) before entry B (has config_path) —
        B still gets its own config_path, A still gets ""."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: no_path
                factory_module: pkg
                factory_attr: A
                default: true
              - name: has_path
                factory_module: pkg
                factory_attr: B
                config_path: data/has/settings.yaml
        """)
        catalog, _ = _load_meta(tmp_path, "demo")
        assert catalog["no_path"].config_path == ""
        assert catalog["has_path"].config_path == "data/has/settings.yaml"

    def test_entry_with_config_path_does_not_contaminate_entry_without(
        self, tmp_path
    ):
        """Entry A (has config_path) before entry B (no config_path) —
        B must still get ""."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: has_path
                factory_module: pkg
                factory_attr: A
                default: true
                config_path: data/a/settings.yaml
              - name: no_path
                factory_module: pkg
                factory_attr: B
        """)
        catalog, _ = _load_meta(tmp_path, "demo")
        assert catalog["has_path"].config_path == "data/a/settings.yaml"
        assert catalog["no_path"].config_path == ""


class TestMetaLoaderConfigPathNegative:
    """Negative / error-guessing for config_path field."""

    def test_entry_config_path_bad_type_integer_coerced_to_str(self, tmp_path):
        """config_path: 42 (integer) → coerced to str, not ValueError.

        ASSUMPTION: the loader coerces non-string config_path values
        to str (consistent with how other scalar fields are handled).
        """
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                config_path: 42
        """)
        # Must not raise — coercion handles it
        catalog, _ = _load_meta(tmp_path, "demo")
        assert isinstance(catalog["eng"].config_path, str)

    def test_config_path_missing_does_not_raise_meta_parse_error(self, tmp_path):
        """Absence of config_path must never raise MetaParseError."""
        from krakey.engine_system.meta_loader import MetaParseError
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
        """)
        try:
            catalog, _ = _load_meta(tmp_path, "demo")
        except MetaParseError:
            pytest.fail("Missing config_path must not raise MetaParseError")


# ---------------------------------------------------------------------------
# ============================================================
# SECTION 3 — EngineImpl.config_path dataclass field
# ============================================================
# ---------------------------------------------------------------------------


class TestEngineImplConfigPathDataclass:
    """Unit tests for the EngineImpl.config_path field at the dataclass level."""

    def test_engine_impl_config_path_defaults_to_empty_string(self):
        """EngineImpl constructed without config_path kwarg → config_path == ""."""
        EngineImpl = _import_engine_impl()
        impl = EngineImpl(cls=None, description="test")
        assert impl.config_path == ""

    def test_engine_impl_config_path_is_str_type(self):
        """EngineImpl.config_path field holds a str."""
        EngineImpl = _import_engine_impl()
        impl = EngineImpl(cls=None, description="test")
        assert isinstance(impl.config_path, str)

    def test_engine_impl_config_path_kwarg_accepted(self):
        """EngineImpl accepts config_path as a constructor kwarg."""
        EngineImpl = _import_engine_impl()
        impl = EngineImpl(cls=None, description="test", config_path="data/foo/settings.yaml")
        assert impl.config_path == "data/foo/settings.yaml"

    def test_engine_impl_config_path_empty_string_kwarg(self):
        """EngineImpl(config_path="") results in config_path == ""."""
        EngineImpl = _import_engine_impl()
        impl = EngineImpl(cls=None, description="test", config_path="")
        assert impl.config_path == ""

    def test_engine_impl_existing_fields_unaffected_by_config_path(self):
        """Adding config_path does not break existing EngineImpl fields."""
        EngineImpl = _import_engine_impl()
        impl = EngineImpl(
            cls=None,
            description="test",
            config_schema=[{"field": "x"}],
            dependencies=["dep-a"],
            post_install=[{"args": ["cmd"], "description": "", "optional": False}],
            config_path="data/settings.yaml",
        )
        assert impl.config_schema == [{"field": "x"}]
        assert impl.dependencies == ["dep-a"]
        assert impl.post_install[0]["args"] == ["cmd"]
        assert impl.config_path == "data/settings.yaml"

    def test_engine_impl_config_path_default_does_not_share_across_instances(self):
        """Two EngineImpl instances with default config_path must not share
        state (relevant if ever mutable; currently str so immutable, but
        guard against a mutable default regression)."""
        EngineImpl = _import_engine_impl()
        a = EngineImpl(cls=None, description="a")
        b = EngineImpl(cls=None, description="b")
        # Both start as "" and must be independent
        assert a.config_path == ""
        assert b.config_path == ""

    def test_engine_impl_has_config_path_attribute(self):
        """EngineImpl instances expose a config_path attribute."""
        EngineImpl = _import_engine_impl()
        impl = EngineImpl(cls=None, description="x")
        assert hasattr(impl, "config_path")


# ---------------------------------------------------------------------------
# ============================================================
# SECTION 4 — EngineRegistry wiring
# ============================================================
# ---------------------------------------------------------------------------


def _make_registry_cfg(slot: str = "memory", short_name: str = "custom") -> Any:
    """Build the minimal Config object the registry tests need.

    Uses the same pattern as test_engine_registry.py: real Config +
    CoreImplementations.
    """
    from krakey.models.config import Config
    from krakey.models.config.core_impls import CoreImplementations
    return Config(core_implementations=CoreImplementations(**{slot: short_name}))


class TestEngineRegistryEngineConfigPositive:
    """Positive tests: _engine_config reads from FileEngineConfigStore."""

    def test_engine_config_returns_file_dict_when_file_exists(
        self, tmp_path, monkeypatch
    ):
        """_engine_config(slot, short_name) returns the dict from the
        engine's settings file when the file exists at config_path under
        workspace_root.

        Approach: monkeypatch _load_slot_catalog to inject an EngineImpl
        with a known config_path, write the settings file under tmp_path,
        construct EngineRegistry(cfg, workspace_root=tmp_path), and assert
        _engine_config returns the file's contents.

        ASSUMPTION: _engine_config calls _load_slot_catalog(slot) to look
        up the EngineImpl for the given (slot, short_name), reads
        impl.config_path, and calls FileEngineConfigStore(workspace_root).
        read(config_path). The short_name must match the catalog key.
        """
        EngineImpl = _import_engine_impl()
        EngineRegistry = _import_registry()
        FileEngineConfigStore = _import_store()
        import krakey.engine_system.registry as reg_mod

        captured_config_path = "data/test/settings.yaml"

        class _CfgAwareImpl:
            def __init__(self, *, config=None):
                pass
            def hello(self):
                return "ok"

        monkeypatch.setattr(
            reg_mod, "_load_slot_catalog",
            lambda slot: (
                {"custom": EngineImpl(
                    cls=_CfgAwareImpl,
                    description="test impl",
                    config_path=captured_config_path,
                )},
                "custom",
            ),
        )

        # Write the settings file at the expected path under tmp_path
        settings_file = tmp_path / "data" / "test" / "settings.yaml"
        settings_file.parent.mkdir(parents=True)
        settings_file.write_text("cache_size_mb: 200\n", encoding="utf-8")

        cfg = _make_registry_cfg(slot="memory", short_name="custom")
        registry = EngineRegistry(cfg, workspace_root=tmp_path)
        result = registry._engine_config("memory", "custom")
        assert result == {"cache_size_mb": 200}

    def test_engine_config_returns_empty_dict_when_no_file(
        self, tmp_path, monkeypatch
    ):
        """_engine_config returns {} when the impl's config_path file is absent."""
        EngineImpl = _import_engine_impl()
        EngineRegistry = _import_registry()
        import krakey.engine_system.registry as reg_mod

        monkeypatch.setattr(
            reg_mod, "_load_slot_catalog",
            lambda slot: (
                {"custom": EngineImpl(
                    cls=None,
                    description="test",
                    config_path="data/nonexistent/settings.yaml",
                )},
                "custom",
            ),
        )

        cfg = _make_registry_cfg(slot="memory", short_name="custom")
        registry = EngineRegistry(cfg, workspace_root=tmp_path)
        result = registry._engine_config("memory", "custom")
        assert result == {}

    def test_engine_config_returns_empty_dict_when_config_path_is_empty(
        self, tmp_path, monkeypatch
    ):
        """_engine_config returns {} when impl.config_path == ""."""
        EngineImpl = _import_engine_impl()
        EngineRegistry = _import_registry()
        import krakey.engine_system.registry as reg_mod

        monkeypatch.setattr(
            reg_mod, "_load_slot_catalog",
            lambda slot: (
                {"custom": EngineImpl(
                    cls=None,
                    description="test",
                    config_path="",  # empty path → no file
                )},
                "custom",
            ),
        )

        cfg = _make_registry_cfg(slot="memory", short_name="custom")
        registry = EngineRegistry(cfg, workspace_root=tmp_path)
        result = registry._engine_config("memory", "custom")
        assert result == {}

    def test_registry_workspace_root_param_accepted(self, tmp_path):
        """EngineRegistry constructor accepts workspace_root as a keyword param."""
        EngineRegistry = _import_registry()
        cfg = _make_registry_cfg(slot="memory", short_name="")
        # Should not raise
        reg = EngineRegistry(cfg, workspace_root=tmp_path)
        assert reg is not None

    def test_registry_workspace_root_default_does_not_crash(self):
        """EngineRegistry() without workspace_root uses a default (e.g.
        'workspace') and does not crash on construction."""
        EngineRegistry = _import_registry()
        cfg = _make_registry_cfg(slot="memory", short_name="")
        # Should not raise at construction time
        reg = EngineRegistry(cfg)
        assert reg is not None


class TestEngineRegistryEngineConfigNegative:
    """Negative / regression tests: _engine_config must NOT touch engine_configs."""

    def test_engine_config_does_not_access_engine_configs_on_cfg(
        self, tmp_path, monkeypatch
    ):
        """REGRESSION GUARD: _engine_config must NOT reference cfg.engine_configs.

        Build a Config-like object (SimpleNamespace) with NO engine_configs
        attribute. Assert that _engine_config does not raise AttributeError.
        This is the hard-switch contract: after this change, _engine_config
        reads from the file store, not from cfg.engine_configs.

        ASSUMPTION: Once the feature is implemented, _engine_config no
        longer reads from cfg.engine_configs. The current implementation
        still has the guard (``hasattr(self._cfg, 'engine_configs')``), but
        the new implementation must not reference it at all. This test is
        RED until the implementation is updated to drop that reference.
        """
        EngineImpl = _import_engine_impl()
        EngineRegistry = _import_registry()
        import krakey.engine_system.registry as reg_mod

        monkeypatch.setattr(
            reg_mod, "_load_slot_catalog",
            lambda slot: (
                {"custom": EngineImpl(
                    cls=None,
                    description="test",
                    config_path="",
                )},
                "custom",
            ),
        )

        # Construct a cfg-like object with NO engine_configs attribute
        cfg_no_engine_configs = types.SimpleNamespace(
            core_implementations=types.SimpleNamespace(
                get=lambda slot, default=None: "custom"
            )
            # NOTE: no engine_configs attribute at all
        )
        registry = EngineRegistry(cfg_no_engine_configs, workspace_root=tmp_path)  # type: ignore[arg-type]
        # Must not raise AttributeError on engine_configs
        result = registry._engine_config("memory", "custom")
        assert isinstance(result, dict)

    def test_engine_config_with_missing_engine_configs_attr_returns_file_contents(
        self, tmp_path, monkeypatch
    ):
        """When cfg has NO engine_configs, _engine_config still returns the
        file dict if the settings file exists (proving it reads from disk,
        not from cfg)."""
        EngineImpl = _import_engine_impl()
        EngineRegistry = _import_registry()
        import krakey.engine_system.registry as reg_mod

        monkeypatch.setattr(
            reg_mod, "_load_slot_catalog",
            lambda slot: (
                {"custom": EngineImpl(
                    cls=None,
                    description="test",
                    config_path="data/settings.yaml",
                )},
                "custom",
            ),
        )

        settings_file = tmp_path / "data" / "settings.yaml"
        settings_file.parent.mkdir(parents=True)
        settings_file.write_text("from_file: true\n", encoding="utf-8")

        cfg_no_engine_configs = types.SimpleNamespace(
            core_implementations=types.SimpleNamespace(
                get=lambda slot, default=None: "custom"
            )
        )
        registry = EngineRegistry(cfg_no_engine_configs, workspace_root=tmp_path)  # type: ignore[arg-type]
        result = registry._engine_config("memory", "custom")
        assert result == {"from_file": True}


class TestEngineRegistryEngineConfigStateTransition:
    """State transitions for _engine_config — no caching staleness."""

    def test_engine_config_reflects_new_file_without_caching(
        self, tmp_path, monkeypatch
    ):
        """Writing a new settings file then calling _engine_config again
        reflects the new values (no caching staleness).

        If the registry caches, this test will fail — which is the
        correct RED signal (the spec says reads should always be fresh
        unless the contract explicitly says otherwise).
        """
        EngineImpl = _import_engine_impl()
        EngineRegistry = _import_registry()
        import krakey.engine_system.registry as reg_mod

        settings_path = "data/test/settings.yaml"

        monkeypatch.setattr(
            reg_mod, "_load_slot_catalog",
            lambda slot: (
                {"custom": EngineImpl(
                    cls=None,
                    description="test",
                    config_path=settings_path,
                )},
                "custom",
            ),
        )

        cfg = _make_registry_cfg(slot="memory", short_name="custom")
        registry = EngineRegistry(cfg, workspace_root=tmp_path)

        # First call: no file → {}
        assert registry._engine_config("memory", "custom") == {}

        # Write the file
        settings_file = tmp_path / "data" / "test" / "settings.yaml"
        settings_file.parent.mkdir(parents=True)
        settings_file.write_text("top_k: 15\n", encoding="utf-8")

        # Second call must see the new file
        result = registry._engine_config("memory", "custom")
        assert result == {"top_k": 15}, (
            "_engine_config is caching (returned {} instead of file contents "
            "after file was written). The contract requires fresh reads."
        )

    def test_engine_config_update_then_read_reflects_change(
        self, tmp_path, monkeypatch
    ):
        """After overwriting the settings file, _engine_config returns
        the updated dict."""
        EngineImpl = _import_engine_impl()
        EngineRegistry = _import_registry()
        import krakey.engine_system.registry as reg_mod

        settings_path = "data/test/settings.yaml"
        settings_file = tmp_path / "data" / "test" / "settings.yaml"
        settings_file.parent.mkdir(parents=True)
        settings_file.write_text("v: 1\n", encoding="utf-8")

        monkeypatch.setattr(
            reg_mod, "_load_slot_catalog",
            lambda slot: (
                {"custom": EngineImpl(
                    cls=None,
                    description="test",
                    config_path=settings_path,
                )},
                "custom",
            ),
        )

        cfg = _make_registry_cfg(slot="memory", short_name="custom")
        registry = EngineRegistry(cfg, workspace_root=tmp_path)

        assert registry._engine_config("memory", "custom") == {"v": 1}
        settings_file.write_text("v: 99\n", encoding="utf-8")
        assert registry._engine_config("memory", "custom") == {"v": 99}


class TestEngineRegistryResolveConfigInjection:
    """Integration: resolve() passes the file's dict as config= kwarg."""

    def test_resolve_passes_file_config_to_engine_constructor(
        self, tmp_path, monkeypatch
    ):
        """When an engine impl has a config_path and a settings file exists
        under workspace_root, resolve() passes that file's dict as the
        engine's config= kwarg.

        Approach: inject a ConfigAwareImpl that captures its config kwarg;
        monkeypatch _load_slot_catalog to return it with a known config_path;
        write the settings file; call resolve(); assert the engine received
        the file's values.

        ASSUMPTION: resolve() uses _engine_config() internally (which in
        turn uses FileEngineConfigStore). The call chain is:
          resolve() → _engine_config(slot, short_name) →
          FileEngineConfigStore(workspace_root).read(config_path).
        """
        from typing import Protocol, runtime_checkable

        EngineImpl = _import_engine_impl()
        EngineRegistry = _import_registry()
        import krakey.engine_system.registry as reg_mod

        @runtime_checkable
        class _DummyProto(Protocol):
            def hello(self) -> str: ...

        captured: dict = {}

        class _CfgAwareImpl:
            def __init__(self, *, config=None):
                captured["config"] = config

            def hello(self) -> str:
                return "ok"

        settings_path = "data/test/settings.yaml"
        settings_file = tmp_path / "data" / "test" / "settings.yaml"
        settings_file.parent.mkdir(parents=True)
        settings_file.write_text("cache_size_mb: 200\n", encoding="utf-8")

        monkeypatch.setattr(
            reg_mod, "_load_slot_catalog",
            lambda slot: (
                {"custom": EngineImpl(
                    cls=_CfgAwareImpl,
                    description="test impl",
                    config_path=settings_path,
                )},
                "custom",
            ),
        )

        cfg = _make_registry_cfg(slot="memory", short_name="custom")
        registry = EngineRegistry(cfg, workspace_root=tmp_path)
        instance = registry.resolve("memory", expected_protocol=_DummyProto)
        assert instance.hello() == "ok"
        assert captured.get("config") == {"cache_size_mb": 200}, (
            f"Expected config={{'cache_size_mb': 200}}, got {captured.get('config')!r}"
        )

    def test_resolve_passes_empty_dict_when_no_settings_file(
        self, tmp_path, monkeypatch
    ):
        """When the impl's config_path file is absent, resolve() passes {}
        as config= (not None) — consistent with existing behaviour."""
        from typing import Protocol, runtime_checkable

        EngineImpl = _import_engine_impl()
        EngineRegistry = _import_registry()
        import krakey.engine_system.registry as reg_mod

        @runtime_checkable
        class _DummyProto(Protocol):
            def hello(self) -> str: ...

        captured: dict = {}

        class _CfgAwareImpl:
            def __init__(self, *, config=None):
                captured["config"] = config

            def hello(self) -> str:
                return "ok"

        monkeypatch.setattr(
            reg_mod, "_load_slot_catalog",
            lambda slot: (
                {"custom": EngineImpl(
                    cls=_CfgAwareImpl,
                    description="test impl",
                    config_path="data/nonexistent/settings.yaml",
                )},
                "custom",
            ),
        )

        cfg = _make_registry_cfg(slot="memory", short_name="custom")
        registry = EngineRegistry(cfg, workspace_root=tmp_path)
        registry.resolve("memory", expected_protocol=_DummyProto)
        assert captured.get("config") == {}

    def test_resolve_passes_empty_dict_when_config_path_is_empty(
        self, tmp_path, monkeypatch
    ):
        """When impl.config_path == "", resolve() passes {} as config=."""
        from typing import Protocol, runtime_checkable

        EngineImpl = _import_engine_impl()
        EngineRegistry = _import_registry()
        import krakey.engine_system.registry as reg_mod

        @runtime_checkable
        class _DummyProto(Protocol):
            def hello(self) -> str: ...

        captured: dict = {}

        class _CfgAwareImpl:
            def __init__(self, *, config=None):
                captured["config"] = config

            def hello(self) -> str:
                return "ok"

        monkeypatch.setattr(
            reg_mod, "_load_slot_catalog",
            lambda slot: (
                {"custom": EngineImpl(
                    cls=_CfgAwareImpl,
                    description="test impl",
                    config_path="",
                )},
                "custom",
            ),
        )

        cfg = _make_registry_cfg(slot="memory", short_name="custom")
        registry = EngineRegistry(cfg, workspace_root=tmp_path)
        registry.resolve("memory", expected_protocol=_DummyProto)
        assert captured.get("config") == {}


# ---------------------------------------------------------------------------
# ASSUMPTIONS (consolidated)
# ---------------------------------------------------------------------------
#
# A1. ``FileEngineConfigStore`` lives at
#     ``krakey/engine_system/config_store.py`` and is importable as
#     ``from krakey.engine_system.config_store import FileEngineConfigStore``.
#
# A2. ``FileEngineConfigStore.__init__`` signature is
#     ``(self, workspace_root: Path | str)``.  ``read(config_path: str)``
#     and ``write(config_path: str, config: dict) -> Path`` signatures match
#     the contract spec.
#
# A3. ``EngineImpl`` gains a ``config_path: str = ""`` field. Existing
#     fields (cls, description, config_schema, dependencies, post_install)
#     remain unchanged.
#
# A4. ``load_slot_meta`` parses ``config_path`` from each builtin_engines
#     entry and stores it on the returned EngineImpl.  Missing key → "".
#     Non-string values are coerced to str (consistent with the coercion
#     pattern used for other scalar fields in the loader).
#
# A5. ``EngineRegistry.__init__`` gains ``workspace_root: Path | str =
#     "workspace"`` as a keyword argument.
#
# A6. ``EngineRegistry._engine_config(slot, short_name)`` is REWRITTEN to:
#       1. Call ``_load_slot_catalog(slot)`` to get the catalog dict.
#       2. Look up ``catalog[short_name]`` to get the ``EngineImpl``.
#       3. Read ``impl.config_path``.
#       4. Return ``FileEngineConfigStore(self._workspace_root).read(impl.config_path)``.
#     It must NOT access ``cfg.engine_configs`` at all.
#
# A7. The lookup key in the catalog dict is the short_name string (same as
#     the ``name`` field in meta.yaml).
#
# A8. ``resolve()`` calls ``_engine_config(slot, name_or_path)`` when the
#     override is a short name (not a dotted path) and ``config`` is not
#     already in kwargs — same condition as the current code.
#
# A9. ``FileEngineConfigStore.read`` does NOT cache results in memory.
#     Each call reads from disk. Tests in
#     TestEngineRegistryEngineConfigStateTransition depend on this.
#     If caching is added, those tests become a correct RED signal for
#     the cache invalidation contract (which would also need spec'ing).
#
# A10. The Config model still has ``engine_configs`` at test-authoring time
#      (branch ``Engine_config_update``). The regression-guard tests in
#      TestEngineRegistryEngineConfigNegative use ``types.SimpleNamespace``
#      to simulate the post-removal Config shape (no engine_configs attr).
#      These tests are specifically designed to go RED on the existing code
#      path where _engine_config still conditionally reads engine_configs.
