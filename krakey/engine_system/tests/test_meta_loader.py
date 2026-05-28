"""Edge tests for ``krakey/engine_system/meta_loader.py``.

Target behavior: ``load_slot_meta(slot, *, engines_root=None)``
reads ``engines/<slot>/meta.yaml`` and returns
``({short_name: EngineImpl}, default_short_name)``.

THE PIVOTAL CHANGE UNDER TEST: ``config_schema`` must be read
per-engine-entry (from each ``builtin_engines`` list item), not from the
slot-level ``config_schema:`` sibling key.  A slot-level ``config_schema:``
must be silently ignored.

All tests use ``tmp_path`` fixtures and pass ``engines_root=Path(tmp)`` to
avoid touching the real ``krakey/engines/`` tree.

Test structure (pytest section comments mirror the outer describe pattern):
  - positive / equivalence-partition
  - BVA / boundary values
  - state-transition / cross-contamination
  - negative / error-guessing
  - _coerce_config_schema unit tests (coercion contract)
  - regression guards (existing behaviors that must still hold)
  - TestDependenciesPositive / TestDependenciesBoundary
  - TestPostInstallPositive / TestPostInstallBoundary
  - TestNewFieldsCrossContamination
  - TestNewFieldsNegative
  - _coerce_dependencies / _coerce_post_install unit tests
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from krakey.engine_system.catalog import EngineImpl
from krakey.engine_system.meta_loader import (
    MetaParseError,
    _coerce_config_schema,
    load_slot_meta,
)

try:
    from krakey.engine_system.meta_loader import _coerce_dependencies
    _HAVE_COERCE_DEPENDENCIES = True
except ImportError:
    _HAVE_COERCE_DEPENDENCIES = False

try:
    from krakey.engine_system.meta_loader import _coerce_post_install
    _HAVE_COERCE_POST_INSTALL = True
except ImportError:
    _HAVE_COERCE_POST_INSTALL = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_meta(tmp_path: Path, slot: str, content: str) -> Path:
    """Write ``content`` to ``<tmp_path>/<slot>/meta.yaml`` and return the
    meta.yaml path."""
    slot_dir = tmp_path / slot
    slot_dir.mkdir(parents=True, exist_ok=True)
    meta = slot_dir / "meta.yaml"
    meta.write_text(textwrap.dedent(content), encoding="utf-8")
    return meta


def _load(tmp_path: Path, slot: str):
    """Thin wrapper: ``load_slot_meta(slot, engines_root=tmp_path)``."""
    return load_slot_meta(slot, engines_root=tmp_path)


# ---------------------------------------------------------------------------
# POSITIVE / EQUIVALENCE-PARTITION
# ---------------------------------------------------------------------------


class TestPositive:
    """Valid inputs from each equivalence class return correct catalog."""

    # --- per-entry config_schema (THE PIVOTAL NEW BEHAVIOUR) ---

    def test_two_entries_distinct_per_entry_config_schema(self, tmp_path):
        """Multi-entry slot: each engine receives ONLY its own entry's
        config_schema.  The spec's central correctness requirement."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: alpha
                factory_module: some.pkg.alpha
                factory_attr: AlphaEngine
                default: true
                config_schema:
                  - field: x
                    type: number_int
              - name: beta
                factory_module: some.pkg.beta
                factory_attr: BetaEngine
                config_schema:
                  - field: y
                    type: text
        """)
        catalog, default = _load(tmp_path, "demo")
        assert catalog["alpha"].config_schema == [{"field": "x", "type": "number_int"}]
        assert catalog["beta"].config_schema == [{"field": "y", "type": "text"}]
        assert default == "alpha"

    def test_slot_level_config_schema_is_ignored(self, tmp_path):
        """A ``config_schema:`` key at the slot level (sibling of
        ``builtin_engines``) must NOT propagate into any engine's
        ``EngineImpl.config_schema``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: alpha
                factory_module: some.pkg.alpha
                factory_attr: AlphaEngine
                default: true
                config_schema:
                  - field: x
                    type: number_int
              - name: beta
                factory_module: some.pkg.beta
                factory_attr: BetaEngine
                config_schema:
                  - field: y
                    type: text
            config_schema:
              - field: z
                type: bool
        """)
        catalog, _ = _load(tmp_path, "demo")
        # Neither engine must contain the slot-level field "z"
        alpha_fields = [d.get("field") for d in catalog["alpha"].config_schema]
        beta_fields = [d.get("field") for d in catalog["beta"].config_schema]
        assert "z" not in alpha_fields, (
            "slot-level config_schema field 'z' leaked into alpha"
        )
        assert "z" not in beta_fields, (
            "slot-level config_schema field 'z' leaked into beta"
        )
        # Own fields still present
        assert "x" in alpha_fields
        assert "y" in beta_fields

    def test_return_type_is_tuple_of_dict_and_str(self, tmp_path):
        """Return type contract: ``(dict[str, EngineImpl], str)``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: some.mod
                factory_attr: Cls
                default: true
                config_schema:
                  - field: a
                    type: text
        """)
        result = _load(tmp_path, "demo")
        assert isinstance(result, tuple) and len(result) == 2
        catalog, default = result
        assert isinstance(catalog, dict)
        assert isinstance(default, str)

    def test_catalog_values_are_engine_impl_instances(self, tmp_path):
        """Every value in the returned catalog must be an ``EngineImpl``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: some.mod
                factory_attr: Cls
                default: true
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert all(isinstance(v, EngineImpl) for v in catalog.values())

    def test_description_read_per_entry(self, tmp_path):
        """Regression: each entry's ``description`` key is preserved per entry."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: alpha
                factory_module: pkg.a
                factory_attr: A
                default: true
                description: Alpha description
              - name: beta
                factory_module: pkg.b
                factory_attr: B
                description: Beta description
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["alpha"].description == "Alpha description"
        assert catalog["beta"].description == "Beta description"

    def test_per_entry_config_schema_multiple_fields(self, tmp_path):
        """Per-entry config_schema with multiple fields is preserved verbatim."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg.a
                factory_attr: A
                default: true
                config_schema:
                  - field: temperature
                    type: number_float
                    default: 0.7
                    help: Sampling temperature
                  - field: max_tokens
                    type: number_int
                    default: 512
        """)
        catalog, _ = _load(tmp_path, "demo")
        schema = catalog["eng"].config_schema
        assert len(schema) == 2
        assert schema[0] == {
            "field": "temperature",
            "type": "number_float",
            "default": 0.7,
            "help": "Sampling temperature",
        }
        assert schema[1] == {"field": "max_tokens", "type": "number_int", "default": 512}

    def test_default_name_from_marked_entry(self, tmp_path):
        """Explicitly marked ``default: true`` entry becomes default_short_name."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: slow_engine
                factory_module: pkg
                factory_attr: Slow
              - name: fast_engine
                factory_module: pkg
                factory_attr: Fast
                default: true
        """)
        _, default = _load(tmp_path, "demo")
        assert default == "fast_engine"


# ---------------------------------------------------------------------------
# BVA / BOUNDARY VALUES
# ---------------------------------------------------------------------------


class TestBoundaryValues:
    """Boundary conditions on collection sizes, missing keys, and
    coercion edge cases."""

    # --- missing per-entry config_schema → [] ---

    def test_entry_missing_config_schema_yields_empty_list(self, tmp_path):
        """An entry with NO ``config_schema`` key must produce
        ``EngineImpl.config_schema == []``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].config_schema == []

    def test_entry_explicit_null_config_schema_yields_empty_list(self, tmp_path):
        """``config_schema: null`` (yaml None) in an entry coerces to ``[]``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                config_schema:
        """)  # YAML bare key → null
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].config_schema == []

    def test_entry_empty_list_config_schema_yields_empty_list(self, tmp_path):
        """``config_schema: []`` in an entry passes through as ``[]``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                config_schema: []
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].config_schema == []

    def test_entry_config_schema_single_field(self, tmp_path):
        """Boundary: exactly one field in the list passes through correctly."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                config_schema:
                  - field: lone
                    type: text
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].config_schema == [{"field": "lone", "type": "text"}]

    # --- slot-level only (no per-entry) → every engine gets [] ---

    def test_slot_level_only_config_schema_means_all_engines_get_empty(self, tmp_path):
        """Only a slot-level ``config_schema`` (no per-entry ones):
        every engine's ``config_schema`` must be ``[]``."""
        _write_meta(tmp_path, "demo", """
            config_schema:
              - field: slot_field
                type: text
            builtin_engines:
              - name: eng_a
                factory_module: pkg
                factory_attr: A
                default: true
              - name: eng_b
                factory_module: pkg
                factory_attr: B
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng_a"].config_schema == []
        assert catalog["eng_b"].config_schema == []

    # --- single-entry implicit default ---

    def test_single_entry_no_default_flag_is_implicit_default(self, tmp_path):
        """Single-entry catalog without ``default: true`` → that entry is
        the implicit default."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: only_one
                factory_module: pkg
                factory_attr: Cls
        """)
        catalog, default = _load(tmp_path, "demo")
        assert default == "only_one"
        assert set(catalog.keys()) == {"only_one"}

    def test_single_entry_with_default_true_still_works(self, tmp_path):
        """Single entry with explicit ``default: true`` is valid."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: only_one
                factory_module: pkg
                factory_attr: Cls
                default: true
        """)
        _, default = _load(tmp_path, "demo")
        assert default == "only_one"

    # --- large config_schema list (many fields) ---

    def test_many_fields_in_per_entry_config_schema(self, tmp_path):
        """Large per-entry config_schema list (10 fields) passes verbatim."""
        fields_yaml = "\n".join(
            f"                  - field: f{i}\n                    type: text"
            for i in range(10)
        )
        _write_meta(tmp_path, "demo", f"""
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                config_schema:
{fields_yaml}
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert len(catalog["eng"].config_schema) == 10
        for i, item in enumerate(catalog["eng"].config_schema):
            assert item["field"] == f"f{i}"

    # --- non-dict items inside per-entry config_schema are dropped ---

    def test_non_dict_items_in_per_entry_config_schema_are_dropped(self, tmp_path):
        """Non-dict items within a per-entry ``config_schema`` list are
        filtered out; valid dicts are kept."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                config_schema:
                  - field: good
                    type: text
                  - just_a_string
                  - 42
        """)
        catalog, _ = _load(tmp_path, "demo")
        # Only the dict item survives
        assert catalog["eng"].config_schema == [{"field": "good", "type": "text"}]

    def test_all_non_dict_items_in_per_entry_config_schema_gives_empty(self, tmp_path):
        """If every item in a per-entry config_schema list is non-dict,
        the result is ``[]``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                config_schema:
                  - just_a_string
                  - 99
                  - true
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].config_schema == []

    # --- factory is NOT imported at load time ---

    def test_bogus_factory_module_does_not_raise_at_load_time(self, tmp_path):
        """Lazy import: a completely non-existent factory_module must NOT
        cause any ImportError during ``load_slot_meta``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: totally_nonexistent_module_xyz_999
                factory_attr: DoesNotExist
                default: true
        """)
        # Must not raise ImportError (or any error)
        catalog, default = _load(tmp_path, "demo")
        assert "eng" in catalog
        assert default == "eng"


# ---------------------------------------------------------------------------
# STATE-TRANSITION / CROSS-CONTAMINATION
# ---------------------------------------------------------------------------


class TestCrossContamination:
    """Verify that two entries in the SAME slot do not share schema state.
    The pivotal regression: a loop-level re-use of a single variable could
    cause all entries to end up with the LAST entry's schema."""

    def test_three_entries_each_gets_own_schema(self, tmp_path):
        """Three entries each with distinct config_schema — no entry
        should acquire another's fields."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: a
                factory_module: pkg
                factory_attr: A
                default: true
                config_schema:
                  - field: alpha_field
                    type: text
              - name: b
                factory_module: pkg
                factory_attr: B
                config_schema:
                  - field: beta_field
                    type: number_int
              - name: c
                factory_module: pkg
                factory_attr: C
                config_schema:
                  - field: gamma_field
                    type: bool
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert [d["field"] for d in catalog["a"].config_schema] == ["alpha_field"]
        assert [d["field"] for d in catalog["b"].config_schema] == ["beta_field"]
        assert [d["field"] for d in catalog["c"].config_schema] == ["gamma_field"]

    def test_entry_with_schema_does_not_contaminate_entry_without(self, tmp_path):
        """When entry A has a config_schema and entry B has none, B must
        still get ``[]`` — not A's schema."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: has_schema
                factory_module: pkg
                factory_attr: A
                default: true
                config_schema:
                  - field: from_a
                    type: text
              - name: no_schema
                factory_module: pkg
                factory_attr: B
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["has_schema"].config_schema == [{"field": "from_a", "type": "text"}]
        assert catalog["no_schema"].config_schema == []

    def test_entry_without_schema_does_not_contaminate_entry_with(self, tmp_path):
        """Converse: B (no schema) before A (has schema) — A still gets
        only its own schema."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: no_schema
                factory_module: pkg
                factory_attr: B
                default: true
              - name: has_schema
                factory_module: pkg
                factory_attr: A
                config_schema:
                  - field: from_a
                    type: text
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["no_schema"].config_schema == []
        assert catalog["has_schema"].config_schema == [{"field": "from_a", "type": "text"}]

    def test_per_entry_schemas_are_separate_objects_not_shared(self, tmp_path):
        """The EngineImpl.config_schema lists for two entries must be
        distinct objects (not the same list ref) even when they have the
        same content."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: a
                factory_module: pkg
                factory_attr: A
                default: true
                config_schema:
                  - field: shared_field
                    type: text
              - name: b
                factory_module: pkg
                factory_attr: B
                config_schema:
                  - field: shared_field
                    type: text
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["a"].config_schema is not catalog["b"].config_schema, (
            "config_schema lists must not be the same object (aliasing risk)"
        )

    def test_slot_level_schema_plus_per_entry_schema_no_merge(self, tmp_path):
        """Slot-level AND per-entry both present: per-entry wins; slot-level
        field must not appear in any engine's schema."""
        _write_meta(tmp_path, "demo", """
            config_schema:
              - field: z
                type: bool
            builtin_engines:
              - name: alpha
                factory_module: pkg
                factory_attr: A
                default: true
                config_schema:
                  - field: x
                    type: number_int
              - name: beta
                factory_module: pkg
                factory_attr: B
                config_schema:
                  - field: y
                    type: number_float
        """)
        catalog, _ = _load(tmp_path, "demo")
        all_alpha_fields = {d.get("field") for d in catalog["alpha"].config_schema}
        all_beta_fields = {d.get("field") for d in catalog["beta"].config_schema}
        assert "z" not in all_alpha_fields
        assert "z" not in all_beta_fields
        assert all_alpha_fields == {"x"}
        assert all_beta_fields == {"y"}

    def test_two_loads_same_slot_independent_results(self, tmp_path):
        """Calling load_slot_meta twice on the same slot must return
        independent catalog dicts (no shared mutable state between calls)."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                config_schema:
                  - field: val
                    type: text
        """)
        catalog1, _ = _load(tmp_path, "demo")
        catalog2, _ = _load(tmp_path, "demo")
        # Mutating one result must not affect the other
        catalog1["eng"].config_schema.append({"field": "injected"})
        assert len(catalog2["eng"].config_schema) == 1, (
            "Second call returned the same config_schema list object as the first call"
        )


# ---------------------------------------------------------------------------
# NEGATIVE / ERROR-GUESSING
# ---------------------------------------------------------------------------


class TestNegative:
    """All error paths that the spec mandates must raise."""

    # --- FileNotFoundError ---

    def test_missing_slot_directory_raises_file_not_found(self, tmp_path):
        """No ``<slot>/`` directory at all → ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError):
            _load(tmp_path, "nonexistent_slot")

    def test_missing_meta_yaml_raises_file_not_found(self, tmp_path):
        """Slot directory exists but has no ``meta.yaml`` → ``FileNotFoundError``."""
        (tmp_path / "empty_slot").mkdir()
        with pytest.raises(FileNotFoundError):
            _load(tmp_path, "empty_slot")

    def test_file_not_found_message_mentions_slot(self, tmp_path):
        """FileNotFoundError message should name the missing slot."""
        with pytest.raises(FileNotFoundError) as exc_info:
            _load(tmp_path, "ghost_slot")
        assert "ghost_slot" in str(exc_info.value)

    # --- MetaParseError: structural issues ---

    def test_top_level_not_a_mapping_raises(self, tmp_path):
        """YAML that is a scalar (not a mapping) at the top level → MetaParseError."""
        _write_meta(tmp_path, "demo", "just a string\n")
        with pytest.raises(MetaParseError):
            _load(tmp_path, "demo")

    def test_top_level_list_raises(self, tmp_path):
        """YAML that is a list at the top level → MetaParseError."""
        _write_meta(tmp_path, "demo", "- item1\n- item2\n")
        with pytest.raises(MetaParseError):
            _load(tmp_path, "demo")

    def test_missing_builtin_engines_key_raises(self, tmp_path):
        """No ``builtin_engines`` key → MetaParseError."""
        _write_meta(tmp_path, "demo", "slot: demo\ndescription: desc\n")
        with pytest.raises(MetaParseError):
            _load(tmp_path, "demo")

    def test_empty_builtin_engines_list_raises(self, tmp_path):
        """``builtin_engines: []`` → MetaParseError."""
        _write_meta(tmp_path, "demo", """
            builtin_engines: []
        """)
        with pytest.raises(MetaParseError):
            _load(tmp_path, "demo")

    def test_builtin_engines_null_raises(self, tmp_path):
        """``builtin_engines:`` (null) → MetaParseError."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
        """)
        with pytest.raises(MetaParseError):
            _load(tmp_path, "demo")

    def test_builtin_engines_not_a_list_raises(self, tmp_path):
        """``builtin_engines`` is a string → MetaParseError."""
        _write_meta(tmp_path, "demo", """
            builtin_engines: "should be a list"
        """)
        with pytest.raises(MetaParseError):
            _load(tmp_path, "demo")

    def test_entry_not_a_mapping_raises(self, tmp_path):
        """An entry inside ``builtin_engines`` that is a plain scalar
        → MetaParseError."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - just_a_string
        """)
        with pytest.raises(MetaParseError):
            _load(tmp_path, "demo")

    def test_entry_missing_name_raises(self, tmp_path):
        """Entry without ``name`` → MetaParseError."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - factory_module: pkg
                factory_attr: Cls
        """)
        with pytest.raises(MetaParseError):
            _load(tmp_path, "demo")

    def test_entry_missing_factory_module_raises(self, tmp_path):
        """Entry without ``factory_module`` → MetaParseError."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_attr: Cls
        """)
        with pytest.raises(MetaParseError):
            _load(tmp_path, "demo")

    def test_entry_missing_factory_attr_raises(self, tmp_path):
        """Entry without ``factory_attr`` → MetaParseError."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
        """)
        with pytest.raises(MetaParseError):
            _load(tmp_path, "demo")

    def test_multiple_default_true_raises(self, tmp_path):
        """Two entries both marked ``default: true`` → MetaParseError."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng_a
                factory_module: pkg
                factory_attr: A
                default: true
              - name: eng_b
                factory_module: pkg
                factory_attr: B
                default: true
        """)
        with pytest.raises(MetaParseError):
            _load(tmp_path, "demo")

    def test_multi_entry_no_default_raises(self, tmp_path):
        """Multiple entries with none marked ``default: true``
        → MetaParseError (ambiguous default)."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng_a
                factory_module: pkg
                factory_attr: A
              - name: eng_b
                factory_module: pkg
                factory_attr: B
        """)
        with pytest.raises(MetaParseError):
            _load(tmp_path, "demo")

    def test_meta_parse_error_not_file_not_found(self, tmp_path):
        """Confirm that structural errors raise MetaParseError, not
        FileNotFoundError — they are distinct failure modes."""
        _write_meta(tmp_path, "demo", "just a string\n")
        with pytest.raises(MetaParseError):
            _load(tmp_path, "demo")
        # Ensure it's specifically MetaParseError
        with pytest.raises(MetaParseError) as exc_info:
            _load(tmp_path, "demo")
        assert not isinstance(exc_info.value, FileNotFoundError)

    def test_per_entry_config_schema_is_a_dict_not_list_coerced_to_empty(self, tmp_path):
        """Entry-level ``config_schema`` that is a plain dict (not a list)
        is a malformed type → coerced to ``[]`` (not MetaParseError)."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                config_schema:
                  field: wrong_shape
                  type: text
        """)
        # Must not raise — coercion handles it
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].config_schema == []

    def test_per_entry_config_schema_is_a_string_coerced_to_empty(self, tmp_path):
        """Entry-level ``config_schema`` that is a bare string → ``[]``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                config_schema: "not a list"
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].config_schema == []

    def test_per_entry_config_schema_is_a_number_coerced_to_empty(self, tmp_path):
        """Entry-level ``config_schema`` that is an integer → ``[]``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                config_schema: 42
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].config_schema == []

    def test_entry_name_empty_string_raises(self, tmp_path):
        """An entry with ``name: ""`` (falsy) → MetaParseError (name is
        required to be non-empty)."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: ""
                factory_module: pkg
                factory_attr: Cls
        """)
        with pytest.raises(MetaParseError):
            _load(tmp_path, "demo")


# ---------------------------------------------------------------------------
# _coerce_config_schema UNIT TESTS
# ---------------------------------------------------------------------------


class TestCoerceConfigSchema:
    """Direct unit tests for the ``_coerce_config_schema`` helper.
    Spec: list-of-dicts passes verbatim (each dict copied);
    non-list → []; non-dict items within a list are dropped."""

    def test_none_yields_empty_list(self):
        assert _coerce_config_schema(None) == []

    def test_string_yields_empty_list(self):
        assert _coerce_config_schema("hello") == []

    def test_dict_yields_empty_list(self):
        """A bare dict (not wrapped in a list) → []."""
        assert _coerce_config_schema({"field": "x"}) == []

    def test_integer_yields_empty_list(self):
        assert _coerce_config_schema(42) == []

    def test_float_yields_empty_list(self):
        assert _coerce_config_schema(3.14) == []

    def test_bool_yields_empty_list(self):
        assert _coerce_config_schema(True) == []

    def test_empty_list_yields_empty_list(self):
        assert _coerce_config_schema([]) == []

    def test_list_of_dicts_passes_through(self):
        raw = [{"field": "a", "type": "text"}, {"field": "b", "type": "bool"}]
        result = _coerce_config_schema(raw)
        assert result == raw

    def test_list_of_dicts_returns_copies_not_same_objects(self):
        """Each dict in the output must be a copy, not the same object."""
        original = {"field": "a", "type": "text"}
        raw = [original]
        result = _coerce_config_schema(raw)
        assert result[0] == original
        assert result[0] is not original, (
            "_coerce_config_schema must copy each dict, not alias it"
        )

    def test_list_with_mixed_types_keeps_only_dicts(self):
        raw = [
            {"field": "good"},
            "string_item",
            42,
            None,
            ["nested_list"],
            {"field": "also_good"},
        ]
        result = _coerce_config_schema(raw)
        assert result == [{"field": "good"}, {"field": "also_good"}]

    def test_list_of_only_non_dicts_yields_empty_list(self):
        assert _coerce_config_schema(["a", "b", 1, None]) == []

    def test_single_dict_in_list_passes_through(self):
        raw = [{"field": "sole", "type": "number_int", "default": 0}]
        result = _coerce_config_schema(raw)
        assert result == raw

    def test_output_mutation_does_not_affect_input(self):
        """Output list is independent of input — mutations don't propagate."""
        raw = [{"field": "x"}]
        result = _coerce_config_schema(raw)
        result[0]["field"] = "mutated"
        assert raw[0]["field"] == "x", (
            "mutation of output dict must not affect the input dict"
        )


# ---------------------------------------------------------------------------
# DEPENDENCIES FIELD — POSITIVE / EQUIVALENCE-PARTITION
# ---------------------------------------------------------------------------


class TestDependenciesPositive:
    """Valid ``dependencies`` inputs return a correctly coerced list."""

    def test_single_entry_with_two_deps_exact_order_preserved(self, tmp_path):
        """``dependencies: ["foo>=1.0", "bar"]`` → exact list in order."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                dependencies:
                  - foo>=1.0
                  - bar
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].dependencies == ["foo>=1.0", "bar"]

    def test_single_entry_complex_version_spec(self, tmp_path):
        """Complex pip spec with both >= and < bounds is kept verbatim."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                dependencies:
                  - "MemoryOS>=2.0.17,<3.0"
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].dependencies == ["MemoryOS>=2.0.17,<3.0"]

    def test_dependencies_field_present_on_engine_impl(self, tmp_path):
        """``EngineImpl`` must expose a ``dependencies`` attribute."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                dependencies:
                  - somepkg
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert hasattr(catalog["eng"], "dependencies")

    def test_two_entries_distinct_deps_each_gets_own(self, tmp_path):
        """THE PIVOTAL MULTI-ENTRY TEST: two engines each with distinct
        ``dependencies`` — each ``EngineImpl`` carries only its own list."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: alpha
                factory_module: pkg
                factory_attr: A
                default: true
                dependencies:
                  - alpha-dep>=1.0
                  - shared-dep
              - name: beta
                factory_module: pkg
                factory_attr: B
                dependencies:
                  - beta-dep==2.3
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["alpha"].dependencies == ["alpha-dep>=1.0", "shared-dep"]
        assert catalog["beta"].dependencies == ["beta-dep==2.3"]

    def test_two_entries_distinct_deps_no_cross_aliasing(self, tmp_path):
        """Mutating one engine's dependencies list must not affect the other."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: alpha
                factory_module: pkg
                factory_attr: A
                default: true
                dependencies:
                  - dep-a
              - name: beta
                factory_module: pkg
                factory_attr: B
                dependencies:
                  - dep-b
        """)
        catalog, _ = _load(tmp_path, "demo")
        catalog["alpha"].dependencies.append("injected")
        assert catalog["beta"].dependencies == ["dep-b"], (
            "Mutating alpha.dependencies must not affect beta.dependencies"
        )

    def test_five_deps_all_kept(self, tmp_path):
        """A list of five valid requirement specs all pass through."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                dependencies:
                  - dep-a>=1.0
                  - dep-b
                  - dep-c!=2.0
                  - dep-d[extra]
                  - dep-e>=1,<3
        """)
        catalog, _ = _load(tmp_path, "demo")
        result = catalog["eng"].dependencies
        assert len(result) == 5
        assert result[0] == "dep-a>=1.0"
        assert result[4] == "dep-e>=1,<3"


# ---------------------------------------------------------------------------
# DEPENDENCIES FIELD — BOUNDARY VALUES
# ---------------------------------------------------------------------------


class TestDependenciesBoundary:
    """Boundary and coercion edge cases for ``dependencies``."""

    def test_key_absent_yields_empty_list(self, tmp_path):
        """No ``dependencies`` key → ``EngineImpl.dependencies == []``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].dependencies == []

    def test_key_explicit_null_yields_empty_list(self, tmp_path):
        """``dependencies:`` (yaml null) → ``[]``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                dependencies:
        """)  # bare key → null
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].dependencies == []

    def test_explicit_empty_list_yields_empty_list(self, tmp_path):
        """``dependencies: []`` → ``[]``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                dependencies: []
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].dependencies == []

    def test_single_dep_is_preserved(self, tmp_path):
        """Boundary: exactly one valid dep string → one-element list."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                dependencies:
                  - only-one
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].dependencies == ["only-one"]

    def test_whitespace_string_is_stripped(self, tmp_path):
        """``"  trimmed  "`` → ``"trimmed"`` (leading/trailing whitespace stripped)."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                dependencies:
                  - "  trimmed  "
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].dependencies == ["trimmed"]

    def test_empty_string_item_is_filtered(self, tmp_path):
        """``dependencies: [""]`` → ``[]`` (empty string after strip is dropped)."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                dependencies:
                  - ""
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].dependencies == []

    def test_whitespace_only_string_item_is_filtered(self, tmp_path):
        """A string of only spaces strips to empty and is filtered out."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                dependencies:
                  - "   "
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].dependencies == []

    def test_non_string_items_dropped_int(self, tmp_path):
        """Integer items in the list are silently dropped."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                dependencies:
                  - 42
                  - ok-dep
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].dependencies == ["ok-dep"]

    def test_non_string_items_dropped_bool(self, tmp_path):
        """Boolean items in the list are silently dropped."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                dependencies:
                  - true
                  - good-dep
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].dependencies == ["good-dep"]

    def test_mixed_invalid_and_valid_items_the_pivotal_coercion_test(self, tmp_path):
        """THE PIVOTAL COERCION TEST: ``["  foo>=1.0  ", "", null_equiv, 42, "bar"]``
        → exactly ``["foo>=1.0", "bar"]``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                dependencies:
                  - "  foo>=1.0  "
                  - ""
                  - ~
                  - 42
                  - bar
        """)  # ~ is YAML null
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].dependencies == ["foo>=1.0", "bar"]

    def test_dependencies_result_is_independent_list_object(self, tmp_path):
        """Two calls to load_slot_meta return independent list objects for
        ``dependencies`` — no shared module-level state."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                dependencies:
                  - dep-a
        """)
        catalog1, _ = _load(tmp_path, "demo")
        catalog2, _ = _load(tmp_path, "demo")
        catalog1["eng"].dependencies.append("injected")
        assert catalog2["eng"].dependencies == ["dep-a"], (
            "Second call must not return the same dependencies list object"
        )


# ---------------------------------------------------------------------------
# POST_INSTALL FIELD — POSITIVE / EQUIVALENCE-PARTITION
# ---------------------------------------------------------------------------


class TestPostInstallPositive:
    """Valid ``post_install`` inputs return correctly normalized entries."""

    def test_single_entry_full_descriptor_all_fields(self, tmp_path):
        """A ``post_install`` entry with all three fields (args, description,
        optional) is normalized correctly."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
                  - args:
                      - "{python}"
                      - "-m"
                      - "foo"
                    description: "Run foo module"
                    optional: true
        """)
        catalog, _ = _load(tmp_path, "demo")
        result = catalog["eng"].post_install
        assert len(result) == 1
        assert result[0]["args"] == ["{python}", "-m", "foo"]
        assert result[0]["description"] == "Run foo module"
        assert result[0]["optional"] is True

    def test_post_install_field_present_on_engine_impl(self, tmp_path):
        """``EngineImpl`` must expose a ``post_install`` attribute."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
                  - args:
                      - do-something
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert hasattr(catalog["eng"], "post_install")

    def test_entry_with_only_args_defaults_description_and_optional(self, tmp_path):
        """An entry with only ``args`` → description defaults to ``""``
        and optional defaults to ``False``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
                  - args:
                      - pip
                      - install
                      - something
        """)
        catalog, _ = _load(tmp_path, "demo")
        entry = catalog["eng"].post_install[0]
        assert entry["args"] == ["pip", "install", "something"]
        assert entry["description"] == ""
        assert entry["optional"] is False

    def test_entry_with_optional_false_explicit(self, tmp_path):
        """Explicitly ``optional: false`` passes through correctly."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
                  - args:
                      - cmd
                    optional: false
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].post_install[0]["optional"] is False

    def test_entry_with_optional_true_explicit(self, tmp_path):
        """Explicitly ``optional: true`` passes through correctly."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
                  - args:
                      - cmd
                    optional: true
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].post_install[0]["optional"] is True

    def test_multiple_valid_post_install_entries_all_kept(self, tmp_path):
        """Multiple valid entries are all kept in order."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
                  - args:
                      - step-one
                    description: First
                  - args:
                      - step-two
                    description: Second
                    optional: true
        """)
        catalog, _ = _load(tmp_path, "demo")
        result = catalog["eng"].post_install
        assert len(result) == 2
        assert result[0]["args"] == ["step-one"]
        assert result[0]["description"] == "First"
        assert result[1]["args"] == ["step-two"]
        assert result[1]["optional"] is True

    def test_normalized_entry_has_exactly_three_keys(self, tmp_path):
        """Each normalized post_install entry dict must have exactly
        ``args``, ``description``, and ``optional`` — no extra keys."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
                  - args:
                      - cmd
                    description: test
                    optional: false
        """)
        catalog, _ = _load(tmp_path, "demo")
        entry = catalog["eng"].post_install[0]
        assert set(entry.keys()) == {"args", "description", "optional"}

    def test_two_entries_distinct_post_install_each_gets_own(self, tmp_path):
        """THE PIVOTAL MULTI-ENTRY POST_INSTALL TEST: two engines with distinct
        ``post_install`` lists — each carries only its own."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: alpha
                factory_module: pkg
                factory_attr: A
                default: true
                post_install:
                  - args:
                      - alpha-cmd
                    description: Alpha step
              - name: beta
                factory_module: pkg
                factory_attr: B
                post_install:
                  - args:
                      - beta-cmd
                    description: Beta step
                    optional: true
        """)
        catalog, _ = _load(tmp_path, "demo")
        alpha_pi = catalog["alpha"].post_install
        beta_pi = catalog["beta"].post_install
        assert len(alpha_pi) == 1
        assert alpha_pi[0]["args"] == ["alpha-cmd"]
        assert alpha_pi[0]["description"] == "Alpha step"
        assert len(beta_pi) == 1
        assert beta_pi[0]["args"] == ["beta-cmd"]
        assert beta_pi[0]["optional"] is True

    def test_multi_entry_slot_both_deps_and_post_install(self, tmp_path):
        """Two engines: alpha has dependencies only, beta has post_install only.
        Each field on the other engine must be ``[]``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: alpha
                factory_module: pkg
                factory_attr: A
                default: true
                dependencies:
                  - dep-only
              - name: beta
                factory_module: pkg
                factory_attr: B
                post_install:
                  - args:
                      - post-only-cmd
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["alpha"].dependencies == ["dep-only"]
        assert catalog["alpha"].post_install == []
        assert catalog["beta"].dependencies == []
        assert catalog["beta"].post_install[0]["args"] == ["post-only-cmd"]


# ---------------------------------------------------------------------------
# POST_INSTALL FIELD — BOUNDARY VALUES
# ---------------------------------------------------------------------------


class TestPostInstallBoundary:
    """Boundary and coercion edge cases for ``post_install``."""

    def test_key_absent_yields_empty_list(self, tmp_path):
        """No ``post_install`` key → ``EngineImpl.post_install == []``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].post_install == []

    def test_key_explicit_null_yields_empty_list(self, tmp_path):
        """``post_install:`` (yaml null) → ``[]``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].post_install == []

    def test_explicit_empty_list_yields_empty_list(self, tmp_path):
        """``post_install: []`` → ``[]``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install: []
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].post_install == []

    def test_single_valid_entry_boundary_min_size(self, tmp_path):
        """Boundary: exactly one valid entry in post_install is kept."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
                  - args:
                      - sole-cmd
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert len(catalog["eng"].post_install) == 1
        assert catalog["eng"].post_install[0]["args"] == ["sole-cmd"]

    def test_args_with_single_element(self, tmp_path):
        """Boundary: args list with exactly one element is valid."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
                  - args:
                      - single-arg
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].post_install[0]["args"] == ["single-arg"]

    def test_post_install_result_is_independent_list_object(self, tmp_path):
        """Two calls return independent post_install lists — no shared state."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
                  - args:
                      - cmd
        """)
        catalog1, _ = _load(tmp_path, "demo")
        catalog2, _ = _load(tmp_path, "demo")
        catalog1["eng"].post_install.append({"args": ["injected"], "description": "", "optional": False})
        assert len(catalog2["eng"].post_install) == 1, (
            "Second call must not return the same post_install list object"
        )


# ---------------------------------------------------------------------------
# NEW FIELDS — CROSS-CONTAMINATION / STATE-TRANSITION
# ---------------------------------------------------------------------------


class TestNewFieldsCrossContamination:
    """Verify no cross-entry aliasing for ``dependencies`` and ``post_install``
    across multiple entries in the same slot and across multiple load calls."""

    def test_three_entries_each_gets_own_dependencies(self, tmp_path):
        """Three entries with distinct deps: no entry acquires another's."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: a
                factory_module: pkg
                factory_attr: A
                default: true
                dependencies:
                  - dep-a
              - name: b
                factory_module: pkg
                factory_attr: B
                dependencies:
                  - dep-b1
                  - dep-b2
              - name: c
                factory_module: pkg
                factory_attr: C
                dependencies:
                  - dep-c
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["a"].dependencies == ["dep-a"]
        assert catalog["b"].dependencies == ["dep-b1", "dep-b2"]
        assert catalog["c"].dependencies == ["dep-c"]

    def test_entry_with_deps_does_not_contaminate_entry_without(self, tmp_path):
        """Entry A has dependencies, entry B has none — B gets ``[]``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: has_deps
                factory_module: pkg
                factory_attr: A
                default: true
                dependencies:
                  - some-dep
              - name: no_deps
                factory_module: pkg
                factory_attr: B
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["has_deps"].dependencies == ["some-dep"]
        assert catalog["no_deps"].dependencies == []

    def test_entry_without_post_install_does_not_inherit_from_prev(self, tmp_path):
        """Entry A has post_install, entry B has none — B gets ``[]``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: has_pi
                factory_module: pkg
                factory_attr: A
                default: true
                post_install:
                  - args:
                      - cmd-a
              - name: no_pi
                factory_module: pkg
                factory_attr: B
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert len(catalog["has_pi"].post_install) == 1
        assert catalog["no_pi"].post_install == []

    def test_two_entries_post_install_lists_are_separate_objects(self, tmp_path):
        """Even when two entries have same post_install content, they must be
        distinct list objects (no aliasing)."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: a
                factory_module: pkg
                factory_attr: A
                default: true
                post_install:
                  - args:
                      - cmd
              - name: b
                factory_module: pkg
                factory_attr: B
                post_install:
                  - args:
                      - cmd
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["a"].post_install is not catalog["b"].post_install, (
            "post_install lists must not be the same object"
        )

    def test_two_entries_dependencies_lists_are_separate_objects(self, tmp_path):
        """Even with identical dep content, two entries must get distinct
        list objects for ``dependencies``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: a
                factory_module: pkg
                factory_attr: A
                default: true
                dependencies:
                  - same-dep
              - name: b
                factory_module: pkg
                factory_attr: B
                dependencies:
                  - same-dep
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["a"].dependencies is not catalog["b"].dependencies, (
            "dependencies lists must not be the same object"
        )

    def test_two_loads_same_slot_independent_dependencies_lists(self, tmp_path):
        """Two separate load_slot_meta calls on the same slot produce independent
        ``dependencies`` lists — no shared module-level state."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                dependencies:
                  - dep-x
        """)
        catalog1, _ = _load(tmp_path, "demo")
        catalog2, _ = _load(tmp_path, "demo")
        catalog1["eng"].dependencies.append("injected")
        assert catalog2["eng"].dependencies == ["dep-x"], (
            "Second call should return a new dependencies list, not the same object"
        )

    def test_two_loads_same_slot_independent_post_install_lists(self, tmp_path):
        """Two separate load_slot_meta calls produce independent ``post_install``
        lists."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
                  - args:
                      - cmd
        """)
        catalog1, _ = _load(tmp_path, "demo")
        catalog2, _ = _load(tmp_path, "demo")
        catalog1["eng"].post_install.append(
            {"args": ["injected"], "description": "", "optional": False}
        )
        assert len(catalog2["eng"].post_install) == 1, (
            "Second call should return a new post_install list"
        )

    def test_all_four_fields_in_one_slot_two_entries(self, tmp_path):
        """THE COMBINED PIVOTAL TEST: two engines each with distinct
        ``dependencies`` AND ``post_install`` — all four field values
        land on the correct engine, no cross-entry aliasing."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: alpha
                factory_module: pkg
                factory_attr: A
                default: true
                dependencies:
                  - alpha-dep>=1.0
                  - shared-dep
                post_install:
                  - args:
                      - alpha-cmd
                    description: Alpha post-install
                    optional: false
              - name: beta
                factory_module: pkg
                factory_attr: B
                dependencies:
                  - beta-dep==2.3
                post_install:
                  - args:
                      - beta-cmd-1
                  - args:
                      - beta-cmd-2
                    optional: true
        """)
        catalog, _ = _load(tmp_path, "demo")
        # dependencies
        assert catalog["alpha"].dependencies == ["alpha-dep>=1.0", "shared-dep"]
        assert catalog["beta"].dependencies == ["beta-dep==2.3"]
        # post_install
        alpha_pi = catalog["alpha"].post_install
        assert len(alpha_pi) == 1
        assert alpha_pi[0]["args"] == ["alpha-cmd"]
        assert alpha_pi[0]["description"] == "Alpha post-install"
        assert alpha_pi[0]["optional"] is False
        beta_pi = catalog["beta"].post_install
        assert len(beta_pi) == 2
        assert beta_pi[0]["args"] == ["beta-cmd-1"]
        assert beta_pi[1]["optional"] is True
        # independence
        assert catalog["alpha"].dependencies is not catalog["beta"].dependencies
        assert catalog["alpha"].post_install is not catalog["beta"].post_install


# ---------------------------------------------------------------------------
# NEW FIELDS — NEGATIVE / ERROR-GUESSING
# ---------------------------------------------------------------------------


class TestNewFieldsNegative:
    """Tolerant coercion: bad values must NOT raise; they are silently
    normalised to ``[]`` or the entry is skipped."""

    # --- dependencies: wrong top-level type ---

    def test_dependencies_scalar_string_coerced_to_empty_no_raise(self, tmp_path):
        """``dependencies: "not-a-list"`` → ``[]``, no exception."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                dependencies: "not-a-list"
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].dependencies == []

    def test_dependencies_dict_coerced_to_empty_no_raise(self, tmp_path):
        """``dependencies: {foo: bar}`` → ``[]``, no exception."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                dependencies:
                  foo: bar
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].dependencies == []

    def test_dependencies_integer_coerced_to_empty_no_raise(self, tmp_path):
        """``dependencies: 42`` → ``[]``, no exception."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                dependencies: 42
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].dependencies == []

    def test_dependencies_bad_type_does_not_raise_meta_parse_error(self, tmp_path):
        """Bad ``dependencies`` must not raise ``MetaParseError`` — it is
        coerced tolerantly."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                dependencies: "should be list"
        """)
        # Must not raise MetaParseError (or any error)
        catalog, _ = _load(tmp_path, "demo")
        assert isinstance(catalog["eng"], EngineImpl)

    # --- post_install: wrong top-level type ---

    def test_post_install_scalar_string_coerced_to_empty_no_raise(self, tmp_path):
        """``post_install: "not-a-list"`` → ``[]``, no exception."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install: "not-a-list"
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].post_install == []

    def test_post_install_integer_coerced_to_empty_no_raise(self, tmp_path):
        """``post_install: 99`` → ``[]``, no exception."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install: 99
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].post_install == []

    def test_post_install_bad_type_does_not_raise_meta_parse_error(self, tmp_path):
        """Bad ``post_install`` must not raise ``MetaParseError``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install: "bad"
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert isinstance(catalog["eng"], EngineImpl)

    # --- post_install: invalid entries inside the list ---

    def test_post_install_entry_missing_args_is_skipped(self, tmp_path):
        """Entry with no ``args`` key → skipped, result ``[]``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
                  - description: no args here
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].post_install == []

    def test_post_install_entry_empty_args_list_is_skipped(self, tmp_path):
        """Entry with ``args: []`` (empty) → skipped, result ``[]``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
                  - args: []
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].post_install == []

    def test_post_install_entry_args_has_empty_string_element_is_skipped(self, tmp_path):
        """Entry with ``args: ["", "bar"]`` — empty string element → entire
        entry skipped."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
                  - args:
                      - ""
                      - bar
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].post_install == []

    def test_post_install_entry_args_has_non_string_element_is_skipped(self, tmp_path):
        """Entry with ``args: [123]`` — non-string element → entire entry
        skipped."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
                  - args:
                      - 123
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].post_install == []

    def test_post_install_entry_args_null_is_skipped(self, tmp_path):
        """Entry with ``args: null`` → skipped, result ``[]``."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
                  - args:
        """)  # bare key → null
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].post_install == []

    def test_post_install_entry_args_is_a_string_not_list_is_skipped(self, tmp_path):
        """Entry with ``args: "scalar"`` (string, not list) → skipped."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
                  - args: "scalar-string"
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].post_install == []

    def test_post_install_scalar_entry_in_list_is_skipped(self, tmp_path):
        """A non-dict entry inside the post_install list (e.g. ``"scalar"``)
        is skipped."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
                  - scalar
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].post_install == []

    def test_post_install_integer_entry_in_list_is_skipped(self, tmp_path):
        """An integer entry inside the post_install list is skipped."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
                  - 42
        """)
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].post_install == []

    def test_post_install_mixed_valid_and_invalid_entries(self, tmp_path):
        """Mix of valid and invalid entries: only valid ones are kept."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
                  - args:
                      - good-cmd
                    description: valid
                  - description: no args, skipped
                  - args: []
                  - args:
                      - valid-second-cmd
        """)
        catalog, _ = _load(tmp_path, "demo")
        result = catalog["eng"].post_install
        assert len(result) == 2
        assert result[0]["args"] == ["good-cmd"]
        assert result[1]["args"] == ["valid-second-cmd"]

    def test_post_install_skips_do_not_raise_meta_parse_error(self, tmp_path):
        """Entry skips due to bad args must NEVER raise MetaParseError
        — coercion is permissive (warn-and-skip)."""
        _write_meta(tmp_path, "demo", """
            builtin_engines:
              - name: eng
                factory_module: pkg
                factory_attr: Cls
                default: true
                post_install:
                  - description: missing args
                  - args: []
                  - args:
                      - ""
        """)
        # Must not raise
        catalog, _ = _load(tmp_path, "demo")
        assert catalog["eng"].post_install == []


# ---------------------------------------------------------------------------
# ENGINE_IMPL BACKWARD-COMPATIBILITY REGRESSION
# ---------------------------------------------------------------------------


class TestEngineImplBackwardCompat:
    """Regression: EngineImpl must still be constructable without the new
    fields (the registry fallback path constructs without these kwargs)."""

    def test_engine_impl_default_dependencies_is_empty_list(self):
        """EngineImpl constructed without ``dependencies`` kwarg must have
        ``dependencies == []``."""
        impl = EngineImpl(cls=None, description="test", config_schema=[])
        assert hasattr(impl, "dependencies")
        assert impl.dependencies == []

    def test_engine_impl_default_post_install_is_empty_list(self):
        """EngineImpl constructed without ``post_install`` kwarg must have
        ``post_install == []``."""
        impl = EngineImpl(cls=None, description="test", config_schema=[])
        assert hasattr(impl, "post_install")
        assert impl.post_install == []

    def test_engine_impl_default_fields_are_independent_per_instance(self):
        """Two EngineImpl instances created with default fields must each
        have independent list objects (dataclass field default_factory)."""
        impl_a = EngineImpl(cls=None, description="a", config_schema=[])
        impl_b = EngineImpl(cls=None, description="b", config_schema=[])
        assert impl_a.dependencies is not impl_b.dependencies, (
            "default dependencies lists must not be shared (use field(default_factory=list))"
        )
        assert impl_a.post_install is not impl_b.post_install, (
            "default post_install lists must not be shared (use field(default_factory=list))"
        )

    def test_engine_impl_accepts_dependencies_kwarg(self):
        """EngineImpl accepts ``dependencies`` as a constructor kwarg."""
        impl = EngineImpl(
            cls=None, description="test", config_schema=[],
            dependencies=["dep-a", "dep-b"]
        )
        assert impl.dependencies == ["dep-a", "dep-b"]

    def test_engine_impl_accepts_post_install_kwarg(self):
        """EngineImpl accepts ``post_install`` as a constructor kwarg."""
        entry = {"args": ["cmd"], "description": "", "optional": False}
        impl = EngineImpl(
            cls=None, description="test", config_schema=[],
            post_install=[entry]
        )
        assert impl.post_install == [entry]


# ---------------------------------------------------------------------------
# _coerce_dependencies UNIT TESTS (if importable)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _HAVE_COERCE_DEPENDENCIES,
    reason="_coerce_dependencies not yet importable (red state expected)"
)
class TestCoerceDependencies:
    """Unit tests for the ``_coerce_dependencies`` helper (mirrors
    ``TestCoerceConfigSchema`` style)."""

    def test_none_yields_empty_list(self):
        assert _coerce_dependencies(None) == []  # noqa: F821

    def test_string_yields_empty_list(self):
        assert _coerce_dependencies("not-a-list") == []  # noqa: F821

    def test_dict_yields_empty_list(self):
        assert _coerce_dependencies({"foo": "bar"}) == []  # noqa: F821

    def test_integer_yields_empty_list(self):
        assert _coerce_dependencies(42) == []  # noqa: F821

    def test_bool_yields_empty_list(self):
        assert _coerce_dependencies(True) == []  # noqa: F821

    def test_empty_list_yields_empty_list(self):
        assert _coerce_dependencies([]) == []  # noqa: F821

    def test_valid_strings_pass_through(self):
        raw = ["foo>=1.0", "bar", "baz!=2.0"]
        assert _coerce_dependencies(raw) == ["foo>=1.0", "bar", "baz!=2.0"]  # noqa: F821

    def test_whitespace_stripped(self):
        assert _coerce_dependencies(["  stripped  "]) == ["stripped"]  # noqa: F821

    def test_empty_string_filtered(self):
        assert _coerce_dependencies([""]) == []  # noqa: F821

    def test_whitespace_only_filtered(self):
        assert _coerce_dependencies(["   "]) == []  # noqa: F821

    def test_non_string_items_dropped(self):
        assert _coerce_dependencies([42, None, True, "ok"]) == ["ok"]  # noqa: F821

    def test_mixed_complex_coercion(self):
        """The pivotal mixed-input test."""
        result = _coerce_dependencies(["  foo>=1.0  ", "", None, 42, "bar"])  # noqa: F821
        assert result == ["foo>=1.0", "bar"]

    def test_output_is_new_list_not_input(self):
        """Result is a new list object, not the same reference."""
        raw = ["dep-a"]
        result = _coerce_dependencies(raw)  # noqa: F821
        assert result is not raw


# ---------------------------------------------------------------------------
# _coerce_post_install UNIT TESTS (if importable)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _HAVE_COERCE_POST_INSTALL,
    reason="_coerce_post_install not yet importable (red state expected)"
)
class TestCoercePostInstall:
    """Unit tests for the ``_coerce_post_install`` helper."""

    def test_none_yields_empty_list(self):
        assert _coerce_post_install(None) == []  # noqa: F821

    def test_string_yields_empty_list(self):
        assert _coerce_post_install("not-a-list") == []  # noqa: F821

    def test_integer_yields_empty_list(self):
        assert _coerce_post_install(99) == []  # noqa: F821

    def test_empty_list_yields_empty_list(self):
        assert _coerce_post_install([]) == []  # noqa: F821

    def test_valid_entry_full_fields_normalized(self):
        raw = [{"args": ["cmd", "arg"], "description": "desc", "optional": True}]
        result = _coerce_post_install(raw)  # noqa: F821
        assert result == [{"args": ["cmd", "arg"], "description": "desc", "optional": True}]

    def test_valid_entry_only_args_defaults_applied(self):
        result = _coerce_post_install([{"args": ["cmd"]}])  # noqa: F821
        assert result == [{"args": ["cmd"], "description": "", "optional": False}]

    def test_entry_missing_args_is_skipped(self):
        assert _coerce_post_install([{"description": "no args"}]) == []  # noqa: F821

    def test_entry_empty_args_list_is_skipped(self):
        assert _coerce_post_install([{"args": []}]) == []  # noqa: F821

    def test_entry_args_with_empty_string_is_skipped(self):
        assert _coerce_post_install([{"args": ["", "bar"]}]) == []  # noqa: F821

    def test_entry_args_with_non_string_is_skipped(self):
        assert _coerce_post_install([{"args": [123]}]) == []  # noqa: F821

    def test_entry_args_is_null_is_skipped(self):
        assert _coerce_post_install([{"args": None}]) == []  # noqa: F821

    def test_entry_args_is_string_not_list_is_skipped(self):
        assert _coerce_post_install([{"args": "scalar"}]) == []  # noqa: F821

    def test_scalar_entry_in_list_is_skipped(self):
        assert _coerce_post_install(["scalar"]) == []  # noqa: F821

    def test_integer_entry_in_list_is_skipped(self):
        assert _coerce_post_install([42]) == []  # noqa: F821

    def test_mixed_valid_and_invalid_keeps_valid(self):
        raw = [
            {"args": ["good-cmd"]},
            {"description": "no args"},
            {"args": []},
            {"args": ["also-good"]},
        ]
        result = _coerce_post_install(raw)  # noqa: F821
        assert len(result) == 2
        assert result[0]["args"] == ["good-cmd"]
        assert result[1]["args"] == ["also-good"]

    def test_output_entry_has_exactly_three_keys(self):
        result = _coerce_post_install([{"args": ["cmd"]}])  # noqa: F821
        assert set(result[0].keys()) == {"args", "description", "optional"}

    def test_output_is_new_list_not_input(self):
        raw = [{"args": ["cmd"]}]
        result = _coerce_post_install(raw)  # noqa: F821
        assert result is not raw
