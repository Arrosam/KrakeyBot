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
