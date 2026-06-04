"""Edge tests for MemoryWebSection + SleepSection.auto_sleep_node_threshold.

These tests are written against the contract definition ONLY, before any
implementation exists. They will be RED until the dev lands the change.

Contract:
  - MemoryWebSection(enabled=False, host="127.0.0.1", port=8766)
  - Config.memory_web: MemoryWebSection (default_factory)
  - YAML key: memory_web
  - SleepSection gains auto_sleep_node_threshold: int = 0
  - All sections tolerate missing keys (defaults) and ignore unknown keys.
  - Type coercion: bool(enabled), str(host), int(port)

Minimal-config helper: reused from test_config._minimal_config_body / _write
convention (llm.providers/tags/core_purposes + graph_memory.db_path is enough
for load_config to parse without the LLM bootstrap exit path).
"""

import textwrap

import pytest

from krakey.models.config import Config, SleepSection, load_config
from krakey.models.config import MemoryWebSection  # new symbol under test


# ---------------------------------------------------------------------------
# Shared helpers (mirror test_config.py convention)
# ---------------------------------------------------------------------------

def _write(tmp_path, body):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def _minimal_body(extra: str = "") -> str:
    """Minimal syntactically-valid YAML that load_config accepts."""
    return f"""
        llm:
          providers: {{}}
          tags: {{}}
          core_purposes: {{}}
        graph_memory:
          db_path: ":memory:"
{extra}
    """


# ===========================================================================
# MemoryWebSection
# ===========================================================================

class TestMemoryWebSectionDefaults:
    """Positive / equivalence — direct construction and Config() defaults."""

    def test_direct_construction_defaults(self):
        """MemoryWebSection() with no args has the three spec defaults."""
        s = MemoryWebSection()
        assert s.enabled is False
        assert s.host == "127.0.0.1"
        assert s.port == 8766

    def test_config_memory_web_attribute_exists(self):
        """Config() exposes a .memory_web attribute of the right type."""
        cfg = Config()
        assert isinstance(cfg.memory_web, MemoryWebSection)

    def test_config_memory_web_enabled_default(self):
        """Config().memory_web.enabled defaults to False."""
        assert Config().memory_web.enabled is False

    def test_config_memory_web_host_default(self):
        """Config().memory_web.host defaults to '127.0.0.1'."""
        assert Config().memory_web.host == "127.0.0.1"

    def test_config_memory_web_port_default(self):
        """Config().memory_web.port defaults to 8766."""
        assert Config().memory_web.port == 8766

    def test_direct_construction_explicit_values(self):
        """MemoryWebSection can be constructed with explicit values."""
        s = MemoryWebSection(enabled=True, host="0.0.0.0", port=9000)
        assert s.enabled is True
        assert s.host == "0.0.0.0"
        assert s.port == 9000


class TestMemoryWebSectionYamlRoundTrip:
    """Positive — YAML with all three memory_web fields present."""

    def test_full_memory_web_block_loaded(self, tmp_path):
        """memory_web: {enabled: true, host: "0.0.0.0", port: 9000} round-trips."""
        p = _write(tmp_path, _minimal_body(
            "        memory_web:\n"
            "          enabled: true\n"
            "          host: \"0.0.0.0\"\n"
            "          port: 9000\n"
        ))
        cfg = load_config(p)
        assert cfg.memory_web.enabled is True
        assert cfg.memory_web.host == "0.0.0.0"
        assert cfg.memory_web.port == 9000

    def test_enabled_false_explicit_yaml(self, tmp_path):
        """memory_web.enabled: false is loaded as False (not None)."""
        p = _write(tmp_path, _minimal_body(
            "        memory_web:\n"
            "          enabled: false\n"
        ))
        cfg = load_config(p)
        assert cfg.memory_web.enabled is False

    def test_port_as_integer_in_yaml(self, tmp_path):
        """Explicit integer port in YAML is loaded as int."""
        p = _write(tmp_path, _minimal_body(
            "        memory_web:\n"
            "          port: 8080\n"
        ))
        cfg = load_config(p)
        assert cfg.memory_web.port == 8080
        assert isinstance(cfg.memory_web.port, int)


class TestMemoryWebSectionBVA:
    """Boundary value analysis — port edges, host variants."""

    def test_port_min_valid(self, tmp_path):
        """Port value of 1 (minimum valid TCP port) is accepted."""
        p = _write(tmp_path, _minimal_body(
            "        memory_web:\n"
            "          port: 1\n"
        ))
        cfg = load_config(p)
        assert cfg.memory_web.port == 1

    def test_port_well_known_boundary(self, tmp_path):
        """Port 1024 (well-known/registered boundary) loads as int 1024."""
        p = _write(tmp_path, _minimal_body(
            "        memory_web:\n"
            "          port: 1024\n"
        ))
        cfg = load_config(p)
        assert cfg.memory_web.port == 1024

    def test_port_max_valid(self, tmp_path):
        """Port value 65535 (max TCP port) is accepted."""
        p = _write(tmp_path, _minimal_body(
            "        memory_web:\n"
            "          port: 65535\n"
        ))
        cfg = load_config(p)
        assert cfg.memory_web.port == 65535

    def test_port_dashboard_adjacent(self, tmp_path):
        """Port 8765 (one below dashboard default 8766) is accepted."""
        p = _write(tmp_path, _minimal_body(
            "        memory_web:\n"
            "          port: 8765\n"
        ))
        cfg = load_config(p)
        assert cfg.memory_web.port == 8765

    def test_host_empty_string(self, tmp_path):
        """Empty string host is stored as empty string (no crash)."""
        p = _write(tmp_path, _minimal_body(
            "        memory_web:\n"
            "          host: \"\"\n"
        ))
        cfg = load_config(p)
        assert cfg.memory_web.host == ""

    def test_host_single_char(self, tmp_path):
        """Single-character host string is accepted."""
        p = _write(tmp_path, _minimal_body(
            "        memory_web:\n"
            "          host: \"x\"\n"
        ))
        cfg = load_config(p)
        assert cfg.memory_web.host == "x"

    def test_host_ipv6_loopback(self, tmp_path):
        """IPv6 loopback string is stored verbatim."""
        p = _write(tmp_path, _minimal_body(
            "        memory_web:\n"
            "          host: \"::1\"\n"
        ))
        cfg = load_config(p)
        assert cfg.memory_web.host == "::1"


class TestMemoryWebSectionTolerance:
    """BVA / error-guessing — missing keys, partial blocks, unknown keys,
    type coercion."""

    def test_no_memory_web_key_uses_defaults(self, tmp_path):
        """YAML without any memory_web key yields the three defaults."""
        p = _write(tmp_path, _minimal_body(""))
        cfg = load_config(p)
        assert cfg.memory_web.enabled is False
        assert cfg.memory_web.host == "127.0.0.1"
        assert cfg.memory_web.port == 8766

    def test_partial_block_only_port(self, tmp_path):
        """memory_web with only port overridden leaves host+enabled as defaults."""
        p = _write(tmp_path, _minimal_body(
            "        memory_web:\n"
            "          port: 7000\n"
        ))
        cfg = load_config(p)
        assert cfg.memory_web.port == 7000
        assert cfg.memory_web.host == "127.0.0.1"
        assert cfg.memory_web.enabled is False

    def test_partial_block_only_enabled(self, tmp_path):
        """memory_web with only enabled overridden leaves host+port as defaults."""
        p = _write(tmp_path, _minimal_body(
            "        memory_web:\n"
            "          enabled: true\n"
        ))
        cfg = load_config(p)
        assert cfg.memory_web.enabled is True
        assert cfg.memory_web.host == "127.0.0.1"
        assert cfg.memory_web.port == 8766

    def test_partial_block_only_host(self, tmp_path):
        """memory_web with only host overridden leaves enabled+port as defaults."""
        p = _write(tmp_path, _minimal_body(
            "        memory_web:\n"
            "          host: \"192.168.1.1\"\n"
        ))
        cfg = load_config(p)
        assert cfg.memory_web.host == "192.168.1.1"
        assert cfg.memory_web.enabled is False
        assert cfg.memory_web.port == 8766

    def test_unknown_key_in_memory_web_ignored(self, tmp_path):
        """Extra/unknown key inside memory_web block does not raise."""
        p = _write(tmp_path, _minimal_body(
            "        memory_web:\n"
            "          enabled: true\n"
            "          port: 9001\n"
            "          bogus: 1\n"
            "          future_field: some_value\n"
        ))
        cfg = load_config(p)  # must not raise
        assert cfg.memory_web.enabled is True
        assert cfg.memory_web.port == 9001

    def test_port_as_string_coerced_to_int(self, tmp_path):
        """port given as a YAML string "9001" is coerced to int 9001.

        Assumption: the loader applies int() coercion per the contract spec
        ('types coerced ... int(port)'). This test asserts the coercion result
        equals 9001 regardless of whether coercion happens inside load_config
        or via the dataclass field coercer. If the loader does NOT coerce and
        instead rejects/raises, this test will reveal that the coercion contract
        is unimplemented.
        """
        p = _write(tmp_path, _minimal_body(
            "        memory_web:\n"
            "          port: \"9001\"\n"  # YAML string, not int
        ))
        cfg = load_config(p)
        assert int(cfg.memory_web.port) == 9001

    def test_enabled_truthy_string_coercion(self, tmp_path):
        """enabled: true (YAML bool) is loaded as Python True.

        The project uses literal YAML booleans (test_config.py convention).
        This test confirms the canonical true/false round-trip.
        """
        p = _write(tmp_path, _minimal_body(
            "        memory_web:\n"
            "          enabled: true\n"
        ))
        cfg = load_config(p)
        assert cfg.memory_web.enabled is True

    def test_enabled_false_yaml_bool(self, tmp_path):
        """enabled: false (YAML bool) is loaded as Python False."""
        p = _write(tmp_path, _minimal_body(
            "        memory_web:\n"
            "          enabled: false\n"
        ))
        cfg = load_config(p)
        assert cfg.memory_web.enabled is False


class TestMemoryWebSectionStateIndependence:
    """State transition — default_factory ensures separate instances per Config."""

    def test_two_configs_have_independent_memory_web_objects(self):
        """Two Config() instances must NOT share the same MemoryWebSection
        object — default_factory, not a shared mutable default."""
        cfg_a = Config()
        cfg_b = Config()
        assert cfg_a.memory_web is not cfg_b.memory_web

    def test_mutating_one_config_does_not_affect_another(self):
        """Mutating cfg_a.memory_web does not affect cfg_b.memory_web."""
        cfg_a = Config()
        cfg_b = Config()
        cfg_a.memory_web.port = 9999
        assert cfg_b.memory_web.port == 8766

    def test_two_memory_web_section_defaults_are_independent(self):
        """Two MemoryWebSection() instances are distinct objects."""
        s1 = MemoryWebSection()
        s2 = MemoryWebSection()
        assert s1 is not s2
        s1.host = "mutated"
        assert s2.host == "127.0.0.1"


# ===========================================================================
# SleepSection — auto_sleep_node_threshold
# ===========================================================================

class TestAutoSleepNodeThresholdDefaults:
    """Positive / equivalence — default value on Config and SleepSection."""

    def test_config_sleep_auto_sleep_node_threshold_default(self):
        """Config().sleep.auto_sleep_node_threshold defaults to 0."""
        assert Config().sleep.auto_sleep_node_threshold == 0

    def test_sleep_section_direct_construction_default(self):
        """SleepSection() direct construction has auto_sleep_node_threshold=0."""
        s = SleepSection()
        assert s.auto_sleep_node_threshold == 0

    def test_auto_sleep_threshold_zero_means_disabled(self):
        """The contract documents 0 as the disabled sentinel — value is 0."""
        cfg = Config()
        assert cfg.sleep.auto_sleep_node_threshold == 0


class TestAutoSleepNodeThresholdYamlRoundTrip:
    """Positive — YAML round-trips for the new field."""

    def test_threshold_loaded_from_yaml(self, tmp_path):
        """sleep.auto_sleep_node_threshold: 250 is loaded as int 250."""
        p = _write(tmp_path, _minimal_body(
            "        sleep:\n"
            "          auto_sleep_node_threshold: 250\n"
        ))
        cfg = load_config(p)
        assert cfg.sleep.auto_sleep_node_threshold == 250

    def test_threshold_zero_explicit_in_yaml(self, tmp_path):
        """Explicit 0 in YAML is loaded as 0 (disabled, not absent)."""
        p = _write(tmp_path, _minimal_body(
            "        sleep:\n"
            "          auto_sleep_node_threshold: 0\n"
        ))
        cfg = load_config(p)
        assert cfg.sleep.auto_sleep_node_threshold == 0

    def test_threshold_alongside_existing_sleep_fields(self, tmp_path):
        """New field co-exists with existing sleep fields without conflict."""
        p = _write(tmp_path, _minimal_body(
            "        sleep:\n"
            "          max_duration_seconds: 7200\n"
            "          auto_sleep_node_threshold: 100\n"
        ))
        cfg = load_config(p)
        assert cfg.sleep.auto_sleep_node_threshold == 100


class TestAutoSleepNodeThresholdBVA:
    """Boundary value analysis — 0, 1, large, negative."""

    def test_threshold_value_one(self, tmp_path):
        """Threshold of 1 (first non-disabled value) is loaded correctly."""
        p = _write(tmp_path, _minimal_body(
            "        sleep:\n"
            "          auto_sleep_node_threshold: 1\n"
        ))
        cfg = load_config(p)
        assert cfg.sleep.auto_sleep_node_threshold == 1

    def test_threshold_large_value(self, tmp_path):
        """Large threshold (10000) is accepted without error."""
        p = _write(tmp_path, _minimal_body(
            "        sleep:\n"
            "          auto_sleep_node_threshold: 10000\n"
        ))
        cfg = load_config(p)
        assert cfg.sleep.auto_sleep_node_threshold == 10000

    def test_threshold_typical_value(self, tmp_path):
        """Typical realistic threshold (500 nodes) loads correctly."""
        p = _write(tmp_path, _minimal_body(
            "        sleep:\n"
            "          auto_sleep_node_threshold: 500\n"
        ))
        cfg = load_config(p)
        assert cfg.sleep.auto_sleep_node_threshold == 500


class TestAutoSleepNodeThresholdTolerance:
    """BVA / error-guessing — missing field with sleep block present,
    type coercion, unknown keys."""

    def test_field_absent_with_sleep_block_present_defaults_to_zero(self, tmp_path):
        """sleep block present but auto_sleep_node_threshold absent → default 0."""
        p = _write(tmp_path, _minimal_body(
            "        sleep:\n"
            "          max_duration_seconds: 3600\n"
        ))
        cfg = load_config(p)
        assert cfg.sleep.auto_sleep_node_threshold == 0

    def test_no_sleep_block_at_all_defaults_to_zero(self, tmp_path):
        """YAML without any sleep key yields auto_sleep_node_threshold=0."""
        p = _write(tmp_path, _minimal_body(""))
        cfg = load_config(p)
        assert cfg.sleep.auto_sleep_node_threshold == 0

    def test_threshold_as_string_coerced_to_int(self, tmp_path):
        """auto_sleep_node_threshold given as string "300" is coerced to int.

        Assumption: loader applies int() coercion per the contract spec. This
        test reveals if coercion is missing.
        """
        p = _write(tmp_path, _minimal_body(
            "        sleep:\n"
            "          auto_sleep_node_threshold: \"300\"\n"
        ))
        cfg = load_config(p)
        assert int(cfg.sleep.auto_sleep_node_threshold) == 300

    def test_unknown_key_in_sleep_block_ignored(self, tmp_path):
        """Extra key inside sleep block does not raise (unknown-key tolerance)."""
        p = _write(tmp_path, _minimal_body(
            "        sleep:\n"
            "          auto_sleep_node_threshold: 50\n"
            "          future_autonomy_field: yes\n"
        ))
        cfg = load_config(p)  # must not raise
        assert cfg.sleep.auto_sleep_node_threshold == 50


# ===========================================================================
# Import contract — symbols are importable from the declared module path
# ===========================================================================

class TestImportContract:
    """Verify the contract's import surface is satisfied."""

    def test_memory_web_section_importable(self):
        """MemoryWebSection is importable from krakey.models.config."""
        # Import already at module level; this confirms it at test-run time.
        assert MemoryWebSection is not None

    def test_sleep_section_importable(self):
        """SleepSection is importable from krakey.models.config."""
        assert SleepSection is not None

    def test_config_importable(self):
        """Config is importable from krakey.models.config."""
        assert Config is not None

    def test_load_config_importable(self):
        """load_config is importable from krakey.models.config."""
        assert load_config is not None

    def test_all_symbols_in_single_import(self):
        """All four contracted symbols can be imported together."""
        from krakey.models.config import (  # noqa: F401
            Config,
            MemoryWebSection,
            SleepSection,
            load_config,
        )
