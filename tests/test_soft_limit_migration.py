"""Edge tests: gm_node_soft_limit ownership migration.

Contract authority: contracts/memory-access/definition.md,
section "gm_node_soft_limit ownership (moved out of config.fatigue)".

These tests target the FINAL state after the migration and are expected
to be RED until the implementation lands. They do NOT modify any
existing test files.

Coverage
--------
A  FatigueSection field removal (config.fatigue no longer has gm_node_soft_limit)
B  Runtime.memory_soft_limit() hook — exists, returns int, duck-types engine,
   defaults to 1000 when absent, returns a present 0 as 0 (not defaulted)
C  Heartbeat fatigue calc reads the hook (integration, conditional)
D  CLI /status reads the hook (integration, conditional)

Async plumbing: asyncio_mode = auto (see pytest.ini).
"""
from __future__ import annotations

import textwrap

import pytest

from krakey.models.config import Config, FatigueSection, load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path, body: str):
    """Write body to tmp_path/config.yaml and return the path."""
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def _minimal_config_yaml(extra_fatigue_fields: str = "") -> str:
    """Return the smallest syntactically valid config.yaml body.

    Matches the pattern used by test_config.py's _minimal_config_body
    helper but also supplies a minimal `fatigue:` section with the
    surviving fields so the loader doesn't fall back to hard defaults
    in ways that might hide the field-removal.
    """
    return f"""
        llm:
          providers: {{}}
          tags: {{}}
          core_purposes: {{}}
        graph_memory:
          db_path: ":memory:"
        fatigue:
          force_sleep_threshold: 120
          thresholds: {{}}
          {extra_fatigue_fields}
    """


# ===========================================================================
# A. FatigueSection field removal
# ===========================================================================


class TestFatigueSectionFieldRemoval:
    """After the migration FatigueSection must NOT carry gm_node_soft_limit.

    These tests are all RED until the field is removed from
    krakey/models/config/heartbeat.py and its parser (_build_fatigue).
    """

    # --- positive: surviving fields still exist ---

    def test_fatigue_section_has_force_sleep_threshold(self):
        """force_sleep_threshold survives the migration — still present."""
        f = FatigueSection()
        assert hasattr(f, "force_sleep_threshold"), (
            "FatigueSection.force_sleep_threshold must still exist"
        )

    def test_fatigue_section_has_thresholds(self):
        """thresholds dict survives the migration — still present."""
        f = FatigueSection()
        assert hasattr(f, "thresholds"), (
            "FatigueSection.thresholds must still exist"
        )

    def test_fatigue_section_force_sleep_threshold_default_is_int(self):
        """force_sleep_threshold default value is an int."""
        f = FatigueSection()
        assert isinstance(f.force_sleep_threshold, int)

    def test_fatigue_section_thresholds_default_is_dict(self):
        """thresholds default value is a dict."""
        f = FatigueSection()
        assert isinstance(f.thresholds, dict)

    # --- negative: gm_node_soft_limit is absent from FatigueSection ---

    def test_fatigue_section_default_instance_lacks_gm_node_soft_limit(self):
        """FatigueSection() must NOT have a gm_node_soft_limit attribute.

        RED until the field is removed from the dataclass definition.
        """
        f = FatigueSection()
        assert not hasattr(f, "gm_node_soft_limit"), (
            "gm_node_soft_limit must be removed from FatigueSection; "
            "it now belongs to the memory engine's settings"
        )

    def test_config_fatigue_lacks_gm_node_soft_limit(self):
        """Config().fatigue must NOT expose gm_node_soft_limit.

        RED until the field is removed.
        """
        cfg = Config()
        assert not hasattr(cfg.fatigue, "gm_node_soft_limit"), (
            "cfg.fatigue.gm_node_soft_limit must be gone; "
            "use runtime.memory_soft_limit() instead"
        )

    # --- state: YAML key is silently ignored, not parsed into FatigueSection ---

    def test_load_config_with_fatigue_gm_node_soft_limit_does_not_raise(
        self, tmp_path
    ):
        """A YAML with fatigue: {gm_node_soft_limit: 555, ...} must NOT raise.

        Old config files that still carry this key should load cleanly
        (the key is simply ignored).
        RED until _build_fatigue is updated to not consume the key.
        """
        p = _write_yaml(tmp_path, _minimal_config_yaml(
            "gm_node_soft_limit: 555"
        ))
        # Must not raise
        cfg = load_config(p)
        assert cfg is not None

    def test_load_config_with_fatigue_gm_node_soft_limit_ignores_key(
        self, tmp_path
    ):
        """Loading a YAML with fatigue.gm_node_soft_limit does NOT create
        the attribute on the resulting FatigueSection.

        RED until the field is removed AND _build_fatigue no longer reads it.
        """
        p = _write_yaml(tmp_path, _minimal_config_yaml(
            "gm_node_soft_limit: 555"
        ))
        cfg = load_config(p)
        assert not hasattr(cfg.fatigue, "gm_node_soft_limit"), (
            "gm_node_soft_limit=555 in YAML must be ignored; "
            "it must not appear as an attribute on cfg.fatigue"
        )

    def test_load_config_fatigue_surviving_fields_intact_after_ignored_key(
        self, tmp_path
    ):
        """The surviving FatigueSection fields load correctly even when the
        old gm_node_soft_limit key is present in the YAML.
        """
        p = _write_yaml(tmp_path, """
            llm:
              providers: {}
              tags: {}
              core_purposes: {}
            graph_memory:
              db_path: ":memory:"
            fatigue:
              gm_node_soft_limit: 555
              force_sleep_threshold: 99
              thresholds: {}
        """)
        cfg = load_config(p)
        assert cfg.fatigue.force_sleep_threshold == 99, (
            "force_sleep_threshold=99 must load correctly alongside "
            "the ignored gm_node_soft_limit key"
        )

    def test_load_config_fatigue_thresholds_intact_after_ignored_key(
        self, tmp_path
    ):
        """thresholds dict loads correctly even when gm_node_soft_limit is
        present in the YAML alongside it.
        """
        p = _write_yaml(tmp_path, """
            llm:
              providers: {}
              tags: {}
              core_purposes: {}
            graph_memory:
              db_path: ":memory:"
            fatigue:
              gm_node_soft_limit: 555
              force_sleep_threshold: 200
              thresholds:
                50: "(may sleep)"
                75: "(fatigued)"
        """)
        cfg = load_config(p)
        assert 50 in cfg.fatigue.thresholds
        assert 75 in cfg.fatigue.thresholds


# ===========================================================================
# B. Runtime.memory_soft_limit() hook
# ===========================================================================


class TestRuntimeMemorySoftLimitHook:
    """Runtime.memory_soft_limit() -> int

    Contract: duck-type engine.gm_node_soft_limit; default 1000 when absent.

    How the Runtime is obtained
    ---------------------------
    Tests in this class use ``build_runtime_with_fakes`` from
    tests/_runtime_helpers.py, which is the canonical test helper for
    all tests that need a Runtime. The helper constructs a Runtime with a
    real GraphMemoryEngine (in-memory SQLite) whose default
    gm_node_soft_limit is 1000. Some tests monkeypatch the attribute
    directly on runtime.memory to test the duck-typing contract without
    having to touch the engine's constructor.

    Assumption: ``build_runtime_with_fakes`` can be called with no
    arguments other than ``self_llm=ScriptedLLM()``. This matches the
    existing pattern in _runtime_helpers.py. The helper is expected to
    provide a fully-constructed Runtime including a memory engine at
    runtime.memory.
    """

    # --- positive: method exists and returns int ---

    def test_memory_soft_limit_method_exists_on_runtime(self):
        """Runtime has a memory_soft_limit method after the migration.

        RED until Runtime.memory_soft_limit() is added.
        """
        from krakey.main import Runtime
        assert hasattr(Runtime, "memory_soft_limit"), (
            "Runtime must expose a memory_soft_limit() method"
        )
        assert callable(Runtime.memory_soft_limit)

    def test_memory_soft_limit_is_synchronous(self):
        """memory_soft_limit() is a synchronous method per the contract.

        The contract specifies (Synchronous method) — it must NOT be a
        coroutine function.
        RED until the method is added.
        """
        import inspect
        from krakey.main import Runtime
        assert not inspect.iscoroutinefunction(Runtime.memory_soft_limit), (
            "Runtime.memory_soft_limit must be synchronous (not async)"
        )

    def test_memory_soft_limit_returns_int_for_default_runtime(self):
        """memory_soft_limit() returns an int for a default-built runtime.

        RED until the hook is implemented on Runtime and until the default
        GraphMemoryEngine exposes gm_node_soft_limit.
        """
        from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes
        rt = build_runtime_with_fakes(self_llm=ScriptedLLM())
        result = rt.memory_soft_limit()
        assert isinstance(result, int), (
            f"memory_soft_limit() must return int, got {type(result)}"
        )

    def test_memory_soft_limit_default_runtime_returns_1000(self):
        """A default-built Runtime (GraphMemoryEngine, no custom config)
        returns 1000 from memory_soft_limit().

        The contract says the memory engine's default gm_node_soft_limit
        is 1000. The hook returns that value.
        RED until both the engine attribute and the hook are in place.
        """
        from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes
        rt = build_runtime_with_fakes(self_llm=ScriptedLLM())
        assert rt.memory_soft_limit() == 1000

    # --- positive: hook reads from engine attribute (duck-typing) ---

    def test_memory_soft_limit_reads_engine_gm_node_soft_limit(self):
        """When runtime.memory.gm_node_soft_limit = 350 the hook returns 350.

        Verifies the duck-typing contract: the hook reads the engine's
        current attribute value.
        RED until the hook is implemented.
        """
        from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes
        rt = build_runtime_with_fakes(self_llm=ScriptedLLM())
        rt.memory.gm_node_soft_limit = 350
        assert rt.memory_soft_limit() == 350

    def test_memory_soft_limit_reflects_engine_value_change(self):
        """Changing engine.gm_node_soft_limit after construction is reflected
        by subsequent calls to memory_soft_limit().

        The hook reads the attribute at call-time, not at construction.
        RED until the hook is implemented.
        """
        from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes
        rt = build_runtime_with_fakes(self_llm=ScriptedLLM())
        rt.memory.gm_node_soft_limit = 777
        assert rt.memory_soft_limit() == 777
        rt.memory.gm_node_soft_limit = 500
        assert rt.memory_soft_limit() == 500

    # --- BVA: default when attribute is absent ---

    def test_memory_soft_limit_defaults_to_1000_when_attr_absent(self):
        """When memory has NO gm_node_soft_limit attribute the hook returns 1000.

        The contract: getattr(self.memory, "gm_node_soft_limit", 1000).
        RED until the hook is implemented.
        """
        from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes
        rt = build_runtime_with_fakes(self_llm=ScriptedLLM())
        # Remove the attribute from the engine so it's absent
        if hasattr(rt.memory, "gm_node_soft_limit"):
            del rt.memory.gm_node_soft_limit
        assert rt.memory_soft_limit() == 1000

    def test_memory_soft_limit_default_is_1000_with_bare_memory_object(self):
        """A memory object with no gm_node_soft_limit attribute at all
        causes the hook to return the sane default of 1000.

        Uses a minimal object rather than a real Runtime to test the
        duck-typing logic in isolation.
        RED until the hook is implemented.
        """
        from krakey.main import Runtime

        class _MinimalMemory:
            """Memory object with no gm_node_soft_limit."""
            pass

        class _FakeRuntime:
            memory = _MinimalMemory()

        # Bind the unbound method to the fake runtime to test the logic
        # without fully constructing a real Runtime.
        result = Runtime.memory_soft_limit(_FakeRuntime())  # type: ignore[arg-type]
        assert result == 1000

    def test_memory_soft_limit_reads_777_from_attr_on_bare_object(self):
        """A memory object with gm_node_soft_limit=777 causes the hook
        to return 777 — the duck-typing read path.
        RED until the hook is implemented.
        """
        from krakey.main import Runtime

        class _MinimalMemory:
            gm_node_soft_limit = 777

        class _FakeRuntime:
            memory = _MinimalMemory()

        result = Runtime.memory_soft_limit(_FakeRuntime())  # type: ignore[arg-type]
        assert result == 777

    # --- BVA: zero is a real value, NOT defaulted to 1000 ---

    def test_memory_soft_limit_present_zero_returns_zero_not_default(self):
        """When engine.gm_node_soft_limit = 0 the hook returns 0, NOT 1000.

        The default (1000) applies ONLY when the attribute is absent.
        A present-but-zero value is a real value that must pass through.
        This pins the exact getattr(obj, attr, default) semantics.
        RED until the hook is implemented.
        """
        from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes
        rt = build_runtime_with_fakes(self_llm=ScriptedLLM())
        rt.memory.gm_node_soft_limit = 0
        result = rt.memory_soft_limit()
        assert result == 0, (
            f"gm_node_soft_limit=0 (present) must return 0, not {result}. "
            "The default 1000 applies only when the attribute is absent."
        )

    def test_memory_soft_limit_present_zero_on_bare_object_returns_zero(self):
        """Same zero-is-real-value assertion on the bare-object path.
        RED until the hook is implemented.
        """
        from krakey.main import Runtime

        class _MinimalMemory:
            gm_node_soft_limit = 0

        class _FakeRuntime:
            memory = _MinimalMemory()

        result = Runtime.memory_soft_limit(_FakeRuntime())  # type: ignore[arg-type]
        assert result == 0, (
            f"gm_node_soft_limit=0 (present) must return 0, not {result}."
        )

    # --- BVA: large value passes through ---

    def test_memory_soft_limit_large_value_passes_through(self):
        """A large gm_node_soft_limit (e.g. 999_999) passes through unchanged.
        RED until the hook is implemented.
        """
        from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes
        rt = build_runtime_with_fakes(self_llm=ScriptedLLM())
        rt.memory.gm_node_soft_limit = 999_999
        assert rt.memory_soft_limit() == 999_999

    # --- negative: no AttributeError when engine has no such attr ---

    def test_memory_soft_limit_never_raises_attribute_error(self):
        """memory_soft_limit() must never raise AttributeError regardless
        of whether the engine exposes gm_node_soft_limit.
        RED until the hook is implemented.
        """
        from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes
        rt = build_runtime_with_fakes(self_llm=ScriptedLLM())
        if hasattr(rt.memory, "gm_node_soft_limit"):
            del rt.memory.gm_node_soft_limit
        try:
            rt.memory_soft_limit()
        except AttributeError as exc:
            pytest.fail(
                f"memory_soft_limit() raised AttributeError: {exc}. "
                "It must use getattr with a default, not direct attribute access."
            )

    # --- state: the hook is read-only, not a setter ---

    def test_memory_soft_limit_return_type_is_always_int(self):
        """The hook returns an int regardless of the engine attribute type.

        The contract specifies -> int. If the engine stores the value as
        a float or string, the hook must cast to int.
        RED until the hook is implemented (the cast is int(getattr(...))).
        """
        from krakey.main import Runtime

        class _FloatMemory:
            gm_node_soft_limit = 1500.9   # stored as float

        class _FakeRuntime:
            memory = _FloatMemory()

        result = Runtime.memory_soft_limit(_FakeRuntime())  # type: ignore[arg-type]
        assert isinstance(result, int), (
            f"memory_soft_limit() must return int, got {type(result)}"
        )
        # int(1500.9) == 1500
        assert result == 1500


# ===========================================================================
# C. Heartbeat fatigue calc reads memory_soft_limit() (integration)
# ===========================================================================


class TestHeartbeatFatigueUsesHook:
    """_phase_compute_fatigue must use rt.memory_soft_limit() instead of
    rt.config.fatigue.gm_node_soft_limit.

    Integration tests: if the test helper makes it too coupled to drive a
    full heartbeat beat, the test is skipped and the gap is noted.

    SKIP condition: runtime.config.fatigue no longer has gm_node_soft_limit,
    so these tests are only meaningful once A + B are also in place.
    """

    async def test_fatigue_pct_reflects_memory_soft_limit(self):
        """When memory_soft_limit() returns a known value and the GM node
        count is known, _phase_compute_fatigue must produce the matching
        fatigue %.

        Approach: build a runtime, set engine.gm_node_soft_limit to a
        value (e.g. 200), seed GM with a known node count (e.g. 100),
        call _phase_compute_fatigue, assert fatigue_pct == 50.

        RED until the orchestrator reads rt.memory_soft_limit() instead of
        rt.config.fatigue.gm_node_soft_limit.

        NOTE: If this test proves too coupled (e.g. relies on internal
        orchestrator attributes not stable across refactors), it may be
        promoted to a coverage-gap note rather than a mandatory RED test.
        """
        pytest.importorskip(
            "krakey.engines.heartbeat.orchestrator",
            reason="heartbeat orchestrator not available",
        )
        from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes
        rt = build_runtime_with_fakes(self_llm=ScriptedLLM())
        await rt.memory.initialize()

        # Set the memory engine's soft limit to 200
        rt.memory.gm_node_soft_limit = 200

        # Seed the GM with 100 nodes so fatigue should be 50 %
        for i in range(100):
            await rt.memory.ingest(f"node-{i}")

        node_count = await rt.memory.count_nodes()
        # Skip if the ingest didn't produce the expected count (InMemoryFake
        # dedup behaviour, or different engine), to avoid false failures.
        if node_count < 50:
            pytest.skip(
                f"only {node_count} nodes after 100 ingests; "
                "likely dedup in test engine; skipping integration assertion"
            )

        # Drive _phase_compute_fatigue directly
        result = await rt._orchestrator._phase_compute_fatigue()
        expected_pct = int(node_count / 200 * 100)
        assert result.fatigue_pct == expected_pct, (
            f"fatigue_pct={result.fatigue_pct} expected={expected_pct}. "
            "The heartbeat must read soft_limit from rt.memory_soft_limit()."
        )

    async def test_fatigue_calc_not_reading_config_fatigue_gm_node_soft_limit(
        self,
    ):
        """After the migration cfg.fatigue.gm_node_soft_limit is absent.
        _phase_compute_fatigue must not raise AttributeError when it runs.

        This test verifies the orchestrator doesn't crash on the missing
        config field after field removal.
        RED until the orchestrator is updated to use rt.memory_soft_limit().
        """
        pytest.importorskip(
            "krakey.engines.heartbeat.orchestrator",
            reason="heartbeat orchestrator not available",
        )
        from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes
        rt = build_runtime_with_fakes(self_llm=ScriptedLLM())
        await rt.memory.initialize()

        # Precondition: the field must indeed be gone from config.fatigue
        if hasattr(rt.config.fatigue, "gm_node_soft_limit"):
            pytest.skip(
                "config.fatigue still has gm_node_soft_limit — "
                "field removal (part A) not yet done; skipping this test"
            )

        # Must not raise AttributeError
        try:
            await rt._orchestrator._phase_compute_fatigue()
        except AttributeError as exc:
            pytest.fail(
                f"_phase_compute_fatigue raised AttributeError: {exc}. "
                "It must read rt.memory_soft_limit(), not config.fatigue.gm_node_soft_limit."
            )


# ===========================================================================
# D. CLI /status reads memory_soft_limit() (integration)
# ===========================================================================


class TestCliStatusUsesHook:
    """_format_status must read runtime.memory_soft_limit() instead of
    runtime.config.fatigue.gm_node_soft_limit.
    """

    async def test_format_status_does_not_raise_when_config_field_absent(
        self,
    ):
        """After field removal, _format_status must not raise AttributeError.

        Approach: build a runtime, confirm config.fatigue lacks
        gm_node_soft_limit (skip if field removal is not yet done), then
        call handle_command("status", runtime) and assert no AttributeError.

        RED until commands.py is updated to use runtime.memory_soft_limit().
        """
        from krakey.runtime.commands.commands import handle_command
        from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes

        rt = build_runtime_with_fakes(self_llm=ScriptedLLM())

        if hasattr(rt.config.fatigue, "gm_node_soft_limit"):
            pytest.skip(
                "config.fatigue still has gm_node_soft_limit — "
                "field removal (part A) not yet done; skip CLI test"
            )

        await rt.memory.initialize()
        try:
            result = await handle_command("status", rt)
        except AttributeError as exc:
            pytest.fail(
                f"handle_command('status') raised AttributeError: {exc}. "
                "commands.py must use runtime.memory_soft_limit() "
                "instead of runtime.config.fatigue.gm_node_soft_limit."
            )
        assert result is not None

    async def test_format_status_fatigue_pct_reflects_memory_soft_limit(self):
        """_format_status must compute fatigue% using memory_soft_limit().

        Approach: set runtime.memory.gm_node_soft_limit = 500, confirm
        the output contains a fatigue% that is computed relative to 500
        (not the old config field).

        RED until commands.py is updated.
        """
        from krakey.runtime.commands.commands import handle_command
        from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes

        rt = build_runtime_with_fakes(self_llm=ScriptedLLM())

        if hasattr(rt.config.fatigue, "gm_node_soft_limit"):
            pytest.skip(
                "config.fatigue still has gm_node_soft_limit — "
                "field removal (part A) not yet done; skip CLI test"
            )

        rt.memory.gm_node_soft_limit = 500
        # Zero nodes → 0 %
        await rt.memory.initialize()
        result = await handle_command("status", rt)
        assert "fatigue=0%" in result.output, (
            f"Expected fatigue=0% (0 nodes / 500 soft_limit) in status output, "
            f"got: {result.output!r}"
        )

    async def test_format_status_output_contains_fatigue_field(self):
        """_format_status output must still contain a 'fatigue=<N>%' token.

        Regression guard: the refactor must not accidentally drop the
        fatigue field from the /status output.
        RED until commands.py is updated (it will crash on the missing
        config field before this assertion is reached; fixing the crash
        lands both tests together).
        """
        from krakey.runtime.commands.commands import handle_command
        from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes

        rt = build_runtime_with_fakes(self_llm=ScriptedLLM())

        if hasattr(rt.config.fatigue, "gm_node_soft_limit"):
            pytest.skip(
                "config.fatigue still has gm_node_soft_limit — skip CLI test"
            )

        await rt.memory.initialize()
        result = await handle_command("status", rt)
        assert "fatigue=" in result.output, (
            f"'fatigue=' token must still appear in /status output; "
            f"got: {result.output!r}"
        )
