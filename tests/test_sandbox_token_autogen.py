"""Edge tests for Runtime._build_environment_router() Part B:
sandbox agent.token auto-generation.

Spec under test
---------------
When config.environments.sandbox is present with guest_os AND agent.url
set but agent.token is EMPTY:

  Success path  (runtime._config_path is a real writable file):
    - Generate a 64-char hex token via secrets.token_hex(32).
    - Set it on the in-memory sb.agent.token.
    - Write it back into config.yaml under
      environments.sandbox.agent.token via PyYAML round-trip.
    - Do NOT register the sandbox env this startup: the guest cannot
      yet hold the freshly-generated token, so preflight would only
      waste a timeout. Next startup, the token is on disk → normal
      register-and-preflight path runs. (env_names == ["local"])
    - Emit a stderr notice mentioning "token" instructing the user to
      provision the guest then restart.

  Graceful-skip path  (runtime._config_path is None, or write fails):
    - No crash.
    - Sandbox stays disabled (env_names == ["local"]).
    - Emit a stderr warning mentioning "token".

  Opt-in gate unchanged:
    - No environments.sandbox block → no generation, sandbox disabled.

  Idempotent:
    - agent.token already non-empty → no regeneration, no rewrite,
      sandbox registered with the existing token.

Test layout
-----------
  Positive / equivalence  — success path happy-path cases
  BVA / boundary          — missing block, partial block, None config_path
  State / persistence     — simulate next-startup re-read
  Negative / error guessing — write-failure graceful-skip
"""
from __future__ import annotations

import yaml
import pytest

from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes
from krakey.models.config import (
    SandboxEnvironmentConfig,
    SandboxAgentSection,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runtime():
    """Minimal runtime; _config_path intentionally stays None."""
    return build_runtime_with_fakes(
        self_llm=ScriptedLLM(), hypo_llm=ScriptedLLM(),
    )


def _write_minimal_sandbox_yaml(path):
    """Write an on-disk config.yaml containing the navigable structure
    that the write-back needs: environments.sandbox.agent.token.
    The value written is "" to simulate a first-run state."""
    path.write_text(
        yaml.safe_dump({"environments": {"sandbox": {"agent": {"token": ""}}}}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 1. Positive / equivalence — success path
# ---------------------------------------------------------------------------


class TestPositiveSuccessPath:
    """Full auto-gen: guest_os + url set, token empty, real writable file."""

    def test_sandbox_disabled_this_run_after_autogen(self, tmp_path):
        """After _build_environment_router() with an empty token and a
        real config_path, the sandbox is NOT registered this startup
        (token saved; restart required after the user provisions the
        guest VM)."""
        cfg_path = tmp_path / "config.yaml"
        _write_minimal_sandbox_yaml(cfg_path)

        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://10.0.2.10:8765", token=""),
        )
        rt._config_path = str(cfg_path)

        router = rt._build_environment_router()

        assert router.env_names() == ["local"]

    def test_inmemory_token_is_non_empty_after_autogen(self, tmp_path):
        """The in-memory sb.agent.token must be non-empty after a
        successful auto-generation — previously it was ''."""
        cfg_path = tmp_path / "config.yaml"
        _write_minimal_sandbox_yaml(cfg_path)

        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://10.0.2.10:8765", token=""),
        )
        rt._config_path = str(cfg_path)

        rt._build_environment_router()

        assert rt.config.environments.sandbox.agent.token != ""

    def test_ondisk_token_matches_inmemory_token(self, tmp_path):
        """The token written to config.yaml must be identical to the
        token that is set on the in-memory config object — no drift."""
        cfg_path = tmp_path / "config.yaml"
        _write_minimal_sandbox_yaml(cfg_path)

        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://10.0.2.10:8765", token=""),
        )
        rt._config_path = str(cfg_path)

        rt._build_environment_router()

        in_memory_token = rt.config.environments.sandbox.agent.token
        on_disk = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        on_disk_token = on_disk["environments"]["sandbox"]["agent"]["token"]

        assert on_disk_token == in_memory_token

    def test_generated_token_is_64_char_hex(self, tmp_path):
        """secrets.token_hex(32) produces exactly 64 lowercase hex chars.
        The generated token must satisfy length == 64 and be entirely
        composed of characters in [0-9a-f]."""
        cfg_path = tmp_path / "config.yaml"
        _write_minimal_sandbox_yaml(cfg_path)

        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://10.0.2.10:8765", token=""),
        )
        rt._config_path = str(cfg_path)

        rt._build_environment_router()

        token = rt.config.environments.sandbox.agent.token
        assert len(token) == 64
        assert all(c in "0123456789abcdef" for c in token)

    def test_autogen_emits_notice_to_stderr(self, tmp_path, capsys):
        """A successful auto-generation must emit a notice to stderr.
        The exact wording is not tested; only the presence of the word
        'token' (case-insensitive) is asserted."""
        cfg_path = tmp_path / "config.yaml"
        _write_minimal_sandbox_yaml(cfg_path)

        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://10.0.2.10:8765", token=""),
        )
        rt._config_path = str(cfg_path)

        rt._build_environment_router()

        err = capsys.readouterr().err.lower()
        assert "token" in err

    def test_idempotent_existing_token_unchanged(self, tmp_path):
        """If agent.token is already 'preexisting', calling
        _build_environment_router() must NOT replace it with a new
        token — idempotency invariant."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            yaml.safe_dump({
                "environments": {
                    "sandbox": {"agent": {"token": "preexisting"}}
                }
            }),
            encoding="utf-8",
        )

        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(
                url="http://10.0.2.10:8765", token="preexisting"
            ),
        )
        rt._config_path = str(cfg_path)

        rt._build_environment_router()

        assert rt.config.environments.sandbox.agent.token == "preexisting"

    def test_idempotent_sandbox_registered_with_existing_token(self, tmp_path):
        """With a pre-existing non-empty token, the sandbox env MUST
        still be registered (env_names includes 'sandbox')."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            yaml.safe_dump({
                "environments": {
                    "sandbox": {"agent": {"token": "preexisting"}}
                }
            }),
            encoding="utf-8",
        )

        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(
                url="http://10.0.2.10:8765", token="preexisting"
            ),
        )
        rt._config_path = str(cfg_path)

        router = rt._build_environment_router()

        assert "sandbox" in router.env_names()

    def test_idempotent_ondisk_token_not_replaced(self, tmp_path):
        """When agent.token is already 'preexisting', the on-disk
        config.yaml must NOT be rewritten with a new value."""
        cfg_path = tmp_path / "config.yaml"
        original_content = yaml.safe_dump({
            "environments": {
                "sandbox": {"agent": {"token": "preexisting"}}
            }
        })
        cfg_path.write_text(original_content, encoding="utf-8")
        original_mtime = cfg_path.stat().st_mtime

        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(
                url="http://10.0.2.10:8765", token="preexisting"
            ),
        )
        rt._config_path = str(cfg_path)

        rt._build_environment_router()

        # On-disk token must still be "preexisting"
        on_disk = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert on_disk["environments"]["sandbox"]["agent"]["token"] == "preexisting"


# ---------------------------------------------------------------------------
# 2. BVA / boundary — no block, partial block, no config_path
# ---------------------------------------------------------------------------


class TestBoundaryAndNegative:
    """Boundary cases: missing sandbox block, partial config, None path."""

    def test_no_sandbox_block_sandbox_disabled(self):
        """Default runtime with no environments.sandbox → env_names
        must be exactly ['local']; auto-gen must NOT be triggered."""
        rt = _make_runtime()
        # Confirm no sandbox block is present (default)
        assert rt.config.environments.sandbox is None

        router = rt._build_environment_router()

        assert router.env_names() == ["local"]

    def test_no_sandbox_block_no_token_generated(self):
        """Without an environments.sandbox block, config.environments.sandbox
        must remain None after the router is built — nothing is injected."""
        rt = _make_runtime()
        rt._build_environment_router()
        assert rt.config.environments.sandbox is None

    def test_partial_block_guest_os_only_sandbox_disabled(self, capsys):
        """Partial config: guest_os set but agent fields empty/defaulted
        (url has default, token is empty). This is the pre-Part-B disabled
        gate — partial still means disabled even after Part B."""
        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            # agent defaults: url="http://10.0.2.10:8765", token=""
        )
        # No config_path → graceful-skip path (also tests partial independently)
        router = rt._build_environment_router()

        assert router.env_names() == ["local"]

    def test_partial_block_no_url_no_generation(self, tmp_path, capsys):
        """Partial config: guest_os set but agent.url explicitly empty.
        Auto-gen must NOT fire — both guest_os AND url must be present."""
        cfg_path = tmp_path / "config.yaml"
        _write_minimal_sandbox_yaml(cfg_path)

        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="", token=""),
        )
        rt._config_path = str(cfg_path)

        router = rt._build_environment_router()

        assert router.env_names() == ["local"]
        # In-memory token stays empty — was not generated
        assert rt.config.environments.sandbox.agent.token == ""

    def test_partial_block_no_url_emits_warning(self, tmp_path, capsys):
        """Missing agent.url must produce a warning to stderr naming the
        missing / incomplete configuration."""
        cfg_path = tmp_path / "config.yaml"
        _write_minimal_sandbox_yaml(cfg_path)

        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="", token=""),
        )
        rt._config_path = str(cfg_path)

        rt._build_environment_router()

        err = capsys.readouterr().err.lower()
        # The warning must reference either "agent", "url", or "sandbox"
        assert "agent" in err or "url" in err or "sandbox" in err

    def test_config_path_none_sandbox_disabled(self, capsys):
        """When runtime._config_path is None and token is empty,
        the graceful-skip path must leave sandbox disabled."""
        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://10.0.2.10:8765", token=""),
        )
        # _config_path is already None from build_runtime_with_fakes

        router = rt._build_environment_router()

        assert router.env_names() == ["local"]

    def test_config_path_none_no_crash(self, capsys):
        """Graceful-skip must not raise any exception when
        runtime._config_path is None."""
        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://10.0.2.10:8765", token=""),
        )

        # Must not raise
        rt._build_environment_router()

    def test_config_path_none_warning_mentions_token(self, capsys):
        """The graceful-skip stderr warning must contain 'token' so that
        the existing test_defaulted_agent_url_with_empty_token_keeps_sandbox_disabled
        assertion ('token' in stderr) continues to hold under Part B."""
        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://10.0.2.10:8765", token=""),
        )

        rt._build_environment_router()

        err = capsys.readouterr().err.lower()
        assert "token" in err

    def test_config_path_none_inmemory_token_stays_empty(self):
        """On the graceful-skip path, the in-memory token must NOT be
        partially set — it must remain '' so no confusion arises."""
        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://10.0.2.10:8765", token=""),
        )

        rt._build_environment_router()

        assert rt.config.environments.sandbox.agent.token == ""

    def test_no_guest_os_sandbox_disabled(self, tmp_path):
        """Sandbox block present but guest_os is empty string → sandbox
        must be left disabled regardless of token state."""
        cfg_path = tmp_path / "config.yaml"
        _write_minimal_sandbox_yaml(cfg_path)

        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="",  # not set
            agent=SandboxAgentSection(url="http://10.0.2.10:8765", token=""),
        )
        rt._config_path = str(cfg_path)

        router = rt._build_environment_router()

        assert router.env_names() == ["local"]


# ---------------------------------------------------------------------------
# 3. State / persistence — simulate next-startup re-read
# ---------------------------------------------------------------------------


class TestStatePersistence:
    """Verify the generated token survives a restart simulation."""

    def test_next_startup_token_present_sandbox_registered(self, tmp_path):
        """After a successful auto-gen+write, simulate a next startup:
        1. Read the written config.yaml to obtain the persisted token.
        2. Build a fresh runtime with that token already set.
        3. Call _build_environment_router() — sandbox must be registered
           and the token must be UNCHANGED (no second generation)."""
        cfg_path = tmp_path / "config.yaml"
        _write_minimal_sandbox_yaml(cfg_path)

        # --- First startup (auto-gen) ---
        rt1 = _make_runtime()
        rt1.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://10.0.2.10:8765", token=""),
        )
        rt1._config_path = str(cfg_path)
        rt1._build_environment_router()
        written_token = rt1.config.environments.sandbox.agent.token

        # --- Second startup (read written token) ---
        on_disk = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        persisted_token = on_disk["environments"]["sandbox"]["agent"]["token"]
        assert persisted_token == written_token  # round-trip coherence

        rt2 = _make_runtime()
        rt2.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(
                url="http://10.0.2.10:8765", token=persisted_token
            ),
        )
        rt2._config_path = str(cfg_path)

        router2 = rt2._build_environment_router()

        assert "sandbox" in router2.env_names()
        assert rt2.config.environments.sandbox.agent.token == persisted_token

    def test_next_startup_no_second_write(self, tmp_path):
        """After the second startup (non-empty token), the config.yaml
        must NOT be overwritten with a different token — the file's
        sandbox.agent.token value stays exactly what was written on
        the first startup."""
        cfg_path = tmp_path / "config.yaml"
        _write_minimal_sandbox_yaml(cfg_path)

        # First startup
        rt1 = _make_runtime()
        rt1.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://10.0.2.10:8765", token=""),
        )
        rt1._config_path = str(cfg_path)
        rt1._build_environment_router()
        first_token = rt1.config.environments.sandbox.agent.token

        # Second startup
        rt2 = _make_runtime()
        rt2.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(
                url="http://10.0.2.10:8765", token=first_token
            ),
        )
        rt2._config_path = str(cfg_path)
        rt2._build_environment_router()

        on_disk_after = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        on_disk_token = on_disk_after["environments"]["sandbox"]["agent"]["token"]
        assert on_disk_token == first_token

    def test_generated_token_is_present_in_ondisk_file(self, tmp_path):
        """The token the impl writes to disk must be non-empty and
        accessible at environments.sandbox.agent.token in the YAML
        — ensuring the write-back path navigated the dict correctly."""
        cfg_path = tmp_path / "config.yaml"
        _write_minimal_sandbox_yaml(cfg_path)

        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://10.0.2.10:8765", token=""),
        )
        rt._config_path = str(cfg_path)
        rt._build_environment_router()

        on_disk = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        token = on_disk["environments"]["sandbox"]["agent"]["token"]
        assert isinstance(token, str) and token != ""


# ---------------------------------------------------------------------------
# 4. Negative / error guessing — write failure graceful-skip
# ---------------------------------------------------------------------------


class TestWriteFailureGracefulSkip:
    """If the write-back raises (e.g. path doesn't exist), no crash,
    sandbox disabled."""

    def test_nonexistent_dir_no_crash(self, tmp_path, capsys):
        """Writing to a path inside a non-existent directory raises
        FileNotFoundError. _build_environment_router() must catch it and
        return normally — no exception propagates to the caller."""
        bad_path = tmp_path / "nope" / "config.yaml"
        # Intentionally do NOT create tmp_path/nope/

        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://10.0.2.10:8765", token=""),
        )
        rt._config_path = str(bad_path)

        # Must not raise
        rt._build_environment_router()

    def test_nonexistent_dir_sandbox_disabled(self, tmp_path):
        """When the write fails because the parent directory does not
        exist, the sandbox must remain unregistered — env_names == ['local']."""
        bad_path = tmp_path / "nope" / "config.yaml"

        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://10.0.2.10:8765", token=""),
        )
        rt._config_path = str(bad_path)

        router = rt._build_environment_router()

        assert router.env_names() == ["local"]

    def test_nonexistent_dir_returns_router(self, tmp_path):
        """_build_environment_router() must return a router object even
        when the token write fails — callers must not receive None."""
        bad_path = tmp_path / "nope" / "config.yaml"

        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://10.0.2.10:8765", token=""),
        )
        rt._config_path = str(bad_path)

        router = rt._build_environment_router()

        assert router is not None

    def test_ondisk_file_missing_sandbox_agent_key_graceful_skip(self, tmp_path):
        """If the on-disk config.yaml exists but lacks the
        environments.sandbox.agent key, the write-back navigation
        fails. Must be treated as a write-failure → graceful-skip
        (sandbox disabled, no crash)."""
        cfg_path = tmp_path / "config.yaml"
        # On-disk YAML has environments block but NOT the sandbox.agent key
        cfg_path.write_text(
            yaml.safe_dump({"environments": {}}),
            encoding="utf-8",
        )

        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://10.0.2.10:8765", token=""),
        )
        rt._config_path = str(cfg_path)

        router = rt._build_environment_router()

        assert router.env_names() == ["local"]

    def test_write_failure_warning_emitted(self, tmp_path, capsys):
        """A write failure (non-existent directory) must produce a
        stderr warning — the user needs to know the token was not saved."""
        bad_path = tmp_path / "nope" / "config.yaml"

        rt = _make_runtime()
        rt.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://10.0.2.10:8765", token=""),
        )
        rt._config_path = str(bad_path)

        rt._build_environment_router()

        err = capsys.readouterr().err.lower()
        assert "token" in err

    def test_two_distinct_autogen_calls_produce_different_tokens(
        self, tmp_path
    ):
        """Two independent auto-generation runs (each with a fresh
        empty token + fresh config file) must produce two different
        tokens — collision probability is negligible for 64-char hex."""
        cfg_path_a = tmp_path / "config_a.yaml"
        cfg_path_b = tmp_path / "config_b.yaml"
        _write_minimal_sandbox_yaml(cfg_path_a)
        _write_minimal_sandbox_yaml(cfg_path_b)

        rt_a = _make_runtime()
        rt_a.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://10.0.2.10:8765", token=""),
        )
        rt_a._config_path = str(cfg_path_a)
        rt_a._build_environment_router()
        token_a = rt_a.config.environments.sandbox.agent.token

        rt_b = _make_runtime()
        rt_b.config.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://10.0.2.10:8765", token=""),
        )
        rt_b._config_path = str(cfg_path_b)
        rt_b._build_environment_router()
        token_b = rt_b.config.environments.sandbox.agent.token

        assert token_a != token_b
