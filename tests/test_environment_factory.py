"""Edge tests for ``krakey.environment.build_environment_router`` — the
composition factory that decouples runtime from the environment node.

These verify the factory reproduces the behavior that used to live in
``Runtime._build_environment_router`` / ``_write_sandbox_token``, now as
a standalone, self-contained, injectable function:

  * Local always registered; sandbox conditional on completeness.
  * Token auto-generation + persistence when guest_os + url set, token
    empty, config_path writable.
  * Graceful disable (never raises / never blocks) on partial config,
    write failure, missing config_path, or unknown provider.
  * Diagnostic status seeding (local→ok, sandbox→unconfigured).
  * ``log_warn`` injection + stdlib fallback.

Applies the four techniques: positive/equivalence, boundary, state
transition (token persisted vs not), negative (error guessing).
"""
from __future__ import annotations

import logging

import yaml
import pytest

from krakey.environment import build_environment_router
from krakey.environment.router import EnvironmentRouter
from krakey.models.config import (
    Config,
    SandboxEnvironmentConfig,
    SandboxAgentSection,
    LocalEnvironmentConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg() -> Config:
    """A bare Config with default (empty) environments — no sandbox."""
    return Config()


def _collect_warns():
    """Returns (sink, list) — a log_warn callable that appends to list."""
    msgs: list[str] = []
    return (lambda m: msgs.append(m)), msgs


def _write_minimal_sandbox_yaml(path):
    path.write_text(
        yaml.safe_dump({"environments": {"sandbox": {"agent": {"token": ""}}}}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 1. Positive / equivalence
# ---------------------------------------------------------------------------

class TestPositive:
    def test_returns_environment_router(self):
        router = build_environment_router(_cfg())
        assert isinstance(router, EnvironmentRouter)

    def test_local_always_registered(self):
        router = build_environment_router(_cfg())
        assert "local" in router.env_names()

    def test_no_sandbox_block_local_only(self):
        cfg = _cfg()
        cfg.environments.sandbox = None
        router = build_environment_router(cfg)
        assert router.env_names() == ["local"]

    def test_local_allow_list_flows_through(self):
        cfg = _cfg()
        cfg.environments.local = LocalEnvironmentConfig(
            allowed_plugins=["cli_exec"],
        )
        router = build_environment_router(cfg)
        # cli_exec is allow-listed for local -> resolves without raising
        env = router.for_plugin("cli_exec", "local")
        assert env is not None

    def test_fully_configured_sandbox_is_registered(self):
        cfg = _cfg()
        cfg.environments.sandbox = SandboxEnvironmentConfig(
            allowed_plugins=["cli_exec"],
            guest_os="linux",
            agent=SandboxAgentSection(
                url="http://10.0.2.10:8765", token="deadbeef",
            ),
        )
        router = build_environment_router(cfg)
        assert "sandbox" in router.env_names()

    def test_local_status_seeded_ok(self):
        router = build_environment_router(_cfg())
        assert router.env_status()["local"][0] == "ok"


# ---------------------------------------------------------------------------
# 2. Token auto-generation (state transition)
# ---------------------------------------------------------------------------

class TestTokenAutogen:
    def _cfg_empty_token(self):
        cfg = _cfg()
        cfg.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://10.0.2.10:8765", token=""),
        )
        return cfg

    def test_sandbox_not_registered_this_run_after_autogen(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        _write_minimal_sandbox_yaml(cfg_path)
        cfg = self._cfg_empty_token()
        router = build_environment_router(cfg, config_path=str(cfg_path))
        # Token freshly generated -> sandbox deferred to next startup.
        assert router.env_names() == ["local"]

    def test_inmemory_token_non_empty_after_autogen(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        _write_minimal_sandbox_yaml(cfg_path)
        cfg = self._cfg_empty_token()
        build_environment_router(cfg, config_path=str(cfg_path))
        assert cfg.environments.sandbox.agent.token != ""

    def test_ondisk_token_matches_inmemory(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        _write_minimal_sandbox_yaml(cfg_path)
        cfg = self._cfg_empty_token()
        build_environment_router(cfg, config_path=str(cfg_path))
        on_disk = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert (
            on_disk["environments"]["sandbox"]["agent"]["token"]
            == cfg.environments.sandbox.agent.token
        )

    def test_generated_token_is_64_char_hex(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        _write_minimal_sandbox_yaml(cfg_path)
        cfg = self._cfg_empty_token()
        build_environment_router(cfg, config_path=str(cfg_path))
        token = cfg.environments.sandbox.agent.token
        assert len(token) == 64
        assert all(c in "0123456789abcdef" for c in token)

    def test_existing_token_is_not_regenerated(self, tmp_path):
        """Idempotent: a non-empty token skips the whole autogen branch
        and the sandbox is registered immediately."""
        cfg_path = tmp_path / "config.yaml"
        _write_minimal_sandbox_yaml(cfg_path)
        cfg = self._cfg_empty_token()
        cfg.environments.sandbox.agent.token = "preset-token"
        router = build_environment_router(cfg, config_path=str(cfg_path))
        assert cfg.environments.sandbox.agent.token == "preset-token"
        assert "sandbox" in router.env_names()


# ---------------------------------------------------------------------------
# 3. Boundary / partial config
# ---------------------------------------------------------------------------

class TestBoundaryPartialConfig:
    def test_missing_guest_os_disables_sandbox(self):
        cfg = _cfg()
        cfg.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="",
            agent=SandboxAgentSection(url="http://x:8765", token="t"),
        )
        router = build_environment_router(cfg)
        assert "sandbox" not in router.env_names()
        assert router.env_status()["sandbox"][0] == "unconfigured"

    def test_missing_url_disables_sandbox(self):
        cfg = _cfg()
        cfg.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="", token="t"),
        )
        router = build_environment_router(cfg)
        assert "sandbox" not in router.env_names()
        assert router.env_status()["sandbox"][0] == "unconfigured"

    def test_empty_token_no_config_path_disables_sandbox(self):
        """Token empty + config_path None → cannot auto-gen → disabled,
        and a warning is emitted (never raises)."""
        cfg = _cfg()
        cfg.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://x:8765", token=""),
        )
        warn, msgs = _collect_warns()
        router = build_environment_router(cfg, config_path=None, log_warn=warn)
        assert "sandbox" not in router.env_names()
        assert any("token" in m.lower() for m in msgs)


# ---------------------------------------------------------------------------
# 4. Negative / error guessing
# ---------------------------------------------------------------------------

class TestNegative:
    def test_token_write_failure_is_non_fatal(self, tmp_path):
        """config_path points at a file whose parent structure makes the
        YAML round-trip raise (no environments key). Factory must catch,
        warn, and still return a local-only router — never raise."""
        cfg_path = tmp_path / "config.yaml"
        # Structurally invalid for the write-back (no 'environments' map):
        cfg_path.write_text(yaml.safe_dump({"unrelated": True}), encoding="utf-8")
        cfg = _cfg()
        cfg.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://x:8765", token=""),
        )
        warn, msgs = _collect_warns()
        router = build_environment_router(
            cfg, config_path=str(cfg_path), log_warn=warn,
        )
        assert router.env_names() == ["local"]
        assert any("token" in m.lower() for m in msgs)

    def test_log_warn_fallback_to_stdlib_does_not_raise(self, caplog):
        """When log_warn is None the factory must fall back to stdlib
        logging without raising."""
        cfg = _cfg()
        cfg.environments.sandbox = SandboxEnvironmentConfig(
            guest_os="linux",
            agent=SandboxAgentSection(url="http://x:8765", token=""),
        )
        with caplog.at_level(logging.WARNING):
            router = build_environment_router(cfg)  # no log_warn, no config_path
        assert router.env_names() == ["local"]
