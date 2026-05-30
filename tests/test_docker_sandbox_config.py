"""Edge tests for the Phase D ``environments.sandbox.docker`` config
sub-block (DockerSandboxSection) and its parsing in
``_build_sandbox_env``.

Additive, non-contract change — a config that omits ``docker:`` must
get dataclass defaults identical to pre-Phase-D behavior.

Techniques: positive/equivalence, boundary (defaults + coercion),
negative (malformed sub-block degrades, not crashes where the existing
code degrades).
"""
from __future__ import annotations

import pytest

from krakey.models.config import (
    SandboxEnvironmentConfig,
    DockerSandboxSection,
)
from krakey.models.config.environments import _build_sandbox_env


# ---------------------------------------------------------------------------
# 1. Dataclass defaults
# ---------------------------------------------------------------------------

class TestDockerSectionDefaults:
    def test_section_exists_with_defaults(self):
        d = DockerSandboxSection()
        assert d.image == ""
        assert d.container_name == "krakey-sandbox"
        assert d.host_port == 18765
        assert d.host_bind_dirs == []
        assert d.auto_start is False
        assert d.wait_seconds == 30.0

    def test_host_bind_dirs_is_independent_list(self):
        a = DockerSandboxSection()
        b = DockerSandboxSection()
        a.host_bind_dirs.append("/x:/y")
        assert b.host_bind_dirs == []  # no shared mutable default

    def test_sandbox_config_has_docker_field_by_default(self):
        sb = SandboxEnvironmentConfig()
        assert isinstance(sb.docker, DockerSandboxSection)
        assert sb.docker.container_name == "krakey-sandbox"


# ---------------------------------------------------------------------------
# 2. Parsing — omitted block falls back to defaults
# ---------------------------------------------------------------------------

class TestParsingDefaults:
    def test_omitted_docker_block_uses_defaults(self):
        sb = _build_sandbox_env({"guest_os": "linux"})
        assert isinstance(sb.docker, DockerSandboxSection)
        assert sb.docker.image == ""
        assert sb.docker.host_port == 18765

    def test_empty_docker_block_uses_defaults(self):
        sb = _build_sandbox_env({"guest_os": "linux", "docker": {}})
        assert sb.docker.container_name == "krakey-sandbox"
        assert sb.docker.auto_start is False


# ---------------------------------------------------------------------------
# 3. Parsing — explicit values flow through (equivalence)
# ---------------------------------------------------------------------------

class TestParsingExplicit:
    def test_full_docker_block_parsed(self):
        sb = _build_sandbox_env({
            "guest_os": "linux",
            "docker": {
                "image": "krakey/sandbox:latest",
                "container_name": "my-box",
                "host_port": 9999,
                "host_bind_dirs": ["/host:/guest", "/a:/b:ro"],
                "auto_start": True,
                "wait_seconds": 12.5,
            },
        })
        assert sb.docker.image == "krakey/sandbox:latest"
        assert sb.docker.container_name == "my-box"
        assert sb.docker.host_port == 9999
        assert sb.docker.host_bind_dirs == ["/host:/guest", "/a:/b:ro"]
        assert sb.docker.auto_start is True
        assert sb.docker.wait_seconds == 12.5

    def test_partial_docker_block_mixes_with_defaults(self):
        sb = _build_sandbox_env({
            "guest_os": "linux",
            "docker": {"image": "ubuntu:22.04"},
        })
        assert sb.docker.image == "ubuntu:22.04"
        # untouched fields keep defaults
        assert sb.docker.host_port == 18765
        assert sb.docker.container_name == "krakey-sandbox"


# ---------------------------------------------------------------------------
# 4. Boundary / coercion
# ---------------------------------------------------------------------------

class TestCoercion:
    def test_host_port_string_coerced_to_int(self):
        sb = _build_sandbox_env({
            "guest_os": "linux",
            "docker": {"host_port": "4321"},
        })
        assert sb.docker.host_port == 4321
        assert isinstance(sb.docker.host_port, int)

    def test_wait_seconds_int_coerced_to_float(self):
        sb = _build_sandbox_env({
            "guest_os": "linux",
            "docker": {"wait_seconds": 5},
        })
        assert sb.docker.wait_seconds == 5.0
        assert isinstance(sb.docker.wait_seconds, float)

    def test_auto_start_truthy_coerced_to_bool(self):
        sb = _build_sandbox_env({
            "guest_os": "linux",
            "docker": {"auto_start": 1},
        })
        assert sb.docker.auto_start is True

    def test_host_bind_dirs_none_falls_back_to_default_empty(self):
        sb = _build_sandbox_env({
            "guest_os": "linux",
            "docker": {"host_bind_dirs": None},
        })
        assert sb.docker.host_bind_dirs == []


# ---------------------------------------------------------------------------
# 5. Negative — non-mapping docker block degrades to defaults (no crash)
# ---------------------------------------------------------------------------

class TestNegative:
    def test_non_mapping_docker_block_degrades(self):
        # _coerce_mapping turns a non-dict into {} with a warning —
        # docker then falls back entirely to defaults rather than crashing.
        sb = _build_sandbox_env({
            "guest_os": "linux",
            "docker": ["not", "a", "mapping"],
        })
        assert isinstance(sb.docker, DockerSandboxSection)
        assert sb.docker.image == ""
        assert sb.docker.host_port == 18765
