"""Infrastructure-side config: sandbox VM connectivity primitives.

Two leaf dataclasses live here:

  * ``SandboxResourcesSection`` — VM CPU / RAM / disk hints.
  * ``SandboxAgentSection`` — host-only NIC URL + shared token.

Both are consumed by the ``environments.sandbox`` block (see
``models/config/environments.py``); they used to live under a now-
removed top-level ``sandbox:`` block. Kept here as separate
dataclasses so the ``EnvironmentsSection`` shape stays
declaratively assembled.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SandboxResourcesSection:
    cpu: int = 2
    memory_mb: int = 4096
    disk_gb: int = 40


@dataclass
class SandboxAgentSection:
    url: str = "http://10.0.2.10:8765"
    token: str = ""  # shared secret — no safe default; empty token keeps sandbox
                     # disabled-by-default at Router-build time (a non-empty
                     # default would silently enable the sandbox for anyone who
                     # copy-pastes a config without setting their own token)


@dataclass
class DockerSandboxSection:
    """Docker-provider-specific sandbox fields (Phase D).

    Consumed by ``DockerSandboxEnvironment`` (environment node) when
    ``environments.sandbox.provider == "docker"``, and by the env
    composition factory's provider dispatch. Sibling to the QEMU-shaped
    top-level sandbox fields; the ``provider`` discriminant decides
    which set is authoritative at Router-build time.

    ``host_port`` is the host side of the agent port mapping
    (``-p <host_port>:8765``); ``host_bind_dirs`` are verbatim
    ``docker run -v`` strings (e.g. ``"/host:/guest"`` or
    ``"/host:/guest:ro"``).
    """
    image: str = ""
    container_name: str = "krakey-sandbox"
    host_port: int = 18765
    host_bind_dirs: list[str] = field(default_factory=list)
    auto_start: bool = False
    wait_seconds: float = 30.0
