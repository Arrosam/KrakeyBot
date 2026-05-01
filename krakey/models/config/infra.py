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

from dataclasses import dataclass


@dataclass
class SandboxResourcesSection:
    cpu: int = 2
    memory_mb: int = 4096
    disk_gb: int = 40


@dataclass
class SandboxAgentSection:
    url: str = ""
    token: str = ""
