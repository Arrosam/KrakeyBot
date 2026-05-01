"""Infrastructure-side config: sandbox VM.

Describes how the runtime relates to the external sandbox surface (a
guest VM and its agent endpoint). The dashboard used to live here
too, but it became a regular plugin (`krakey/plugins/dashboard/`) and
its config moved to the per-plugin yaml at
`workspace/plugins/dashboard/config.yaml`.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SandboxResourcesSection:
    cpu: int = 2
    memory_mb: int = 4096
    disk_gb: int = 40


@dataclass
class SandboxAgentSection:
    url: str = ""
    token: str = ""


@dataclass
class SandboxSection:
    """Sandbox VM configuration. Required when any plugin that declares
    ``requires_sandbox: true`` in its meta.yaml AND has its own
    ``sandbox: true`` config is enabled. Runtime refuses to start in
    that case if the required fields here are missing or the agent is
    unreachable. (No in-tree plugin currently declares this — the
    sandbox infrastructure is dormant pending the code-execution
    redesign.)
    """
    guest_os: str = ""         # "linux" | "macos" | "windows" — REQUIRED
    provider: str = "qemu"     # qemu | virtualbox | utm
    vm_name: str = ""
    # "headed" — user can see the VM's desktop (spice/sdl/vnc window
    # with a display server). "headless" — VM runs with no display.
    # Declarative only for now: the user launches the VM themselves;
    # this flag documents intent + drives lifecycle tooling later.
    display: str = "headed"    # headed | headless
    resources: SandboxResourcesSection = field(
        default_factory=SandboxResourcesSection
    )
    agent: SandboxAgentSection = field(default_factory=SandboxAgentSection)
    # Network model documentation only; enforced in the VM provisioning,
    # not by this config. Stored for clarity + future tooling.
    network_mode: str = "nat_allowlist"  # nat_allowlist | host_only | isolated
    allowlist_domains: list[str] = field(default_factory=list)


def _build_sandbox(raw: dict[str, Any] | None) -> SandboxSection:
    raw = raw or {}
    d = SandboxSection()
    res_raw = raw.get("resources") or {}
    agent_raw = raw.get("agent") or {}
    display = str(raw.get("display", d.display)).lower()
    if display not in ("headed", "headless"):
        print(
            f"warning: sandbox.display={display!r} not recognised; "
            "falling back to 'headed'. Valid values: headed | headless.",
            file=sys.stderr,
        )
        display = "headed"
    return SandboxSection(
        guest_os=str(raw.get("guest_os", d.guest_os)),
        provider=str(raw.get("provider", d.provider)),
        vm_name=str(raw.get("vm_name", d.vm_name)),
        display=display,
        resources=SandboxResourcesSection(
            cpu=int(res_raw.get("cpu", d.resources.cpu)),
            memory_mb=int(res_raw.get("memory_mb", d.resources.memory_mb)),
            disk_gb=int(res_raw.get("disk_gb", d.resources.disk_gb)),
        ),
        agent=SandboxAgentSection(
            url=str(agent_raw.get("url", d.agent.url)),
            token=str(agent_raw.get("token", d.agent.token)),
        ),
        network_mode=str(raw.get("network_mode", d.network_mode)),
        allowlist_domains=list(raw.get("allowlist_domains")
                                 or d.allowlist_domains),
    )
