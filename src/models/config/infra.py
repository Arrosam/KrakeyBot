"""Infrastructure-side config sections: dashboard server + sandbox VM.

Both describe how the runtime relates to *external* surfaces — the
dashboard exposes a local HTTP server, the sandbox describes a guest
VM and its agent endpoint. Grouped because they're "shape of what
runs alongside Krakey," not core algorithm tuning.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DashboardSection:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8765
    # Ring buffer for the "Prompts" tab. Runtime keeps the last N fully
    # built heartbeat prompts so the UI can show a scrollable log rather
    # than only the single latest one. Per-run, not persisted to disk.
    prompt_log_size: int = 20


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
    """Sandbox VM configuration. Required when any sandboxed tentacle is
    enabled (coding / gui_control / cli / file_read / file_write / browser).
    Runtime refuses to start if any of those has sandbox=true but the
    required fields here are missing or the agent is unreachable.
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


def _build_dashboard(raw: dict[str, Any] | None) -> DashboardSection:
    raw = raw or {}
    d = DashboardSection()
    return DashboardSection(
        enabled=bool(raw.get("enabled", d.enabled)),
        host=str(raw.get("host", d.host)),
        port=int(raw.get("port", d.port)),
        prompt_log_size=max(1, int(raw.get("prompt_log_size",
                                               d.prompt_log_size))),
    )


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
