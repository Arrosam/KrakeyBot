"""Top-level ``environments:`` config block.

Replaces the old top-level ``sandbox:`` section. Two shapes:

  * ``LocalEnvironmentConfig`` — only carries ``allowed_plugins``.
    Local is zero-config; the field exists purely so the user can
    grant specific plugins host-process access.
  * ``SandboxEnvironmentConfig`` — absorbs every field that used
    to live on the top-level ``sandbox:`` block (guest_os, agent,
    resources, display, network_mode, allowlist_domains) PLUS the
    new ``allowed_plugins`` list.

Both are optional. Absent ``environments.local`` ⇒ Local has empty
allow-list (still registered; just denies every plugin). Absent
``environments.sandbox`` ⇒ Sandbox not registered at all.

Top-level form (rather than per-plugin assignment) chosen so the
security-relevant allow-list lives in one auditable block — at a
glance the user can answer "what plugin can use the sandbox VM?".
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from krakey.models.config.infra import (
    SandboxAgentSection, SandboxResourcesSection,
)


@dataclass
class LocalEnvironmentConfig:
    """Allow-list (and nothing else) for the always-on Local env.

    Default empty: no plugin gets host access without explicit
    opt-in. Local is zero-config — there are no other knobs to
    turn here.
    """
    allowed_plugins: list[str] = field(default_factory=list)


@dataclass
class SandboxEnvironmentConfig:
    """Allow-list + the VM connectivity fields the runtime needs to
    talk to the guest agent. ``guest_os`` / ``agent.url`` /
    ``agent.token`` are all REQUIRED when this section is present
    — partial config raises at Router-build time so a typo fails
    fast.
    """
    allowed_plugins: list[str] = field(default_factory=list)
    guest_os: str = ""         # "linux" | "macos" | "windows"
    provider: str = "qemu"     # qemu | virtualbox | utm
    vm_name: str = ""
    display: str = "headed"    # headed | headless (declarative only)
    resources: SandboxResourcesSection = field(
        default_factory=SandboxResourcesSection
    )
    agent: SandboxAgentSection = field(default_factory=SandboxAgentSection)
    # Network model documentation only; enforced in the VM
    # provisioning, not by this config.
    network_mode: str = "nat_allowlist"
    allowlist_domains: list[str] = field(default_factory=list)


@dataclass
class EnvironmentsSection:
    """Top-level ``environments:`` block. Either subsection can
    be omitted — absent Local still has an empty-allow-list Local
    env in the Router; absent Sandbox simply means no sandbox env
    is registered.
    """
    local: LocalEnvironmentConfig = field(
        default_factory=LocalEnvironmentConfig
    )
    sandbox: SandboxEnvironmentConfig | None = None


def _coerce_mapping(value: Any, ctx: str) -> dict[str, Any]:
    """Return ``value`` if it's a dict (or empty dict for None);
    otherwise warn + return empty dict.

    Used at every YAML-mapping boundary in this module — top-level
    ``environments:``, the ``local`` / ``sandbox`` sub-blocks, and
    the deeper ``sandbox.resources`` / ``sandbox.agent`` blocks.
    A non-mapping anywhere in the chain (failed env-var template
    leaving a literal string, user wrote a list by mistake, etc.)
    must not hard-fail boot — degrade with a warning so the rest
    of the config can still load.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        print(
            f"warning: `{ctx}` should be a mapping; got "
            f"{type(value).__name__}; treating as empty.",
            file=sys.stderr,
        )
        return {}
    return value


def _build_environments(
    raw: dict[str, Any] | None,
) -> EnvironmentsSection:
    """Parse the ``environments:`` mapping. Missing / null blocks
    fall back to defaults (empty allow-list for local; no sandbox).

    Non-mapping values at any level get a warning and are treated
    as absent — a config typo shouldn't hard-fail boot.
    """
    raw = _coerce_mapping(raw, "environments")
    local_raw = _coerce_mapping(raw.get("local"), "environments.local")
    local_cfg = LocalEnvironmentConfig(
        allowed_plugins=_clean_allowed(
            local_raw.get("allowed_plugins"), "environments.local",
        ),
    )

    sandbox_raw = raw.get("sandbox")
    sandbox_cfg: SandboxEnvironmentConfig | None
    if sandbox_raw is None:
        sandbox_cfg = None
    elif not isinstance(sandbox_raw, dict):
        print(
            f"warning: `environments.sandbox` should be a mapping; "
            f"got {type(sandbox_raw).__name__}; treating as absent.",
            file=sys.stderr,
        )
        sandbox_cfg = None
    else:
        sandbox_cfg = _build_sandbox_env(sandbox_raw)

    return EnvironmentsSection(local=local_cfg, sandbox=sandbox_cfg)


def _build_sandbox_env(raw: dict[str, Any]) -> SandboxEnvironmentConfig:
    d = SandboxEnvironmentConfig()
    res_raw = _coerce_mapping(
        raw.get("resources"), "environments.sandbox.resources",
    )
    agent_raw = _coerce_mapping(
        raw.get("agent"), "environments.sandbox.agent",
    )
    display = str(raw.get("display", d.display)).lower()
    if display not in ("headed", "headless"):
        print(
            f"warning: environments.sandbox.display={display!r} not "
            "recognised; falling back to 'headed'. Valid values: "
            "headed | headless.",
            file=sys.stderr,
        )
        display = "headed"
    return SandboxEnvironmentConfig(
        allowed_plugins=_clean_allowed(
            raw.get("allowed_plugins"), "environments.sandbox",
        ),
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
        allowlist_domains=list(
            raw.get("allowlist_domains") or d.allowlist_domains
        ),
    )


def _clean_allowed(value: Any, ctx: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        print(
            f"warning: {ctx}.allowed_plugins should be a list; got "
            f"{type(value).__name__}; treating as empty.",
            file=sys.stderr,
        )
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            print(
                f"warning: {ctx}.allowed_plugins entry {item!r} is "
                "not a non-empty string; skipping.",
                file=sys.stderr,
            )
            continue
        out.append(item.strip())
    return out
