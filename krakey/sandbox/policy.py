"""Sandbox policy — pick a CodeRunner + preflight the guest agent.

The runtime composes its plugin context but doesn't make the
sandbox-vs-subprocess decision itself; that's a deployment-policy
question (is this run trusted? is the guest VM up?). Two entry points:

  * ``build_code_runner(coding_cfg, sandbox_cfg)`` — return a
    SubprocessRunner when the coding plugin opts out, else build a
    SandboxRunner from the central sandbox config (refusing to start
    if the config is incomplete).
  * ``preflight_if_required(config)`` — scan ``config.plugins`` for
    plugins that self-declare ``requires_sandbox: true`` in their
    meta.yaml AND have their own ``sandbox`` config field set; if any
    do, ping the guest agent and refuse to start when unreachable.

Both functions live here (not on Runtime) because the runtime owning
this logic put deployment policy on the heartbeat composition root —
unrelated to its job.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from krakey.sandbox.backend import (
    SandboxConfig, SandboxRunner, SandboxUnavailableError, preflight,
)
from krakey.sandbox.subprocess_runner import SubprocessRunner


def build_code_runner(coding_cfg: dict, sandbox_cfg) -> Any:
    """Return Subprocess on sandbox=false, SandboxRunner otherwise.

    Sandbox defaults to TRUE. When any tool enables sandbox but
    the top-level `sandbox` config is incomplete, refuse to start
    with a clear error — user must configure the guest VM first.
    """
    want_sandbox = bool(coding_cfg.get("sandbox", True))
    if not want_sandbox:
        return SubprocessRunner()
    missing = []
    if not sandbox_cfg.guest_os:
        missing.append("sandbox.guest_os")
    if not sandbox_cfg.agent.url:
        missing.append("sandbox.agent.url")
    if not sandbox_cfg.agent.token:
        missing.append("sandbox.agent.token")
    if missing:
        raise RuntimeError(
            "coding.sandbox=true but sandbox is not configured. "
            "Missing: " + ", ".join(missing) + ". "
            "Either complete the `sandbox:` block in config.yaml or "
            "set tool.coding.sandbox=false (unsafe)."
        )
    return SandboxRunner(SandboxConfig(
        agent_url=sandbox_cfg.agent.url,
        agent_token=sandbox_cfg.agent.token,
        guest_os=sandbox_cfg.guest_os,
    ))


async def preflight_if_required(
    config, *, plugin_configs_root: Path | str = "workspace/plugins",
) -> dict[str, Any] | None:
    """Ping the guest agent if any enabled plugin self-declared
    ``requires_sandbox: true`` in its meta.yaml AND has its own
    ``sandbox`` config field on. Refuses to start when the agent is
    unreachable; returns ``None`` when no preflight is needed.

    Iterates ``config.plugins`` (the explicit enabled list) and loads
    each one's meta.yaml by name — no full filesystem scan.
    """
    from krakey.interfaces.plugin_context import load_plugin_config
    from krakey.plugin_system.loader import load_plugin_meta

    cfg_root = Path(plugin_configs_root)
    any_sandboxed = False
    for name in config.plugins or []:
        meta = load_plugin_meta(name)
        if meta is None or not meta.requires_sandbox:
            continue
        # Plugin's own `sandbox: false` opts out (e.g. coding on a
        # trusted host).
        plugin_cfg = load_plugin_config(name, cfg_root)
        if bool(plugin_cfg.get("sandbox", True)):
            any_sandboxed = True
            break
    if not any_sandboxed:
        return None
    sb = config.sandbox
    cfg = SandboxConfig(
        agent_url=sb.agent.url,
        agent_token=sb.agent.token,
        guest_os=sb.guest_os,
    )
    try:
        return await preflight(cfg)
    except SandboxUnavailableError as e:
        raise RuntimeError(
            f"sandbox preflight failed: {e}. "
            "Start the guest agent or disable sandboxed tools."
        )
